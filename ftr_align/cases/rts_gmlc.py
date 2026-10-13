"""RTS-GMLC case: the 73-bus system as the objects ``toy.py`` produces -- a
:class:`PhysicalNetwork`, a :class:`Contingency` list, and a :class:`DamInstance`
-- so the core (``build`` -> ``clear_dam`` / ``SupportProblem``) runs unchanged.

The data comes from the ``rts_gmlc`` package (``~/dev/rts-gmlc``), which pins the
upstream commit, verifies every file against a SHA256 manifest, and serves the
CSVs as polars tables with snake_cased columns. This module makes the modeling
choices on top of that data:

* **DC PTDF with magnitude transformer taps** (``tr_ratio``); RTS has no phase
  shifters, so the tap susceptance scaling is exact.  Shunts / line charging ``b``
  / ``basekv`` are reactive/voltage quantities and do not enter a MW-based DCOPF.
* **Limits**: ``cont_rating`` in the base case (pre-contingency), ``lte_rating``
  under each N-1 outage (steady-state post-contingency).
* **Bids**: the package's ``offer_blocks`` (piecewise-linear step offers from each
  thermal unit's heat-rate segments x fuel price); one ``M_gen`` block-column per
  segment of positive width.
* **Renewables**: PV/RTPV/Wind/Hydro caps pulled from the *same interval* as load,
  as a single zero-cost block per unit, replacing the unit's heat-rate blocks.
* **No unit commitment**: single-period economic dispatch, ``pmin`` relaxed to 0.

TODO(storage): batteries (``unit_type == STORAGE``) need charge/discharge + SOC,
which single-period :class:`DamInstance` cannot express; excluded here.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import rts_gmlc as rts

from ..network import Contingency, PhysicalNetwork, is_connected
from ..solve import DamInstance

#: The pinned upstream commit, for provenance in notebooks and docs.
RTS_GMLC_REF = rts.REF

# Renewable day-ahead caps keyed by gen_uid columns; same interval index as load.
_RENEWABLES = ("pv", "rtpv", "wind", "hydro")


# ----------------------------------------------------------------------------
# Canonical bus ordering: row index <-> bus_id (sorted by bus_id)
# ----------------------------------------------------------------------------
def bus_table() -> tuple[pl.DataFrame, np.ndarray, dict[int, int]]:
    """The bus frame sorted by ``bus_id`` (the canonical matrix row order), the
    id array, and the ``bus_id -> row`` map.  Row ``i`` of every matrix is
    ``ids[i]``; invert with the returned dict to go from a bus id to its index."""
    bus = rts.bus()
    ids = bus["bus_id"].to_numpy()
    return bus, ids, rts.bus_index()


# ----------------------------------------------------------------------------
# Physical network
# ----------------------------------------------------------------------------
def load_network() -> PhysicalNetwork:
    """Build the :class:`PhysicalNetwork` from ``bus`` + ``branch``."""
    bus, ids, id_to_row = bus_table()
    branch = rts.branch()
    n, ell = len(ids), len(branch)

    A = np.zeros((n, ell))
    frm = branch["from_bus"].to_numpy()
    to = branch["to_bus"].to_numpy()
    for j in range(ell):
        A[id_to_row[int(frm[j])], j] = 1.0
        A[id_to_row[int(to[j])], j] = -1.0

    x = branch["x"].to_numpy().astype(float)
    tr = branch["tr_ratio"].to_numpy().astype(float)
    tap = np.where(tr > 0.0, tr, 1.0)  # 0 in the data means "not a transformer"

    ref_rows = np.nonzero(bus["bus_type"].to_numpy() == "Ref")[0]
    if len(ref_rows) != 1:
        raise ValueError(f"expected exactly one Ref bus, found {len(ref_rows)}")

    return PhysicalNetwork(
        A=A,
        x=x,
        slack_idx=int(ref_rows[0]),
        node_names=bus["bus_name"].to_numpy(),
        element_names=branch["uid"].to_numpy(),
        tap=tap,
    )


def _branch_limits() -> dict[str, np.ndarray]:
    """``cont``/``lte`` rating vectors (MW) in element order."""
    branch = rts.branch()
    return {
        "cont": branch["cont_rating"].to_numpy().astype(float),
        "lte": branch["lte_rating"].to_numpy().astype(float),
    }


def n1_contingencies(
    network: PhysicalNetwork | None = None, *, verbose: bool = True
) -> list[Contingency]:
    """Base case (``cont_rating``) + one outage per non-bridge branch enforcing
    ``lte_rating``.  Bridge outages (which would island the network and make the
    PTDF singular) are skipped -- matching ISO practice of excluding radial
    outages from thermal flow constraints -- and logged."""
    network = network or load_network()
    lim = _branch_limits()
    cont, lte = lim["cont"], lim["lte"]
    ell = network.n_elements

    conts = [Contingency(None, cont)]
    skipped: list[int] = []
    for i in range(ell):
        if not is_connected(network.A, i):
            skipped.append(i)
            continue
        upper = lte.copy()
        upper[i] = np.inf  # the outaged element carries no flow -> unmonitored
        conts.append(Contingency(i, upper))

    if verbose and skipped:
        names = network.element_names
        labels = [str(names[i]) for i in skipped] if names is not None else skipped
        print(f"n1_contingencies: skipped {len(skipped)} bridge outage(s): {labels}")
    return conts


# ----------------------------------------------------------------------------
# DAM instance (interval-specific): bids + demand + renewable caps
# ----------------------------------------------------------------------------
def interval_index(month: int, day: int, period: int, year: int = 2020) -> int:
    """Row index of a ``(year, month, day, period)`` hour in the day-ahead files
    (all share the same year/month/day/period index)."""
    return rts.interval_index(month, day, period, year=year, market="day_ahead")


def _renewable_caps(interval: int) -> dict[str, float]:
    """``gen_uid -> available MW`` at ``interval``, across all renewable files."""
    caps: dict[str, float] = {}
    for kind in _RENEWABLES:
        row = rts.timeseries(kind, "day_ahead").row(interval, named=True)
        for col, val in row.items():
            if col not in rts.TIME_COLS:
                caps[col] = float(val)
    return caps


def dam_instance(interval: int, network: PhysicalNetwork | None = None) -> DamInstance:
    """A DAM clearing instance for the given ``interval`` (row index into the
    day-ahead files; use :func:`interval_index` for a calendar lookup).

    Generators expand into offer blocks: renewables -> one zero-cost block at the
    interval cap; thermal -> heat-rate step blocks.  Storage/sync-cond excluded.
    Demand is the regional day-ahead load disaggregated to buses by load share."""
    network = network or load_network()
    bus, ids, id_to_row = bus_table()
    n = network.n_nodes
    caps = _renewable_caps(interval)

    # --- generation offer blocks: (bus row, max MW, marginal cost) ---
    blocks = rts.offer_blocks().filter(pl.col("mw") > 0.0)
    by_unit: dict[str, list[tuple[float, float]]] = {}
    for uid, mw, cost in blocks.select("gen_uid", "mw", "cost").iter_rows():
        by_unit.setdefault(uid, []).append((mw, cost))

    cols: list[tuple[int, float, float]] = []
    for g in rts.gen().iter_rows(named=True):
        unit_type = str(g["unit_type"] or "").upper()
        if "STORAGE" in unit_type or "SYNC" in unit_type:
            continue  # TODO(storage): single-period DamInstance can't model these
        bus_row = id_to_row[int(g["bus_id"])]
        uid = g["gen_uid"]
        if uid in caps:  # renewable: single zero-cost block at the interval cap
            if caps[uid] > 0.0:
                cols.append((bus_row, caps[uid], 0.0))
            continue
        for width, price in by_unit.get(uid, []):
            cols.append((bus_row, width, price))

    n_blocks = len(cols)
    M_gen = np.zeros((n, n_blocks))
    max_gen = np.empty(n_blocks)
    p_gen = np.empty(n_blocks)
    for k, (bus_row, mw, price) in enumerate(cols):
        M_gen[bus_row, k] = 1.0
        max_gen[k] = mw
        p_gen[k] = price

    # --- demand: regional day-ahead load split to buses by load share ---
    load_row = rts.timeseries("load", "day_ahead").row(interval, named=True)
    regional = {a: float(load_row[str(a)]) for a in (1, 2, 3)}
    bus_load = bus["mw_load"].to_numpy().astype(float)
    area = bus["area"].to_numpy().astype(int)
    area_total = {a: bus_load[area == a].sum() for a in regional}

    load_rows = [i for i in range(n) if bus_load[i] > 0.0]
    M_dem = np.zeros((n, len(load_rows)))
    q_dem = np.empty(len(load_rows))
    for k, i in enumerate(load_rows):
        a = int(area[i])
        M_dem[i, k] = 1.0
        q_dem[k] = regional[a] * bus_load[i] / area_total[a]

    return DamInstance(
        M_gen=M_gen,
        M_dem=M_dem,
        min_gen=np.zeros(n_blocks),  # PMin relaxed: no unit commitment
        max_gen=max_gen,
        p_gen=p_gen,
        q_dem=q_dem,
    )
