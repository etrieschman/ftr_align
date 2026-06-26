"""RTS-GMLC loader: a hand-rolled parse of the RTS-GMLC source CSVs into the same
objects ``toy.py`` produces -- a :class:`PhysicalNetwork`, a :class:`Contingency`
list, and a :class:`DamInstance` -- so the core (``build`` -> ``clear_dam`` /
``SupportProblem``) runs unchanged on the 73-bus system.

Data provenance is fully pinned: every CSV is fetched from a fixed RTS-GMLC commit
(:data:`RTS_GMLC_REF`) into a gitignored cache and verified byte-for-byte against
:data:`MANIFEST`.  ``master`` moving upstream cannot change results, and another
machine reproduces the identical dataset.

Modeling choices (see the plan / CLAUDE.md):

* **DC PTDF with magnitude transformer taps** (``Tr Ratio``); RTS has no phase
  shifters, so the tap susceptance scaling is exact.  Shunts / line charging ``B``
  / ``BaseKV`` are reactive/voltage quantities and do not enter a MW-based DCOPF.
* **Limits**: ``Cont Rating`` in the base case (pre-contingency), ``LTE Rating``
  under each N-1 outage (steady-state post-contingency).
* **Bids**: piecewise-linear step offers from each thermal unit's incremental
  heat-rate segments x fuel price; one ``M_gen`` block-column per segment.
* **Renewables**: PV/RTPV/Wind/Hydro caps pulled from the *same interval* as load,
  as a single zero-cost block per unit.
* **No unit commitment**: single-period economic dispatch, ``PMin`` relaxed to 0.

TODO(storage): batteries (``Unit Type == STORAGE``) need charge/discharge + SOC,
which single-period :class:`DamInstance` cannot express; excluded here.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

from ..network import Contingency, PhysicalNetwork, is_connected
from ..solve import DamInstance

# Pinned RTS-GMLC commit -> byte-frozen data (content-addressed; immune to master).
RTS_GMLC_REF = "3ece0d3725c844056132393ee252b3083dd4eab4"
_RAW = "https://raw.githubusercontent.com/GridMod/RTS-GMLC/{ref}/RTS_Data/{relpath}"
_CACHE = Path(__file__).resolve().parent / "data" / "rts_gmlc"

_BUS = "SourceData/bus.csv"
_BRANCH = "SourceData/branch.csv"
_GEN = "SourceData/gen.csv"
_LOAD = "timeseries_data_files/Load/DAY_AHEAD_regional_Load.csv"
# Renewable day-ahead caps, keyed by GEN UID columns; same interval index as load.
_RENEWABLES = {
    "PV": "timeseries_data_files/PV/DAY_AHEAD_pv.csv",
    "RTPV": "timeseries_data_files/RTPV/DAY_AHEAD_rtpv.csv",
    "WIND": "timeseries_data_files/WIND/DAY_AHEAD_wind.csv",
    "Hydro": "timeseries_data_files/Hydro/DAY_AHEAD_hydro.csv",
}
_TIME_COLS = ("Year", "Month", "Day", "Period")

# SHA256 of each fetched file at RTS_GMLC_REF; verified on every read.  Populated
# by ``python -m ftr_align.cases.rts_gmlc`` (see __main__ below).
MANIFEST: dict[str, str] = {
    _BUS: "cec3f776222812d43eaeaf7ac85b577719dc565f354a04e120c12a647ddb710e",
    _BRANCH: "e92d16d13b1c5d2899f7871c0c99c3eb94605528beef157bf9e1ccbcffe5ef4e",
    _GEN: "988466f29132b73739de60c9204dd4a2a9ceb0adf572e5966c086611272f4068",
    _LOAD: "7a9470d32d49068a91334cb36db54cceb2feb5cb1f702b0fa0847af8bac6cf21",
    _RENEWABLES["PV"]: "bfede6e558df5ea0f244b6326940a4ee0b95138643aa8a062897c67134c9c185",
    _RENEWABLES["RTPV"]: "13a6933c2e0a513e1a453143876dadef6977e6add7701a21f56fe6a753afce42",
    _RENEWABLES["WIND"]: "6a1a8dc7d10a518523b3b319902ecc1ca1c400223832b26c6edb3a6e69d01dbc",
    _RENEWABLES["Hydro"]: "4030660920df850138472c5561322c71e5037813c8e3232d3f9bde512a40606d",
}


# ----------------------------------------------------------------------------
# Fetch + verify
# ----------------------------------------------------------------------------
def _fetch(relpath: str) -> Path:
    """Download ``relpath`` from the pinned ref into the cache (once) and verify
    its SHA256 against :data:`MANIFEST` (when a hash is recorded)."""
    dest = _CACHE / relpath
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = _RAW.format(ref=RTS_GMLC_REF, relpath=relpath)
        urllib.request.urlretrieve(url, dest)
    want = MANIFEST.get(relpath)
    if want:
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got != want:
            raise ValueError(
                f"checksum mismatch for {relpath}: got {got}, expected {want}. "
                f"Delete {dest} to re-fetch from {RTS_GMLC_REF}."
            )
    return dest


def _read_csv(relpath: str) -> pl.DataFrame:
    # "NA" appears in unused heat-rate segments -> parse as null, not string.
    # infer_schema_length=None scans the whole file (gen.csv has int-looking early
    # rows, e.g. Output_pct_3=1, then 0.8 later) to pick float dtypes correctly.
    return pl.read_csv(_fetch(relpath), null_values=["NA"], infer_schema_length=None)


# ----------------------------------------------------------------------------
# Canonical bus ordering: row index <-> Bus ID (sorted by Bus ID)
# ----------------------------------------------------------------------------
def bus_table() -> tuple[pl.DataFrame, np.ndarray, dict[int, int]]:
    """The bus frame sorted by ``Bus ID`` (the canonical matrix row order), the
    id array, and the ``bus_id -> row`` map.  Row ``i`` of every matrix is
    ``ids[i]``; invert with the returned dict to go from a Bus ID to its index."""
    bus = _read_csv(_BUS).sort("Bus ID")
    ids = bus["Bus ID"].to_numpy()
    return bus, ids, {int(b): i for i, b in enumerate(ids)}


# ----------------------------------------------------------------------------
# Physical network
# ----------------------------------------------------------------------------
def load_network() -> PhysicalNetwork:
    """Build the :class:`PhysicalNetwork` from ``bus.csv`` + ``branch.csv``."""
    bus, ids, id_to_row = bus_table()
    branch = _read_csv(_BRANCH)
    n, ell = len(ids), len(branch)

    inc = np.zeros((n, ell))
    frm = branch["From Bus"].to_numpy()
    to = branch["To Bus"].to_numpy()
    for j in range(ell):
        inc[id_to_row[int(frm[j])], j] = 1.0
        inc[id_to_row[int(to[j])], j] = -1.0

    x = branch["X"].to_numpy().astype(float)
    tr = branch["Tr Ratio"].to_numpy().astype(float)
    tap = np.where(tr > 0.0, tr, 1.0)  # 0 in the data means "not a transformer"

    ref_rows = np.nonzero(bus["Bus Type"].to_numpy() == "Ref")[0]
    if len(ref_rows) != 1:
        raise ValueError(f"expected exactly one Ref bus, found {len(ref_rows)}")

    return PhysicalNetwork(
        inc=inc,
        x=x,
        slack_idx=int(ref_rows[0]),
        node_names=bus["Bus Name"].to_numpy(),
        element_names=branch["UID"].to_numpy(),
        tap=tap,
    )


def branch_limits() -> dict[str, np.ndarray]:
    """``cont``/``lte`` rating vectors (MW) in element order."""
    branch = _read_csv(_BRANCH)
    return {
        "cont": branch["Cont Rating"].to_numpy().astype(float),
        "lte": branch["LTE Rating"].to_numpy().astype(float),
    }


def n1_contingencies(
    network: PhysicalNetwork | None = None, *, verbose: bool = True
) -> list[Contingency]:
    """Base case (``Cont Rating``) + one outage per non-bridge branch enforcing
    ``LTE Rating``.  Bridge outages (which would island the network and make the
    PTDF singular) are skipped -- matching ISO practice of excluding radial
    outages from thermal flow constraints -- and logged."""
    network = network or load_network()
    lim = branch_limits()
    cont, lte = lim["cont"], lim["lte"]
    ell = network.n_elements

    conts = [Contingency(None, cont)]
    skipped: list[int] = []
    for i in range(ell):
        if not is_connected(network.inc, i):
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
    (all share the same Year/Month/Day/Period index)."""
    df = _read_csv(_LOAD).with_row_index("__i")
    sel = df.filter(
        (pl.col("Year") == year)
        & (pl.col("Month") == month)
        & (pl.col("Day") == day)
        & (pl.col("Period") == period)
    )
    if len(sel) == 0:
        raise ValueError(f"no interval for {year}-{month:02d}-{day:02d} period {period}")
    return int(sel["__i"][0])


def _renewable_caps(interval: int) -> dict[str, float]:
    """``GEN UID -> available MW`` at ``interval``, across all renewable files."""
    caps: dict[str, float] = {}
    for relpath in _RENEWABLES.values():
        row = _read_csv(relpath).row(interval, named=True)
        for col, val in row.items():
            if col not in _TIME_COLS:
                caps[col] = float(val)
    return caps


def _thermal_blocks(g: dict) -> list[tuple[float, float]]:
    """Piecewise-linear step offer for a thermal unit: ``[(block width MW, marginal
    cost $/MWh), ...]``.  Heat input is PWL in output with breakpoints
    ``Output_pct_k * PMax``; the first segment's slope is ``HR_avg_0`` and segment
    ``k``'s is ``HR_incr_k`` (BTU/kWh -> /1000 -> MMBTU/MWh), priced at
    ``slope * Fuel Price + VOM``.  Blocks are independent LP columns, so the LP
    realizes the lower convex envelope regardless of order."""
    pmax = float(g["PMax MW"])
    fuel_price = float(g["Fuel Price $/MMBTU"] or 0.0)
    vom = float(g["VOM"] or 0.0)
    if pmax <= 0.0:
        return []

    pcts = [g[f"Output_pct_{k}"] for k in range(5)]
    mw = [float(p) * pmax for p in pcts if p is not None]
    if len(mw) < 1:
        return []

    blocks: list[tuple[float, float]] = []
    hr_avg0 = float(g["HR_avg_0"] or 0.0)
    if mw[0] > 0.0:
        blocks.append((mw[0], hr_avg0 / 1000.0 * fuel_price + vom))
    for k in range(1, len(mw)):
        hr_incr = g[f"HR_incr_{k}"]
        if hr_incr is None:
            continue
        blocks.append((mw[k] - mw[k - 1], float(hr_incr) / 1000.0 * fuel_price + vom))
    return blocks


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
    cols: list[tuple[int, float, float]] = []
    for g in _read_csv(_GEN).iter_rows(named=True):
        unit_type = str(g["Unit Type"] or "").upper()
        if "STORAGE" in unit_type or "SYNC" in unit_type:
            continue  # TODO(storage): single-period DamInstance can't model these
        bus_row = id_to_row[int(g["Bus ID"])]
        uid = g["GEN UID"]
        if uid in caps:  # renewable: single zero-cost block at the interval cap
            if caps[uid] > 0.0:
                cols.append((bus_row, caps[uid], 0.0))
            continue
        for width, price in _thermal_blocks(g):
            if width > 0.0:
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
    load_row = _read_csv(_LOAD).row(interval, named=True)
    regional = {a: float(load_row[str(a)]) for a in (1, 2, 3)}
    bus_load = bus["MW Load"].to_numpy().astype(float)
    area = bus["Area"].to_numpy().astype(int)
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


# ----------------------------------------------------------------------------
# Manifest bootstrap:  python -m ftr_align.cases.rts_gmlc
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Fetching RTS-GMLC @ {RTS_GMLC_REF} and hashing ...")
    for rel in MANIFEST:
        path = _fetch(rel)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f'    "{rel}": "{digest}",')
