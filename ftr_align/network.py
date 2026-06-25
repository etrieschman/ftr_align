"""Network geometry: PTDF, contingencies (each carrying its own line limits),
and the network model that assembles them into the stacked constraint system.

Notation follows the FTR pitch memo:

* ``K``  -- PTDF, maps nodal injections ``q`` to monitored flows.
* ``A = [K; -K]`` -- the stacked constraint matrix.  The first ``C*ell`` rows are
  the upper-limit constraints ``Kq <= b_upper``; the next ``C*ell`` are the
  lower-limit constraints ``-Kq <= -b_lower``.  Rows within a block run over
  ``(contingency, element)`` in contingency order, then element.

A ``NetworkModel`` owns its geometry: a network plus a list of contingencies,
each of which carries the line limits enforced *under that contingency*.  It
derives ``A`` and the limit vector ``b``.  DAM and FTR are two such models,
defined independently; comparison of their per-row duals needs :func:`embed` /
:func:`align` to put them on a common row index (see those for the caveat that
this requires common PTDFs).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl

# A contingency key identifies which elements are out:  None -> base case
# (nothing out);  int -> one outaged element;  tuple[int, ...] -> several.
ContingencyKey = int | tuple[int, ...] | None


# ----------------------------------------------------------------------------
# PTDF and physical topology
# ----------------------------------------------------------------------------
def compute_ptdf(inc: np.ndarray, x: np.ndarray, slack_idx: int) -> np.ndarray:
    """DC PTDF ``K`` (``(ell, n)``) for incidence ``inc`` (``(n, ell)``, entries
    in ``{-1, 0, +1}``), reactances ``x`` (``(ell,)``), and a slack bus."""
    inc = np.asarray(inc, dtype=float)
    x = np.asarray(x, dtype=float)
    n = inc.shape[0]

    y_line = np.diag(1.0 / x)
    y_bus = inc @ y_line @ inc.T
    keep = np.delete(np.eye(n), slack_idx, axis=0)  # drop slack row
    return y_line @ inc.T @ keep.T @ np.linalg.inv(keep @ y_bus @ keep.T) @ keep


@dataclass(frozen=True)
class PhysicalNetwork:
    """Physical topology common to every contingency."""

    inc: np.ndarray  # (n, ell) incidence, node x line
    x: np.ndarray  # (ell,) reactances
    slack_idx: int = -1
    node_names: np.ndarray | None = None
    element_names: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return self.inc.shape[0]

    @property
    def n_elements(self) -> int:
        return self.inc.shape[1]

    def ptdf(self, key: ContingencyKey = None) -> np.ndarray:
        """PTDF with the contingency's outaged elements removed (their incidence
        columns zeroed, so they carry no flow).  ``key`` is a contingency key:
        ``None`` (base), an ``int`` element, or a tuple of element indices."""
        inc = np.array(self.inc, dtype=float, copy=True)
        if key is not None:
            out = [int(key)] if isinstance(key, (int, np.integer)) else list(key)
            inc[:, out] = 0.0
        return compute_ptdf(inc, self.x, self.slack_idx)


def contingency_label(key: ContingencyKey, element_names=None) -> str:
    """Display label for a contingency key (``"base"``, an element name, or the
    raw key for multi-element contingencies)."""
    if key is None:
        return "base"
    if element_names is not None and isinstance(key, (int, np.integer)):
        return str(element_names[key])
    return str(key)


def element_label(element_names, i: int) -> str:
    """Display label for element ``i`` -- its name if available, else its index."""
    return str(element_names[i]) if element_names is not None else str(i)


# ----------------------------------------------------------------------------
# Contingency: a contingency key together with the limits enforced under it
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Contingency:
    """A contingency (its ``key``) and the per-element flow limits enforced under
    it.  Pass a single ``upper`` for symmetric limits; give ``lower`` only when
    it differs.  Use ``+inf`` to leave an element unmonitored under this
    contingency."""

    key: ContingencyKey
    upper: np.ndarray  # (ell,)
    lower: np.ndarray | None = None  # (ell,); defaults to upper (symmetric)

    def __post_init__(self) -> None:
        upper = np.asarray(self.upper, dtype=float)
        lower = upper if self.lower is None else np.asarray(self.lower, dtype=float)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "lower", lower)


# ----------------------------------------------------------------------------
# Network model: geometry (A) + limits (b), assembled from contingencies
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class NetworkModel:
    """A network model owns its geometry.  Build it with :meth:`build` from a
    network and a list of :class:`Contingency`; it assembles ``A = [K; -K]`` and
    the stacked limit vector ``b``.  ``b`` and any per-row vector (a certificate
    ``y``, duals ``mu``) line up entrywise over the rows of ``A``."""

    network: PhysicalNetwork
    contingencies: tuple[Contingency, ...]
    A: np.ndarray  # (2 * C * ell, n), dense
    b: np.ndarray  # (2 * C * ell,) limits; +inf marks an unmonitored row

    @classmethod
    def build(
        cls, network: PhysicalNetwork, contingencies: Iterable[Contingency]
    ) -> NetworkModel:
        conts = tuple(contingencies)
        k = np.vstack([network.ptdf(c.key) for c in conts])
        A = np.vstack([k, -k])
        b = np.concatenate(
            [np.concatenate([c.upper for c in conts]),
             np.concatenate([c.lower for c in conts])]
        )
        return cls(network=network, contingencies=conts, A=A, b=b)

    @property
    def keys(self) -> list[ContingencyKey]:
        return [c.key for c in self.contingencies]

    @property
    def ell(self) -> int:
        return self.network.n_elements

    @property
    def n_rows(self) -> int:
        return self.A.shape[0]

    @property
    def active(self) -> np.ndarray:
        """Rows with finite limits (monitored)."""
        return np.isfinite(self.b)

    def rows_upper(self, key: ContingencyKey) -> np.ndarray:
        s = self.keys.index(key) * self.ell
        return np.arange(s, s + self.ell)

    def rows_lower(self, key: ContingencyKey) -> np.ndarray:
        half = len(self.contingencies) * self.ell
        s = self.keys.index(key) * self.ell
        return np.arange(half + s, half + s + self.ell)

    def labels(self) -> pl.DataFrame:
        """Per-row identity (contingency, element, side) -- for output tables."""
        ell = self.ell
        names = self.network.element_names
        conts = [contingency_label(c.key, names) for c in self.contingencies for _ in range(ell)]
        elems = [
            element_label(names, i) for _ in self.contingencies for i in range(ell)
        ]
        return pl.DataFrame(
            {
                "row": np.arange(self.n_rows),
                "contingency": conts * 2,
                "element": elems * 2,
                "side": ["upper"] * (self.n_rows // 2) + ["lower"] * (self.n_rows // 2),
            }
        )


# ----------------------------------------------------------------------------
# Result-conversion tools: put two models' per-row vectors on a common index
# ----------------------------------------------------------------------------
def embed(
    values: np.ndarray, source: NetworkModel, target: NetworkModel, fill: float = 0.0
) -> np.ndarray:
    """Re-express a per-row vector (a certificate ``y`` or a dual ``mu``) from
    ``source`` rows onto ``target`` rows, matching by ``(contingency, element,
    side)``.  Target rows absent from source get ``fill`` (0 for ``y``/``mu``).

    Needed only for *row-level* cross-model comparison (e.g. lining up
    ``mu_f`` and ``mu_g``); support values and the gap use the node-space
    direction and need no alignment.  Valid only under common PTDFs.
    """
    out = np.full(target.n_rows, fill)
    source_keys = set(source.keys)
    for key in target.keys:
        if key in source_keys:
            out[target.rows_upper(key)] = values[source.rows_upper(key)]
            out[target.rows_lower(key)] = values[source.rows_lower(key)]
    return out


def align(*models: NetworkModel) -> list[NetworkModel]:
    """Rebuild several models onto one common (union) contingency set so their
    rows line up entrywise.  Contingencies a model does not enforce are added
    with ``+inf`` limits (unmonitored).  Used for row-level attribution
    comparison; not required to compute support values or the gap."""
    network = models[0].network
    union: list[ContingencyKey] = []
    for model in models:
        for key in model.keys:
            if key not in union:
                union.append(key)

    ell = network.n_elements
    out = []
    for model in models:
        by_key = {c.key: c for c in model.contingencies}
        conts = [
            by_key.get(key, Contingency(key, np.full(ell, np.inf)))
            for key in union
        ]
        out.append(NetworkModel.build(network, conts))
    return out
