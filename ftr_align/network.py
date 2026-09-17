"""Network geometry: incidence, PTDF, and the models built on them.

A :class:`NetworkModel` is a constraint system on a node set,

    K = [H; -H]        b = [upper; lower]        Q = {q : K q <= b, 1^T q = 0}

stored one contingency at a time: each :class:`ContingencyRows` carries its own
PTDF rows ``H_c`` and limits, and ``H``, ``K`` and ``b`` are assembled from them on
first use.  Nothing downstream needs more than ``(K, b)`` and the node set; the
physical network is one way to *build* the rows (:meth:`NetworkModel.build`), not
part of what a model is.  So two models built on different networks -- different
elements, shift factors or reference buses -- are comparable as long as they share
nodes, and :func:`intersection` is just the stack.

Keeping the contingencies separate is for scale.  It lets an intersection share
its parents' arrays instead of copying them, and it is the seam where rows can
later be generated or screened per contingency rather than materialised as one
dense ``K``.

``b`` is a full-length vector over the model's rows, ``+inf`` where a row is
unmonitored.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np
import polars as pl

# A contingency key identifies which elements are out:  None -> base case
# (nothing out);  int -> one outaged element;  tuple[int, ...] -> several.
ContingencyKey = int | tuple[int, ...] | None


# ----------------------------------------------------------------------------
# PTDF and physical topology
# ----------------------------------------------------------------------------
def is_connected(A: np.ndarray, key: ContingencyKey = None) -> bool:
    """Whether the network stays connected with ``outaged`` removed.

    Guards against a radial outage, which islands the network and makes the PTDF
    singular.
    """
    A = np.asarray(A, dtype=float)
    n, ell = A.shape
    if key is not None:
        out = [int(key)] if isinstance(key, (int, np.integer)) else list(key)
        A = A.copy()
        A[:, out] = 0.0

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for j in range(ell):
        nz = np.nonzero(A[:, j])[0]
        for k in nz[1:]:
            ra, rk = find(int(nz[0])), find(int(k))
            if ra != rk:
                parent[ra] = rk

    return len({find(i) for i in range(n)}) == 1


def compute_ptdf(
    A: np.ndarray, x: np.ndarray, slack_idx: int, tap: np.ndarray | None = None
) -> np.ndarray:
    """DC PTDF: sensitivity of each element's flow to each nodal injection.

    Susceptance is ``1/(x * tap)``, so a magnitude transformer tap scales it.
    """
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)
    n = A.shape[0]
    tap = np.ones(A.shape[1]) if tap is None else np.asarray(tap, dtype=float)

    if not is_connected(A):
        raise ValueError(
            "network is disconnected (islanded): the reduced bus-susceptance "
            "matrix is singular.  An N-1 outage of a bridge element causes this; "
            "such outages should be filtered out with is_connected before solving."
        )

    y_line = np.diag(1.0 / (x * tap))
    y_bus = A @ y_line @ A.T
    keep = np.delete(np.eye(n), slack_idx, axis=0)  # drop slack row
    return y_line @ A.T @ keep.T @ np.linalg.inv(keep @ y_bus @ keep.T) @ keep


@dataclass(frozen=True)
class PhysicalNetwork:
    """Physical topology common to every contingency."""

    A: np.ndarray  # (n, ell) node-branch incidence, node x line
    x: np.ndarray  # (ell,) reactances
    slack_idx: int = -1
    node_names: np.ndarray | None = None
    element_names: np.ndarray | None = None
    tap: np.ndarray | None = None  # (ell,) off-nominal magnitude ratios; None -> ones

    @property
    def n_nodes(self) -> int:
        return self.A.shape[0]

    @property
    def n_elements(self) -> int:
        return self.A.shape[1]

    def ptdf(self, key: ContingencyKey = None) -> np.ndarray:
        """PTDF ``H_c`` with the contingency's outaged elements removed (their
        incidence columns zeroed, so they carry no flow).  ``key`` is a
        contingency key: ``None`` (base), an ``int`` element, or a tuple of
        element indices."""
        A = np.array(self.A, dtype=float, copy=True)
        if key is not None:
            out = [int(key)] if isinstance(key, (int, np.integer)) else list(key)
            A[:, out] = 0.0
        return compute_ptdf(A, self.x, self.slack_idx, self.tap)


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
# Contingency: the builder's input -- a key and the limits enforced under it
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Contingency:
    """A contingency (its ``key``) and the per-element flow limits enforced under
    it -- the input to :meth:`NetworkModel.build`.  Pass a single ``upper`` for
    symmetric limits; give ``lower`` only when it differs.  Use ``+inf`` to leave
    an element unmonitored under this contingency."""

    key: ContingencyKey
    upper: np.ndarray  # (ell,)
    lower: np.ndarray | None = None  # (ell,); defaults to upper (symmetric)

    def __post_init__(self) -> None:
        upper = np.asarray(self.upper, dtype=float)
        lower = upper if self.lower is None else np.asarray(self.lower, dtype=float)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "lower", lower)


# ----------------------------------------------------------------------------
# Network model: constraint rows on a node set, one contingency at a time
# ----------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ContingencyRows:
    """One contingency's rows of a model: the PTDF rows ``H_c`` of its elements
    under that contingency, with their upper and lower flow limits.  Row ``i``
    contributes ``H_c[i] q <= upper[i]`` and ``-H_c[i] q <= lower[i]``."""

    key: Hashable  # identifies the contingency within a model; None is the base case
    label: str
    elements: tuple[str, ...]  # (r,)
    H: np.ndarray  # (r, n)
    upper: np.ndarray  # (r,)
    lower: np.ndarray  # (r,)


@dataclass(frozen=True, eq=False)
class NetworkModel:
    """A network model: constraint rows over a node set.

    ``K = [H; -H]`` with ``H`` the contingencies' PTDF rows stacked in order, and
    ``b = [upper; lower]`` co-indexed with it.  Any per-row vector (a certificate
    ``y``, duals ``mu``) lines up entrywise with ``b``.  Build one from a physical
    network with :meth:`build`, or from two others with :func:`intersection`."""

    nodes: tuple[str, ...]
    contingencies: tuple[ContingencyRows, ...]

    @classmethod
    def build(
        cls, network: PhysicalNetwork, contingencies: Iterable[Contingency]
    ) -> NetworkModel:
        names = network.element_names
        elements = tuple(element_label(names, i) for i in range(network.n_elements))
        nodes = (
            tuple(str(v) for v in network.node_names)
            if network.node_names is not None
            else tuple(str(i) for i in range(network.n_nodes))
        )
        rows = tuple(
            ContingencyRows(
                key=c.key,
                label=contingency_label(c.key, names),
                elements=elements,
                H=network.ptdf(c.key),
                upper=c.upper,
                lower=c.lower,
            )
            for c in contingencies
        )
        return cls(nodes=nodes, contingencies=rows)

    # -- the constraint system, assembled on first use ------------------------
    @cached_property
    def H(self) -> np.ndarray:
        """Stacked PTDF -- the upper half of ``K``, contingencies in order."""
        return np.vstack([c.H for c in self.contingencies])

    @cached_property
    def K(self) -> np.ndarray:
        return np.vstack([self.H, -self.H])

    @cached_property
    def b(self) -> np.ndarray:
        return np.concatenate(
            [c.upper for c in self.contingencies] + [c.lower for c in self.contingencies]
        )

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_rows(self) -> int:
        return 2 * sum(len(c.upper) for c in self.contingencies)

    @property
    def active(self) -> np.ndarray:
        """Rows with finite limits (monitored)."""
        return np.isfinite(self.b)

    @property
    def keys(self) -> list[Hashable]:
        return [c.key for c in self.contingencies]

    # -- row lookup -----------------------------------------------------------
    def _offset(self, key: Hashable) -> tuple[int, int]:
        """Start of ``key``'s rows in ``H`` and their count.  Raises if ``key`` is
        absent or appears more than once, as it can in an intersection."""
        hits = [i for i, c in enumerate(self.contingencies) if c.key == key]
        if len(hits) != 1:
            raise KeyError(
                f"contingency {key!r} appears {len(hits)} times in this model; "
                "look rows up by label instead (see `labels`)."
            )
        start = sum(len(c.upper) for c in self.contingencies[: hits[0]])
        return start, len(self.contingencies[hits[0]].upper)

    def rows_upper(self, key: Hashable) -> np.ndarray:
        start, r = self._offset(key)
        return np.arange(start, start + r)

    def rows_lower(self, key: Hashable) -> np.ndarray:
        start, r = self._offset(key)
        return np.arange(self.n_rows // 2 + start, self.n_rows // 2 + start + r)

    def labels(self) -> pl.DataFrame:
        """Per-constraint identity -- ``constraint`` (the row index into ``K``/
        ``b``/``mu``) with its ``(contingency, element, side)`` -- for output
        tables.  Each row of ``K`` is one constraint ``K[i] q <= b[i]``."""
        return self._labels

    @cached_property
    def _labels(self) -> pl.DataFrame:
        conts = [c.label for c in self.contingencies for _ in c.elements]
        elems = [e for c in self.contingencies for e in c.elements]
        half = len(elems)
        return pl.DataFrame(
            {
                "constraint": np.arange(2 * half),
                "contingency": conts * 2,
                "element": elems * 2,
                "side": ["upper"] * half + ["lower"] * half,
            }
        )


def with_limits(model: NetworkModel, b: np.ndarray) -> NetworkModel:
    """``model`` with limit vector ``b``: same rows, same ``H`` arrays (shared, not
    copied), new ``upper``/``lower`` per contingency."""
    b = np.asarray(b, dtype=float)
    half, start, rows = model.n_rows // 2, 0, []
    for c in model.contingencies:
        r = len(c.upper)
        rows.append(
            replace(c, upper=b[start : start + r], lower=b[half + start : half + start + r])
        )
        start += r
    return NetworkModel(nodes=model.nodes, contingencies=tuple(rows))


def intersection(*models: NetworkModel) -> NetworkModel:
    """The intersection model: ``Q = Q(m_1) inter ... inter Q(m_k)``.

    The constraints of every model, stacked -- ``K = [K_1; ...; K_k]`` up to the
    ordering of rows, and ``b`` likewise.  The only requirement is a common node
    set; the models may differ in elements, shift factors and reference bus.  A
    contingency both models enforce appears once per model, and the tighter limit
    binds.  The rows keep their parents' ``H`` arrays, so nothing is copied until
    ``K`` is assembled.
    """
    nodes = models[0].nodes
    for m in models[1:]:
        if m.nodes != nodes:
            raise ValueError(
                "intersection needs models on one node set, in one order; got "
                f"{nodes} and {m.nodes}."
            )
    return NetworkModel(
        nodes=nodes, contingencies=tuple(c for m in models for c in m.contingencies)
    )
