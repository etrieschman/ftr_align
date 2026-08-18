"""Network geometry: incidence, PTDF, and the models built on them.

A :class:`NetworkModel` owns a :class:`PhysicalNetwork` and a tuple of
:class:`Contingency`, and assembles the stacked system

    K = [H; -H]        b = [upper; lower]        Q(b) = {q : K q <= b, 1^T q = 0}

``b`` is a full-length vector over the model's rows, ``+inf`` where a row is
unmonitored.
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
# Network model: geometry (K) + limits (b), assembled from contingencies
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class NetworkModel:
    """A network model owns its geometry.  Build it with :meth:`build` from a
    network and a list of :class:`Contingency`; it assembles ``K = [H; -H]`` and
    the stacked limit vector ``b``.  ``b`` and any per-row vector (a certificate
    ``y``, duals ``mu``) line up entrywise over the rows of ``K``."""

    network: PhysicalNetwork
    contingencies: tuple[Contingency, ...]
    K: np.ndarray  # (2 * C * ell, n), dense
    b: np.ndarray  # (2 * C * ell,) limits; +inf marks an unmonitored row

    @classmethod
    def build(
        cls, network: PhysicalNetwork, contingencies: Iterable[Contingency]
    ) -> NetworkModel:
        conts = tuple(contingencies)
        H = np.vstack([network.ptdf(c.key) for c in conts])
        K = np.vstack([H, -H])
        b = np.concatenate(
            [
                np.concatenate([c.upper for c in conts]),
                np.concatenate([c.lower for c in conts]),
            ]
        )
        return cls(network=network, contingencies=conts, K=K, b=b)

    @property
    def keys(self) -> list[ContingencyKey]:
        return [c.key for c in self.contingencies]

    @property
    def ell(self) -> int:
        return self.network.n_elements

    @property
    def n_rows(self) -> int:
        return self.K.shape[0]

    @property
    def H(self) -> np.ndarray:
        """Stacked PTDF -- the upper half of ``K = [H; -H]``, one block of ``ell``
        rows per contingency in :attr:`keys` order."""
        return self.K[: self.n_rows // 2]

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
        """Per-constraint identity -- ``constraint`` (the row index into ``K``/
        ``b``/``mu``) with its ``(contingency, element, side)`` -- for output
        tables.  Each row of ``K`` is one constraint ``K[i] q <= b[i]``."""
        ell = self.ell
        names = self.network.element_names
        conts = [
            contingency_label(c.key, names)
            for c in self.contingencies
            for _ in range(ell)
        ]
        elems = [
            element_label(names, i) for _ in self.contingencies for i in range(ell)
        ]
        return pl.DataFrame(
            {
                "constraint": np.arange(self.n_rows),
                "contingency": conts * 2,
                "element": elems * 2,
                "side": ["upper"] * (self.n_rows // 2) + ["lower"] * (self.n_rows // 2),
            }
        )


# ----------------------------------------------------------------------------
# Result-conversion tools: put two models' per-row vectors on a common index
# ----------------------------------------------------------------------------
def with_limits(model: NetworkModel, b: np.ndarray) -> NetworkModel:
    """``model`` with new limits, rebuilding its ``Contingency`` objects too.

    Necessary because ``align`` reads limits off that list, so replacing only ``b``
    would leave a repaired model re-aligning to its old values.
    """
    b = np.asarray(b, dtype=float)
    half, ell = model.n_rows // 2, model.ell
    conts = tuple(
        Contingency(
            c.key,
            upper=b[i * ell : (i + 1) * ell],
            lower=b[half + i * ell : half + (i + 1) * ell],
        )
        for i, c in enumerate(model.contingencies)
    )
    return NetworkModel(
        network=model.network, contingencies=conts, K=model.K, b=b
    )


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
            by_key.get(key, Contingency(key, np.full(ell, np.inf))) for key in union
        ]
        out.append(NetworkModel.build(network, conts))
    return out


def meet(f: NetworkModel, g: NetworkModel) -> NetworkModel:
    """``f ^ g``: the model whose feasible set is ``Q(f) inter Q(g)``.

    Under common PTDFs the stacked system has identical row pairs and collapses
    exactly to the elementwise minimum after alignment.  Raises where the PTDFs
    differ, which is where the stack fallback would go.
    """
    f_u, g_u = align(f, g)
    if not np.allclose(f_u.K, g_u.K):
        raise NotImplementedError(
            "meet() requires a common constraint geometry (Assumption 1): the two "
            "models' PTDFs differ, so their rows do not correspond and there is no "
            "elementwise minimum to take.  The intersection is still well defined "
            "as the stacked system [K_f; K_g] q <= [f; g] -- implement that here."
        )
    return with_limits(f_u, np.minimum(f_u.b, g_u.b))
