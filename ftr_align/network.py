"""Network geometry: PTDF, contingencies, the stacked constraint system, and the
network model that layers limits on top of it.

Notation follows the FTR pitch memo:

* ``K``  -- stacked PTDF, maps nodal injections ``q`` to monitored flows.
* ``A = [K; -K]`` -- the stacked constraint matrix.  The first ``C*ell`` rows are
  the upper-limit constraints ``Kq <= b_upper``; the next ``C*ell`` are the
  lower-limit constraints ``-Kq <= b_lower``.  Rows within a block run over
  ``(contingency, element)`` in the order of ``contingencies`` then element.
* a *network model* is the shared geometry ``A`` plus a limit vector ``b``.  DAM
  and FTR are two models with limit vectors ``f`` and ``g`` over the *same*
  system, which is what makes the dual-feasible set ``Lambda(y)`` shared.

The limit vector ``b``, the certificate ``y``, and the dual multipliers ``mu``
are all full-length vectors over the rows of ``A``, so they line up entrywise.
``A`` is dense: PTDF is structurally dense, so sparse storage would not help.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np
import polars as pl

# A contingency key:  None -> base case;  int -> single-element outage;
#                     tuple[int, ...] -> multi-element outage.
ContingencyKey = Hashable


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
    keep = np.delete(np.eye(n), slack_idx, axis=0)        # drop slack row
    return y_line @ inc.T @ keep.T @ np.linalg.inv(keep @ y_bus @ keep.T) @ keep


@dataclass(frozen=True)
class PhysicalNetwork:
    """Physical topology common to every contingency."""

    inc: np.ndarray                        # (n, ell) incidence, node x line
    x: np.ndarray                          # (ell,) reactances
    slack_idx: int = -1
    node_names: np.ndarray | None = None
    element_names: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return self.inc.shape[0]

    @property
    def n_elements(self) -> int:
        return self.inc.shape[1]

    def ptdf(self, outage_elements: list[int] | None = None) -> np.ndarray:
        """PTDF with ``outage_elements`` removed (incidence columns zeroed, so
        outaged elements carry no flow)."""
        inc = np.array(self.inc, dtype=float, copy=True)
        if outage_elements:
            inc[:, outage_elements] = 0.0
        return compute_ptdf(inc, self.x, self.slack_idx)


def outage_elements(key: ContingencyKey) -> list[int] | None:
    if key is None:
        return None
    if isinstance(key, (int, np.integer)):
        return [int(key)]
    return list(key)


def contingency_label(key: ContingencyKey, element_names=None) -> str:
    if key is None:
        return "base"
    if element_names is not None and isinstance(key, (int, np.integer)):
        return str(element_names[key])
    return str(key)


# ----------------------------------------------------------------------------
# Stacked constraint system  (the shared geometry A = [K; -K])
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class StackedSystem:
    network: PhysicalNetwork
    contingencies: list[ContingencyKey]    # universe ordering
    A: np.ndarray                          # (2 * C * ell, n), dense

    @property
    def n_rows(self) -> int:
        return self.A.shape[0]

    @property
    def ell(self) -> int:
        return self.network.n_elements

    def _block_start(self, key: ContingencyKey) -> int:
        return self.contingencies.index(key) * self.ell

    def rows_upper(self, key: ContingencyKey) -> np.ndarray:
        """Row indices for a contingency's elements in the upper-limit block."""
        s = self._block_start(key)
        return np.arange(s, s + self.ell)

    def rows_lower(self, key: ContingencyKey) -> np.ndarray:
        """Row indices for a contingency's elements in the lower-limit block."""
        half = len(self.contingencies) * self.ell
        s = self._block_start(key)
        return np.arange(half + s, half + s + self.ell)

    def labels(self) -> pl.DataFrame:
        """Per-row identity (contingency, element, side) -- built on demand for
        output tables, not stored."""
        ell = self.ell
        names = self.network.element_names
        conts = [contingency_label(c, names) for c in self.contingencies for _ in range(ell)]
        elems = [str(names[i]) if names is not None else str(i)
                 for _ in self.contingencies for i in range(ell)]
        return pl.DataFrame(
            {
                "row": np.arange(self.n_rows),
                "contingency": conts * 2,
                "element": elems * 2,
                "side": ["upper"] * (self.n_rows // 2) + ["lower"] * (self.n_rows // 2),
            }
        )


def build_stacked_system(
    network: PhysicalNetwork, contingencies: list[ContingencyKey]
) -> StackedSystem:
    """Assemble ``A = [K; -K]`` over the given contingency universe."""
    k = np.vstack([network.ptdf(outage_elements(c)) for c in contingencies])
    A = np.vstack([k, -k])
    return StackedSystem(network=network, contingencies=list(contingencies), A=A)


def shared_system(
    network: PhysicalNetwork, *contingency_sets: list[ContingencyKey]
) -> StackedSystem:
    """Build one stacked system over the *union* of several contingency sets
    (deduplicated, in order of first appearance).  Models built on the result
    have aligned rows, so ``b``, ``y`` and ``mu`` line up entrywise across DAM
    and FTR even when their enforced contingencies differ.

    Valid under Assumption 1 (common PTDFs across markets).  If FTR and DAM use
    *different* PTDFs for the same contingency, no shared system exists and the
    dual-attribution comparison does not apply.
    """
    union: list[ContingencyKey] = []
    for cset in contingency_sets:
        for key in cset:
            if key not in union:
                union.append(key)
    return build_stacked_system(network, union)


def embed(
    values: np.ndarray, source: StackedSystem, target: StackedSystem, fill: float = 0.0
) -> np.ndarray:
    """Re-express a per-row vector (a certificate ``y`` or a dual ``mu``) from
    ``source`` rows onto ``target`` rows, matching by contingency key.  Rows of
    ``target`` absent from ``source`` get ``fill`` (0 for ``y``/``mu``, ``inf``
    for a limit vector ``b``)."""
    out = np.full(target.n_rows, fill)
    for key in source.contingencies:
        if key in target.contingencies:
            out[target.rows_upper(key)] = values[source.rows_upper(key)]
            out[target.rows_lower(key)] = values[source.rows_lower(key)]
    return out


# ----------------------------------------------------------------------------
# Network model  (geometry + limit vector)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class NetworkModel:
    """Shared ``system`` + this model's limit vector ``b`` (full length over
    ``system`` rows; ``+inf`` on rows the model does not enforce, so the
    corresponding ``mu`` is pinned to zero)."""

    system: StackedSystem
    b: np.ndarray                          # (n_rows,)

    @property
    def active(self) -> np.ndarray:
        """Rows with finite limits (enforced)."""
        return np.isfinite(self.b)

    @property
    def enforced(self) -> list[ContingencyKey]:
        """Contingencies with at least one enforced row, in universe order."""
        out = []
        for key in self.system.contingencies:
            if np.isfinite(self.b[self.system.rows_upper(key)]).any():
                out.append(key)
        return out

    @classmethod
    def from_symmetric_limits(
        cls,
        system: StackedSystem,
        enforced: list[ContingencyKey],
        limits: np.ndarray,                # (ell,) per-element |flow| limit
    ) -> "NetworkModel":
        """Build a model with symmetric upper=lower limits on ``enforced``
        contingencies (the common case; apply a derate by scaling ``limits``)."""
        limits = np.asarray(limits, dtype=float)
        b = np.full(system.n_rows, np.inf)
        for key in enforced:
            b[system.rows_upper(key)] = limits
            b[system.rows_lower(key)] = limits
        return cls(system=system, b=b)
