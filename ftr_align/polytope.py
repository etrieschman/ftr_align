"""The V-representation of ``Q(b)``: its vertices, the constraints tight at each,
and a direction exposing it.

Faces of a polytope and cones of its normal fan are dual, so a vertex is a
maximal realizable active set and any direction interior to its cone exposes it.
Enumerating vertices therefore enumerates the regimes, without reference to any
``y``.

Scale: a ``d``-polytope with ``m`` facets has up to ``~m^(d/2)`` vertices, and
``d = n - 1`` here.  Contingencies are survivable; buses are not.  Above
:data:`MAX_NODES` the answer is too big, not the computation.
"""

from __future__ import annotations

from typing import NamedTuple

import cvxpy as cp
import numpy as np
from scipy.spatial import HalfspaceIntersection

from .network import NetworkModel

MAX_NODES = 7  # above this the vertex count, not the runtime, is the problem
VERTEX_TOL = 1e-7  # numerical zero for feasibility / tightness / dedupe


def free_basis(n: int, drop: int) -> np.ndarray:
    """``T``: parameter coordinates ``u in R^(n-1)`` -> node injections ``q = T u``,
    eliminating coordinate ``drop`` via power balance.

    Column ``k`` is "one unit at the k-th kept node, balanced at ``drop``".
    """
    keep = [i for i in range(n) if i != drop % n]
    T = np.zeros((n, n - 1))
    for col, node in enumerate(keep):
        T[node, col] = 1.0
        T[drop % n, col] = -1.0
    return T


def basis_from_columns(columns) -> np.ndarray:
    """Assemble a basis ``T`` from explicit injection patterns, one per axis.

    Each column says what moving one unit along that axis does to the nodal
    injections, and must be balanced.
    """
    T = np.asarray(columns, dtype=float).T
    if abs(T.sum(axis=0)).max() > VERTEX_TOL:
        raise ValueError("each column must be a balanced injection (sum to zero)")
    if np.linalg.matrix_rank(T) != T.shape[1]:
        raise ValueError("columns must be linearly independent")
    return T


def plane_system(
    model: NetworkModel, T: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The enforced constraints in the coordinates of ``T``: ``(M, c, rows)`` with
    ``M = K_rows @ T``.

    Substituting ``q = T u`` turns row ``i`` into ``(T^T k_i)^T u <= b_i``, so the
    reduced normals are just ``K T``.  ``rows`` maps back to global row indices.
    """
    if abs(np.asarray(T).sum(axis=0)).max() > VERTEX_TOL:
        raise ValueError(
            "basis columns must be balanced injections (1^T T = 0); otherwise the "
            "coordinates leave the power-balance plane the polytope lives in."
        )
    rows = np.where(np.isfinite(model.b))[0]
    return model.K[rows] @ T, model.b[rows], rows


def is_bounded(M: np.ndarray) -> bool:
    """Whether ``{u : M u <= c}`` is bounded, i.e. its recession cone ``{z : M z <= 0}``
    is just the origin.  Independent of ``c``.

    One LP, by polar duality.  The recession cone is the polar of
    ``K = cone(rows of M)``, and a polar is trivial exactly when the cone it came
    from is all of ``R^d`` -- that is, when the rows positively span:

        rank(M) = d   and   M^T lambda = 0 for some lambda >= 1

    Given such a lambda, ``-m_i = (1/lambda_i) sum_{j != i} lambda_j m_j`` puts every
    row's negation in the cone.  The rank test is not redundant: a lone zero row
    satisfies the LP while spanning nothing.
    """
    m, d = M.shape
    if np.linalg.matrix_rank(M) < d:
        return False
    lam = cp.Variable(m)
    problem = cp.Problem(cp.Minimize(0), [M.T @ lam == 0, lam >= 1])
    problem.solve(solver=cp.HIGHS)
    return problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


def _polygon_2d(
    M: np.ndarray, c: np.ndarray, tol: float = VERTEX_TOL
) -> list[np.ndarray]:
    """Vertices of a 2-D ``{M u <= c}``, counter-clockwise.

    Every vertex is the intersection of two constraints, so solve each 2x2 system,
    keep the feasible ones, dedupe, and sort by angle.  ``O(m^2)`` pairs, exact, and
    needs no interior point -- so it also handles a flat region, which Qhull cannot.
    """
    pts = []
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            Aij = np.array([M[i], M[j]])
            if abs(np.linalg.det(Aij)) < tol:
                continue  # parallel constraints meet at infinity
            v = np.linalg.solve(Aij, np.array([c[i], c[j]]))
            if np.all(M @ v <= c + tol * np.maximum(1.0, np.abs(c))):
                pts.append(v)
    if not pts:
        return []
    V = np.array(pts)
    keep = []
    for v in V:  # dedupe: degenerate vertices are found once per tight pair
        if not any(np.allclose(v, w, atol=1e-6) for w in keep):
            keep.append(v)
    V = np.array(keep)
    centre = V.mean(axis=0)
    order = np.argsort(np.arctan2(*(V - centre).T[::-1]))
    return list(V[order])


def _vertices_nd(M: np.ndarray, c: np.ndarray) -> list[np.ndarray]:
    """Vertices of a general-dimension ``{M u <= c}`` via Qhull, which needs a
    strictly interior point -- taken as the Chebyshev centre.

    The centre comes from an LP maximising the inscribed radius ``r`` subject to
    ``M u + r ||m_i|| <= c``.  ``r = 0`` means the region is flat in these
    coordinates and its vertices are not well defined.
    """
    d = M.shape[1]
    norms = np.linalg.norm(M, axis=1)
    u, r = cp.Variable(d), cp.Variable(nonneg=True)
    problem = cp.Problem(cp.Maximize(r), [M @ u + r * norms <= c])
    problem.solve(solver=cp.HIGHS)
    if r.value is None or r.value <= VERTEX_TOL:
        raise ValueError(
            "polytope has empty interior in these coordinates, so its vertices "
            "are not well defined; check that the model is not over-constrained."
        )
    halfspaces = np.hstack([M, -c.reshape(-1, 1)])
    return list(HalfspaceIntersection(halfspaces, u.value).intersections)


class Face(NamedTuple):
    """A vertex of ``Q(b)``, the constraints tight there, and a direction exposing it."""

    q: np.ndarray  # vertex, in node coordinates
    rows: np.ndarray  # global row indices of K tight at q
    direction: np.ndarray  # a node-space d whose support problem attains q


def faces(
    model: NetworkModel,
    drop: int | None = None,
    tol: float = VERTEX_TOL,
) -> list[Face]:
    """Every vertex of ``Q(b)``, with its active set and an exposing direction.

    The exposing direction is ``sum_{i in rows} k_i``: a strictly positive
    combination of the normal cone's generators, hence interior to it.  Unnormalised,
    and defined up to the usual ``+c*1`` shift.  In 2-D the vertices come back in
    cyclic order, which is sweep order.
    """
    n = model.network.n_nodes
    if n > MAX_NODES:
        raise ValueError(
            f"vertex enumeration refused at {n} nodes (limit {MAX_NODES}): by the "
            f"Upper Bound Theorem a {n - 1}-dimensional polytope with m facets can "
            f"have ~m^{(n - 1) // 2} vertices, so the enumeration itself is the "
            "wrong question at this scale."
        )
    drop = model.network.slack_idx if drop is None else drop
    T = free_basis(n, drop)
    M, c, rows = plane_system(model, T)
    if not is_bounded(M):
        raise ValueError(
            "Q(b) is unbounded in these coordinates, so it has no finite vertex "
            "set.  Enforce enough rows to bound it, or clip to a box first."
        )

    verts = _polygon_2d(M, c, tol) if M.shape[1] == 2 else _vertices_nd(M, c)
    out = []
    for u in verts:
        tight = rows[np.abs(M @ u - c) <= tol * np.maximum(1.0, np.abs(c))]
        q = T @ u
        out.append(Face(q=q, rows=tight, direction=model.K[tight].sum(axis=0)))
    return out


def polygon(
    model: NetworkModel,
    T: np.ndarray | None = None,
    tol: float = VERTEX_TOL,
    bounds=None,
) -> np.ndarray:
    """The 2-D outline of ``Q(b)`` in the coordinates of ``T``, ordered and ready to
    fill.

    ``bounds`` is an optional sequence of ``(a, c)`` half-planes ``a^T u <= c`` given
    directly in plot coordinates -- market bounds such as a generation cap -- which
    cut the outline down.
    """
    n = model.network.n_nodes
    T = free_basis(n, model.network.slack_idx) if T is None else np.asarray(T)
    if T.shape[1] != 2:
        raise ValueError(f"polygon() needs a 2-column basis, got {T.shape[1]}")
    M, c, _ = plane_system(model, T)
    if bounds:
        extra = np.array([np.asarray(a, dtype=float) for a, _ in bounds])
        M = np.vstack([M, extra])
        c = np.concatenate([c, [float(v) for _, v in bounds]])
    verts = _polygon_2d(M, c, tol)
    return np.array(verts) if verts else np.zeros((0, 2))
