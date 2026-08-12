"""Primal geometry: the V-representation of ``Q(b)``.

``network`` holds the **H-representation** -- a list of half-spaces ``Kq <= b``.
This module converts to the **V-representation**, the same polytope described by
its extreme points, and labels each vertex with the constraints tight there and a
direction exposing it.

Why that labelling is the point.  The faces of ``Q(b)`` and the cones of its
normal fan are dual: a vertex corresponds to a full-dimensional normal cone,
hence to a maximal *realizable active set*, and any direction in the relative
interior of that cone exposes it.  So enumerating vertices enumerates the active
sets attainable at an optimum -- which is what a regime sweep wants.  In two
dimensions, listing vertices in cyclic order and walking the normal fan by angle
are the same traversal, so the ``d``-sweep falls out rather than being separate
code.

Scale.  By the Upper Bound Theorem a ``d``-polytope with ``m`` facets can have
``~m^(d/2)`` vertices, so the count is polynomial in the constraints but with an
exponent linear in the dimension (``d = n - 1`` here, after power balance).
Adding contingencies is survivable; adding buses is not.  Above :data:`MAX_NODES`
the *answer* is too big, not just the computation, and the right move is to
sample realized directions instead -- a different and better-posed question.
"""

from __future__ import annotations

from typing import NamedTuple

import cvxpy as cp
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

from .network import NetworkModel

MAX_NODES = 7  # above this the vertex count, not the runtime, is the problem
MAX_VERTICES = 100_000  # backstop for a case that slips under MAX_NODES
VERTEX_TOL = 1e-7  # numerical zero for feasibility / tightness / dedupe


def free_basis(n: int, drop: int) -> np.ndarray:
    """``T``: plot/parameter coordinates ``u in R^(n-1)`` -> node injections
    ``q = T u in R^n``, eliminating coordinate ``drop`` via power balance.

    Column ``k`` of ``T`` is the injection pattern "one unit at the ``k``-th kept
    node, balanced at ``drop``".  Any ``T`` with ``1^T T = 0`` is a valid basis for
    the balanced subspace; this is the simplest one.
    """
    keep = [i for i in range(n) if i != drop % n]
    T = np.zeros((n, n - 1))
    for col, node in enumerate(keep):
        T[node, col] = 1.0
        T[drop % n, col] = -1.0
    return T


def plane_system(
    model: NetworkModel, T: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The enforced constraints written in the coordinates of ``T``:
    ``(M, c, rows)`` with ``M = K_rows @ T`` and ``c = b_rows``.

    Substituting ``q = T u`` into ``k_i^T q <= b_i`` gives ``(T^T k_i)^T u <= b_i``,
    so the reduced normals are just ``K T`` -- no special handling, which is why
    plotting in a different basis needs no correction beyond this one product.
    ``rows`` maps each reduced row back to its global row index.
    """
    if abs(np.asarray(T).sum(axis=0)).max() > VERTEX_TOL:
        raise ValueError(
            "basis columns must be balanced injections (1^T T = 0); otherwise the "
            "coordinates leave the power-balance plane the polytope lives in."
        )
    rows = np.where(np.isfinite(model.b))[0]
    return model.K[rows] @ T, model.b[rows], rows


def is_bounded(M: np.ndarray, tol: float = VERTEX_TOL) -> bool:
    """Whether ``{u : M u <= c}`` is bounded, i.e. its recession cone
    ``{z : M z <= 0}`` is just the origin.  Independent of ``c``."""
    d = M.shape[1]
    z = cp.Variable(d)
    obj = cp.Parameter(d)
    problem = cp.Problem(cp.Maximize(obj @ z), [M @ z <= 0, z >= -1, z <= 1])
    e = np.zeros(d)
    for k in range(d):
        for sign in (1.0, -1.0):
            e[k] = sign
            obj.value = e.copy()
            problem.solve(solver=cp.HIGHS)
            if problem.value is not None and problem.value > 1e-6:
                return False
        e[k] = 0.0
    return True


def _polygon_2d(
    M: np.ndarray, c: np.ndarray, tol: float = VERTEX_TOL
) -> list[np.ndarray]:
    """Vertices of a 2-D ``{M u <= c}``, in counter-clockwise order.

    Exact and dependency-free: every vertex is the intersection of two
    constraints, so solve each 2x2 system, keep the feasible ones, dedupe, and
    sort by angle about the centroid.  ``O(m^2)`` pairs, which at toy scale is
    nothing and avoids needing a strict interior point.
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
    strictly interior point -- taken as the Chebyshev centre."""
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
    """A vertex of ``Q(b)``, the constraints tight there, and a direction that
    exposes it."""

    q: np.ndarray  # vertex, in node coordinates
    rows: np.ndarray  # global row indices of K tight at q
    direction: np.ndarray  # a node-space d whose support problem attains q


def faces(
    model: NetworkModel,
    drop: int | None = None,
    tol: float = VERTEX_TOL,
    max_vertices: int = MAX_VERTICES,
) -> list[Face]:
    """Every vertex of ``Q(b)``, with its active set and an exposing direction.

    The active sets returned are exactly the ones realizable at an optimum, so
    this enumerates the regimes without reference to any particular ``y``.  In
    two dimensions the result comes back in cyclic order, which *is* the
    direction sweep.

    The exposing direction is ``sum_{i in rows} k_i``: a strictly positive
    combination of the generators of the vertex's normal cone, hence in that
    cone's relative interior.  It is unnormalised, and only defined up to the
    usual ``+c*1`` balance shift.
    """
    n = model.network.n_nodes
    if n > MAX_NODES:
        raise ValueError(
            f"vertex enumeration refused at {n} nodes (limit {MAX_NODES}): by the "
            f"Upper Bound Theorem a {n - 1}-dimensional polytope with m facets can "
            f"have ~m^{(n - 1) // 2} vertices, so the enumeration itself is the "
            "wrong question at this scale.  Sample realized directions instead."
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
    if len(verts) > max_vertices:
        raise ValueError(f"{len(verts)} vertices exceeds max_vertices={max_vertices}")

    out = []
    for u in verts:
        tight = rows[np.abs(M @ u - c) <= tol * np.maximum(1.0, np.abs(c))]
        q = T @ u
        out.append(Face(q=q, rows=tight, direction=model.K[tight].sum(axis=0)))
    return out


def polygon(
    model: NetworkModel, T: np.ndarray | None = None, tol: float = VERTEX_TOL
) -> np.ndarray:
    """The 2-D outline of ``Q(b)`` in the coordinates of ``T``, as an ordered
    ``(n_vertices, 2)`` array ready to hand to a fill/plot call.

    Only the coordinates -- no active sets, no directions.  This is what drawing
    needs; :func:`faces` is what searching needs.
    """
    n = model.network.n_nodes
    T = free_basis(n, model.network.slack_idx) if T is None else np.asarray(T)
    if T.shape[1] != 2:
        raise ValueError(f"polygon() needs a 2-column basis, got {T.shape[1]}")
    M, c, _ = plane_system(model, T)
    verts = _polygon_2d(M, c, tol)
    return np.array(verts) if verts else np.zeros((0, 2))


def hull_2d(points: np.ndarray) -> np.ndarray:
    """Ordered outline of a 2-D point cloud -- the shadow of a higher-dimensional
    polytope is the hull of its projected vertices."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points
    hull = ConvexHull(points)
    return points[hull.vertices]
