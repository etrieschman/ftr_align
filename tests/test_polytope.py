"""Primal geometry: the V-representation and the figures built on it.

The 3-node polytope is small enough to state exactly, so these are oracle tests
rather than invariant tests -- the parallelogram's corners are known.
"""

import numpy as np
import pytest

from ftr_align import Contingency, NetworkModel, PhysicalNetwork, SupportProblem, meet
from ftr_align.polytope import (
    MAX_NODES,
    Face,
    faces,
    free_basis,
    is_bounded,
    plane_system,
    polygon,
)
from ftr_align.cases import toy
from toy_facts import find_case, nesting, uniform_scale

CLEAR = {"solver": "CLARABEL"}


def _area(V):
    x, y = V[:, 0], V[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------------------
# basis and reduction
# ---------------------------------------------------------------------------
def test_free_basis_columns_are_balanced():
    """1^T T = 0 -- every plot coordinate must map to a balanced injection, or
    the coordinates leave the plane the polytope lives in."""
    for drop in range(3):
        T = free_basis(3, drop)
        assert T.shape == (3, 2)
        assert np.allclose(T.sum(axis=0), 0.0)
        assert np.linalg.matrix_rank(T) == 2


def test_plane_system_is_just_K_times_T():
    """Substituting q = T u into k^T q <= b gives (T^T k)^T u <= b, so the
    reduced normals are K T and nothing else -- this is why plotting in another
    basis needs no correction."""
    _, g = toy.MODELS["derate"]
    T = free_basis(3, g.network.slack_idx)
    M, c, rows = plane_system(g, T)
    assert np.allclose(M, g.K[rows] @ T)
    assert np.allclose(c, g.b[rows])


def test_plane_system_rejects_an_unbalanced_basis():
    _, g = toy.MODELS["derate"]
    bad = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # columns don't sum to 0
    with pytest.raises(ValueError, match="balanced"):
        plane_system(g, bad)


# ---------------------------------------------------------------------------
# polygon
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", list(toy.MODELS))
def test_polygon_vertices_are_feasible_and_tight(case):
    """Every vertex the outline returns is network-feasible and sits on at least
    two constraints -- which is what a vertex of a 2-D polytope is."""
    _, g = toy.MODELS[case]
    T = free_basis(3, g.network.slack_idx)
    M, c, _ = plane_system(g, T)
    V = polygon(g, T)
    assert len(V) >= 3
    for u in V:
        slack = c - M @ u
        assert np.all(slack >= -1e-6)
        assert int(np.sum(np.abs(slack) <= 1e-6)) >= 2


def test_polygon_vertices_come_back_in_convex_order():
    """The fill call needs cyclic order; shoelace area is only correct then, and
    in 2-D that ordering *is* the direction sweep."""
    _, g = toy.MODELS["derate"]
    V = polygon(g)
    centre = V.mean(axis=0)
    angles = np.arctan2(*(V - centre).T[::-1])
    assert np.all(np.diff(angles) > 0)


def test_a_uniform_derate_scales_the_polytope():
    """A uniform derate scales the region, so its area scales by alpha^2 in two
    dimensions.  Alpha is read off the limit vectors, not assumed."""
    _, f, g = find_case(lambda f, g: uniform_scale(f, g) is not None,
                        what="a uniform derate")
    alpha = uniform_scale(f, g)
    assert _area(polygon(f)) == pytest.approx(alpha**2 * _area(polygon(g)))


@pytest.mark.parametrize("case", list(toy.MODELS))
def test_meet_region_matches_the_nesting_of_the_pair(case):
    """cor:canonical with the premise derived rather than tabulated: when one
    model's limits dominate the other's, the intersection *is* the tighter
    region.

    When neither dominates, all that follows is containment in both.  Not strict
    containment: `nesting` reads the limit *vectors*, and a row that crosses can
    still be redundant given the others, so the two regions can coincide even
    though the H-representations differ."""
    f, g = toy.MODELS[case]
    a_f, a_g = _area(polygon(f)), _area(polygon(g))
    a_meet = _area(polygon(meet(f, g)))
    match nesting(f, g):
        case "f":
            assert a_meet == pytest.approx(a_f, rel=1e-9)
        case "g":
            assert a_meet == pytest.approx(a_g, rel=1e-9)
        case _:
            assert a_meet <= min(a_f, a_g) + 1e-6


def test_a_crossing_pair_has_a_meet_smaller_than_both():
    """The pair that is NOT one-signed: each model is tighter on some row, so
    neither limit vector dominates.  cor:canonical does not apply and block_table
    refuses the pair in either order -- but geometrically all that is guaranteed
    is Q(f ^ g) inside both, since a crossing row can be redundant."""
    _, f, g = find_case(lambda f, g: nesting(f, g) == "cross", what="a crossing pair")
    a_f, a_g, a_meet = (_area(polygon(m)) for m in (f, g, meet(f, g)))
    assert a_meet <= a_f + 1e-6
    assert a_meet <= a_g + 1e-6


@pytest.mark.parametrize("case", list(toy.MODELS))
def test_the_contained_model_is_never_larger(case):
    """Enforcing more can only cut corners off: the nested model has no more
    area, and where it is strictly smaller it has no fewer vertices."""
    f, g = toy.MODELS[case]
    inner, outer = {"f": (f, g), "g": (g, f)}.get(nesting(f, g), (None, None))
    if inner is None:
        pytest.skip("this pair crosses; neither is the inner model")
    a_in, a_out = _area(polygon(inner)), _area(polygon(outer))
    assert a_in <= a_out + 1e-6
    if a_in < a_out - 1e-6:
        assert len(polygon(inner)) >= len(polygon(outer))


@pytest.mark.parametrize("case", list(toy.MODELS))
def test_faces_label_each_vertex_with_two_tight_rows(case):
    _, g = toy.MODELS[case]
    found = faces(g)
    assert len(found) >= 3
    for face in found:
        assert isinstance(face, Face)
        assert len(face.rows) == 2  # a vertex in 2-D is two tight constraints
        assert np.all(np.abs(g.K[face.rows] @ face.q - g.b[face.rows]) < 1e-6)
        assert face.q.sum() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("case", list(toy.MODELS))
def test_exposing_direction_actually_exposes_its_vertex(case):
    """The claim that makes enumeration a *search*: the representative direction
    is interior to the vertex's normal cone, so the support problem solved at it
    returns that vertex and no other."""
    for model in toy.MODELS[case]:
        for face in faces(model):
            sol = SupportProblem(model, face.direction).solve(
                solver=CLEAR, want_primal=True
            )
            assert np.allclose(sol.q, face.q, atol=1e-4)
            assert sol.value == pytest.approx(face.direction @ face.q, abs=1e-4)


def test_faces_enumerate_every_realizable_active_set():
    """Every direction's optimum is some enumerated vertex -- the enumeration is
    complete, which is what lets a sweep be replaced by a lookup."""
    _, g = toy.MODELS["derate"]
    known = [tuple(np.round(face.q, 6)) for face in faces(g)]
    rng = np.random.default_rng(0)
    for _ in range(40):
        d = rng.normal(size=3)
        d -= d.mean()  # any balanced direction
        sol = SupportProblem(g, d).solve(solver=CLEAR, want_primal=True)
        assert any(np.allclose(sol.q, q, atol=1e-4) for q in known)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def test_is_bounded_detects_an_open_direction():
    _, g = toy.MODELS["derate"]
    T = free_basis(3, g.network.slack_idx)
    M, _, _ = plane_system(g, T)
    assert is_bounded(M)
    assert not is_bounded(M[:1])  # a single half-space bounds nothing
    assert not is_bounded(M[:3])  # half the rows leave a recession direction


def test_is_bounded_needs_the_rank_test_not_just_the_lp():
    """The positive-span LP alone is not enough.  A degenerate row satisfies
    ``M^T lambda = 0`` at ``lambda = 1`` while spanning nothing, so dropping the
    rank check would call these bounded."""
    assert not is_bounded(np.zeros((1, 2)))  # a row that constrains nothing
    assert not is_bounded(np.array([[1.0, 0.0], [-1.0, 0.0]]))  # only x is pinned


def test_unbounded_model_is_refused():
    """A model monitoring only one element in one direction has no vertex set."""
    limits = np.array([np.inf, np.inf, 25.0])
    model = NetworkModel.build(
        toy.NETWORK, [Contingency(None, upper=limits, lower=np.full(3, np.inf))]
    )
    with pytest.raises(ValueError, match="unbounded"):
        faces(model)


def test_node_count_guard_explains_itself():
    """Above the guard the *answer* is too big, not the computation -- the error
    should say so rather than looking like a performance limit."""
    n = MAX_NODES + 1
    A = np.zeros((n, n))
    for j in range(n):  # a cycle, so no bridges and the PTDF is well defined
        A[j, j], A[(j + 1) % n, j] = 1.0, -1.0
    net = PhysicalNetwork(A=A, x=np.ones(n), slack_idx=0)
    model = NetworkModel.build(net, [Contingency(None, np.full(n, 100.0))])
    with pytest.raises(ValueError, match="Upper Bound Theorem"):
        faces(model)


# ---------------------------------------------------------------------------
# slack invariance
# ---------------------------------------------------------------------------
def test_slack_choice_shifts_the_direction_but_changes_nothing_else():
    """PTDF rows differ between slack conventions by a multiple of 1^T, which
    annihilates balanced injections.  So d = K^T y moves by a constant vector
    while Q(b), h and mu are untouched -- the slack is a labelling convention,
    and choosing it to suit a figure's axes costs nothing."""
    y = np.zeros(6)
    y[0] = 435.0
    values, mus, dirs, areas = [], [], [], []
    for slack in range(3):
        net = PhysicalNetwork(
            A=toy.INC, x=toy.X, slack_idx=slack,
            node_names=toy.NODE_NAMES, element_names=toy.ELEMENT_NAMES,
        )
        model = NetworkModel.build(net, [Contingency(None, toy.BASE_LIMITS)])
        d = model.K.T @ y
        sol = SupportProblem(model, d).solve(solver=CLEAR)
        values.append(sol.value)
        mus.append(sol.mu)
        dirs.append(d)
        areas.append(_area(polygon(model, free_basis(3, 2))))

    assert values[0] == pytest.approx(values[1]) == pytest.approx(values[2])
    assert np.allclose(mus[0], mus[1]) and np.allclose(mus[0], mus[2])
    assert areas[0] == pytest.approx(areas[1]) == pytest.approx(areas[2])
    for other in dirs[1:]:
        shift = dirs[0] - other
        assert np.allclose(shift, shift[0])  # a multiple of 1
