"""A network model is constraint rows ``(K, b)`` on a node set, stored one
contingency at a time.  The intersection of two models is their rows stacked --
no common row index, no shared PTDFs -- so it is checked here against what it
must equal where the old elementwise-min construction was valid.
"""

import numpy as np
import pytest

from ftr_align import Contingency, NetworkModel, SupportProblem, intersection, with_limits
from ftr_align.cases import toy
from toy_facts import direction

CLEAR = {"solver": "CLARABEL"}


def _model(net, keys, limits):
    return NetworkModel.build(net, [Contingency(k, limits) for k in keys])


def _h(model, d):
    return SupportProblem(model, d).solve(solver=CLEAR).value


def test_model_is_rows_on_a_node_set():
    net = toy.NETWORK
    dam = _model(net, [None], toy.BASE_LIMITS)
    ftr = _model(net, [None, toy.SL], toy.BASE_LIMITS)
    assert dam.keys == [None]
    assert ftr.keys == [None, toy.SL]
    assert ftr.nodes == tuple(toy.NODE_NAMES)
    assert ftr.K.shape == (ftr.n_rows, 3) and ftr.b.shape == (ftr.n_rows,)
    # K = [H; -H], contingencies in order
    assert np.array_equal(ftr.K, np.vstack([ftr.H, -ftr.H]))
    assert np.array_equal(ftr.H[ftr.rows_upper(toy.SL)], net.ptdf(toy.SL))
    assert ftr.labels().height == ftr.n_rows


def test_with_limits_shares_the_rows():
    model = _model(toy.NETWORK, [None, toy.SL], toy.BASE_LIMITS)
    tightened = with_limits(model, 0.5 * model.b)
    assert np.allclose(tightened.b, 0.5 * model.b)
    assert all(
        np.shares_memory(a.H, b.H)
        for a, b in zip(model.contingencies, tightened.contingencies)
    )


def test_intersection_stacks_the_rows():
    """def:intersection -- J = J_f disjoint-union J_g, nothing merged or dropped."""
    f, g = toy.MODELS["mixed"]
    both = intersection(f, g)
    assert both.n_rows == f.n_rows + g.n_rows
    assert both.keys == f.keys + g.keys
    assert sorted(both.b.tolist()) == sorted(f.b.tolist() + g.b.tolist())
    # rows are shared with the parents, not copied
    assert np.shares_memory(both.contingencies[0].H, f.contingencies[0].H)


@pytest.mark.parametrize("case", list(toy.MODELS))
@pytest.mark.parametrize("scenario", list(toy.SCENARIOS))
def test_intersection_equals_the_elementwise_min_on_one_network(case, scenario):
    """Where both models share one physical network, the stack has identical row
    pairs and the intersection is the elementwise-min model on the union of rows --
    the construction this replaced.  Same polytope, so the same support value."""
    f, g = toy.MODELS[case]
    limit = {}
    for m in (f, g):
        labels = m.labels()
        for c, e, side, b in zip(labels["contingency"], labels["element"], labels["side"], m.b):
            limit[c, e, side] = min(b, limit.get((c, e, side), np.inf))
    keys = list(dict.fromkeys(f.keys + g.keys))
    names = toy.NETWORK.element_names
    collapsed = NetworkModel.build(
        toy.NETWORK,
        [
            Contingency(
                k,
                [limit[f.contingencies[0].label if k is None else str(names[k]), str(e), "upper"] for e in names],
                [limit[f.contingencies[0].label if k is None else str(names[k]), str(e), "lower"] for e in names],
            )
            for k in keys
        ],
    )
    d = direction(g, scenario)
    h = _h(intersection(f, g), d)
    assert h == pytest.approx(_h(collapsed, d), abs=1e-6 * max(1.0, abs(h)))


def test_intersection_across_networks_needs_only_common_nodes():
    """The base toy and its double-circuit variant have different elements and
    different PTDF rows but the same nodes -- the case the old construction could
    not express.  They are electrically identical, so the intersection of a model
    with its redundant twin is that model again."""
    plain = _model(toy.NETWORK, [None], toy.BASE_LIMITS)
    twin = _model(toy.REDUNDANT_NETWORK, [None], toy.REDUNDANT_LIMITS)
    both = intersection(plain, twin)
    assert both.n_rows == plain.n_rows + twin.n_rows
    for scenario in toy.SCENARIOS:
        d = direction(plain, scenario)
        assert _h(both, d) == pytest.approx(_h(plain, d), rel=1e-6)


def test_intersection_rejects_different_node_sets():
    plain = _model(toy.NETWORK, [None], toy.BASE_LIMITS)
    renamed = NetworkModel(nodes=("x", "y", "z"), contingencies=plain.contingencies)
    with pytest.raises(ValueError, match="node set"):
        intersection(plain, renamed)


def test_row_lookup_refuses_an_ambiguous_key():
    """A contingency both models enforce appears twice in their intersection, so
    looking it up by key has no single answer."""
    f, g = toy.MODELS["mixed"]
    with pytest.raises(KeyError):
        intersection(f, g).rows_upper(None)
