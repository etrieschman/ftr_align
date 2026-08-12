"""Independently-defined models with different contingency lists are mapped onto
a common row index by align() for row-level comparison and for building the
intersection.  (Support values and the gap don't need this -- they use the
node-space direction.)
"""

import numpy as np

from ftr_align import Contingency, NetworkModel, align, with_limits
from ftr_align.cases import toy


def _model(net, keys, limits):
    return NetworkModel.build(net, [Contingency(k, limits) for k in keys])


def test_independent_models_have_distinct_geometries():
    net = toy.NETWORK
    dam = _model(net, [None], toy.BASE_LIMITS)            # base only
    ftr = _model(net, [None, toy.SL], toy.BASE_LIMITS)    # base + SL outage
    assert dam.n_rows != ftr.n_rows
    assert dam.keys == [None]
    assert ftr.keys == [None, toy.SL]


def test_align_to_common_index():
    net = toy.NETWORK
    dam = _model(net, [None], toy.BASE_LIMITS)
    ftr = _model(net, [None, toy.SL], toy.BASE_LIMITS)

    dam_u, ftr_u = align(dam, ftr)
    # common geometry, union contingency order
    assert dam_u.keys == ftr_u.keys == [None, toy.SL]
    assert dam_u.n_rows == ftr_u.n_rows
    # DAM does not enforce the SL-outage rows -> unmonitored (inf) after mapping
    sl_rows = np.concatenate([dam_u.rows_upper(toy.SL), dam_u.rows_lower(toy.SL)])
    assert np.isinf(dam_u.b[sl_rows]).all()
    assert np.isfinite(ftr_u.b[sl_rows]).all()
    # base ratings preserved
    assert np.allclose(dam_u.b[dam_u.rows_upper(None)], toy.BASE_LIMITS)


def test_with_limits_keeps_contingencies_consistent():
    """A limit change must rebuild the Contingency objects, not just b.

    dataclasses.replace(model, b=...) leaves the contingency list carrying the
    old limits, and align() reads limits from that list -- so a model built the
    short way would silently re-align to its pre-change values."""
    net = toy.NETWORK
    model = _model(net, [None, toy.SL], toy.BASE_LIMITS)
    tightened = with_limits(model, 0.5 * model.b)

    assert np.allclose(tightened.b, 0.5 * model.b)
    rebuilt = np.concatenate(
        [
            np.concatenate([c.upper for c in tightened.contingencies]),
            np.concatenate([c.lower for c in tightened.contingencies]),
        ]
    )
    assert np.allclose(rebuilt, tightened.b)
    # and the round trip through align() preserves them
    (realigned,) = align(tightened)
    assert np.allclose(realigned.b, tightened.b)
    assert np.shares_memory(tightened.K, model.K)  # K reused, not recomputed
