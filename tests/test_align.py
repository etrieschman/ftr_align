"""Independently-defined models with different contingency lists are mapped onto
a common row index by align()/embed() for row-level comparison.  (Support values
and the gap don't need this -- they use the node-space direction.)
"""

import numpy as np

from ftr_align import Contingency, NetworkModel, align, embed
from ftr_align.cases import toy


def _model(net, keys, limits):
    limits = np.asarray(limits, dtype=float)
    return NetworkModel.build(net, [Contingency(k, limits.copy(), limits.copy()) for k in keys])


def test_independent_models_have_distinct_geometries():
    net = toy.toy_network()
    dam = _model(net, [None], toy.BASE_LIMITS)            # base only
    ftr = _model(net, [None, toy.SL], toy.BASE_LIMITS)    # base + SL outage
    assert dam.n_rows != ftr.n_rows
    assert dam.keys == [None]
    assert ftr.keys == [None, toy.SL]


def test_align_to_common_index():
    net = toy.toy_network()
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


def test_embed_vector_matches_by_identity():
    net = toy.toy_network()
    dam = _model(net, [None], toy.BASE_LIMITS)
    ftr = _model(net, [None, toy.SL], toy.BASE_LIMITS)

    # a per-row vector on the DAM rows, embedded onto the (larger) FTR rows
    mu_dam = np.zeros(dam.n_rows)
    mu_dam[dam.rows_upper(None)[toy.SL]] = 7.0
    mu_ftr = embed(mu_dam, dam, ftr)

    assert mu_ftr[ftr.rows_upper(None)[toy.SL]] == 7.0
    # FTR-only contingency rows have no source -> filled with 0
    assert mu_ftr[ftr.rows_upper(toy.SL)].sum() == 0.0
    assert mu_ftr.shape[0] == ftr.n_rows
