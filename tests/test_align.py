"""Independently-defined models with different contingency lists are mapped
onto a common index by align(); verify the mapping lines rows up correctly.
"""

import numpy as np

from ftr_align import NetworkModel, align, clear_dam, SupportProblem, gap
from ftr_align.cases import toy


def test_independent_models_have_distinct_systems_then_align():
    net = toy.toy_network()
    # DAM: base only.  FTR: base + SL outage.  Defined independently.
    dam = NetworkModel.build(net, [None], toy.BASE_LIMITS)
    ftr = NetworkModel.build(net, [None, toy.SL], toy.BASE_LIMITS)

    # independently they have different geometries
    assert dam.system is not ftr.system
    assert dam.system.n_rows != ftr.system.n_rows

    dam_u, ftr_u = align(dam, ftr)

    # after align: one shared system, rows aligned
    assert dam_u.system is ftr_u.system
    assert dam_u.system.contingencies == [None, toy.SL]
    # DAM does not enforce the SL-outage rows -> inactive (inf) after mapping
    sl_rows = np.concatenate(
        [dam_u.system.rows_upper(toy.SL), dam_u.system.rows_lower(toy.SL)]
    )
    assert np.isinf(dam_u.b[sl_rows]).all()
    assert np.isfinite(ftr_u.b[sl_rows]).all()
    # base rows preserved through the mapping
    base_u = dam_u.system.rows_upper(None)
    assert np.allclose(dam_u.b[base_u], toy.BASE_LIMITS)


def test_align_order_independent():
    """Aligning (dam, ftr) vs (ftr, dam) gives the same support values."""
    net = toy.toy_network()
    dam = NetworkModel.build(net, [None, toy.SC], toy.BASE_LIMITS)
    ftr = NetworkModel.build(net, [None], toy.BASE_LIMITS)
    inst = toy.instance("(a)")

    dam_u, ftr_u = align(dam, ftr)
    y = clear_dam(dam_u, inst, solver="CLARABEL").y
    delta1 = gap(
        SupportProblem(ftr_u, y).solve(solver="CLARABEL"),
        SupportProblem(dam_u, y).solve(solver="CLARABEL"),
    )

    ftr_v, dam_v = align(ftr, dam)
    y2 = clear_dam(dam_v, inst, solver="CLARABEL").y
    delta2 = gap(
        SupportProblem(ftr_v, y2).solve(solver="CLARABEL"),
        SupportProblem(dam_v, y2).solve(solver="CLARABEL"),
    )
    assert delta1 == delta2
