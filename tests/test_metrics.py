"""Reporting layer: the tables agree with the numbers they report.

Deliberately thin.  These are views over quantities already tested in
``test_attribution.py``; what needs checking is that the view does not distort
them, not the quantities themselves.
"""

import numpy as np
import polars as pl
import pytest

from ftr_align import SupportProblem, align, clear_dam, meet, row_table, run_row
from ftr_align.attribution import block_shares, failure_modes
from ftr_align.duality import attribution_blocks, block_totals
from ftr_align.metrics import block_table
from ftr_align.cases import toy

CLEAR = {"solver": "CLARABEL"}


def _direction(g, scenario="(a)"):
    return clear_dam(g, toy.SCENARIOS[scenario], solver=CLEAR).direction


@pytest.mark.parametrize("case", list(toy.MODELS))
def test_run_row_matches_the_underlying_quantities(case):
    f, g = toy.MODELS[case]
    d = _direction(g)
    row = run_row(f, g, d, labels={"case": case}, solver=CLEAR)
    modes = failure_modes(f, g, d, solver=CLEAR)

    assert row["case"] == case
    for key in ("U", "V", "Delta", "h_f", "h_g", "h_meet"):
        assert row[key] == pytest.approx(modes[key], rel=1e-9, abs=1e-6)
    for mode in ("U", "V"):
        if row[f"floor_ratio_{mode}"] is not None:
            assert row[f"floor_ratio_{mode}"] == pytest.approx(
                row[f"floor_{mode}"] / modes[mode]
            )
        # dim ker C is rows-minus-blocks only because blocks partition J*
        assert row[f"dim_trade_space_{mode}"] >= 0
        assert row[f"max_block_{mode}"] >= 1 or row[f"n_blocks_{mode}"] == 0


def test_run_rows_stack_into_a_frame():
    """The point of returning a dict: a sweep is a list comprehension, and a new
    column is a new key."""
    rows = [
        run_row(f, g, _direction(g, s), {"case": c, "scenario": s}, solver=CLEAR)
        for c, (f, g) in toy.MODELS.items()
        for s in toy.SCENARIOS
    ]
    df = pl.DataFrame(rows)
    assert df.height == len(toy.MODELS) * len(toy.SCENARIOS)
    assert {"case", "scenario", "U", "V", "Delta"} <= set(df.columns)


@pytest.mark.parametrize("case", list(toy.MODELS))
@pytest.mark.parametrize("mode", ["U", "V"])
def test_row_table_shares_sum_to_the_failure_mode(case, mode):
    """The per-row shares in the table are the same numbers block_shares sums,
    so the table must reconstruct the mode too -- restricting the rows shown
    must not drop any that carry value."""
    f, g = toy.MODELS[case]
    d = _direction(g)
    table = row_table(f, g, d, mode=mode, solver=CLEAR)
    modes = failure_modes(f, g, d, solver=CLEAR)
    tol = 1e-3 * max(1.0, abs(modes["h_f"]), abs(modes["h_g"]))
    assert table["share"].sum() == pytest.approx(modes[mode], abs=tol)


def test_row_table_reports_limits_and_difference_kinds():
    f, g = toy.MODELS["mixed"]
    d = _direction(g)
    table = row_table(f, g, d, mode="V", solver=CLEAR)

    # meet is the tighter limit, row by row, as reported
    assert np.all(
        table["meet_i"].to_numpy()
        <= np.minimum(table["f_i"].to_numpy(), table["g_i"].to_numpy()) + 1e-9
    )
    # mixed carries both kinds, and the table names them
    kinds = set(table["difference"].drop_nulls().to_list())
    assert {"level_V", "coverage_U"} <= kinds


def test_block_table_totals_match_block_shares():
    f, g = toy.MODELS["mixed"]
    d = _direction(g)
    blocks, shares = block_shares(f, g, d, mode="V", solver=CLEAR)
    _, g_u = align(f, g)
    table = block_table(g_u, blocks, shares, value_name="V")

    assert table.height == len(blocks)
    assert table["size"].to_list() == [len(rows) for rows in blocks]
    assert np.allclose(table["V"].to_numpy(), shares)


def test_block_table_reports_support_totals_too():
    """The same table serves W_{J_r} (a support value split) and U_B (a failure
    mode split) -- only the column name differs."""
    sys = toy.REDUNDANT_MODELS["derate"][1]
    d = _direction(sys)
    prob = SupportProblem(sys, d)
    blocks = attribution_blocks(prob)
    W = block_totals(prob.data.b, prob.solve(solver=CLEAR).mu, blocks)
    table = block_table(sys, blocks, W)

    assert table["W"].sum() == pytest.approx(prob.solve(solver=CLEAR).value, abs=2)
    # the parallel twins are reported as one block, with both members named
    (members,) = table.filter(pl.col("size") == 2)["members"].to_list()
    assert sorted(members) == ["base:SLa:upper", "base:SLb:upper"]
