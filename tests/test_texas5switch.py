"""The station-switch case: one PTDF for every switch regime, and the hours."""

import numpy as np
import pytest

from ftr_align.cases import texas5switch as t5s
from ftr_align.cases.texas5switch import CLOSED, OPEN
from ftr_align.network import PhysicalNetwork, intersection


def test_open_switch_is_the_line_network():
    """y = 0: the switch row is zero and the line rows are the lines-only PTDF."""
    H = t5s.switch_ptdf(OPEN)
    net = PhysicalNetwork(A=t5s.A_L, x=t5s.X_L, slack_idx=t5s.SLACK,
                          node_names=np.array(t5s.NODE_NAMES), element_names=np.array(t5s.LINE_NAMES))
    assert np.allclose(H[t5s.n_lines:], 0.0)
    assert np.allclose(H[: t5s.n_lines], net.ptdf())


def test_closed_switch_is_the_bus_branch_network():
    """y = inf: the line rows equal the five-bus PTDF (H1, H2 merged) lifted to nodes."""
    M = t5s.M_CLOSED
    net = PhysicalNetwork(A=M @ t5s.A_L, x=t5s.X_L, slack_idx=t5s.SLACK,
                          node_names=np.array(["W", "N", "S", "D", "H"]), element_names=np.array(t5s.LINE_NAMES))
    H = t5s.switch_ptdf(CLOSED)
    assert np.allclose(H[: t5s.n_lines], net.ptdf() @ M)
    assert np.allclose(H[t5s.n_lines], t5s.switch_ptdf(1e9)[t5s.n_lines], atol=1e-6)  # the switch row, by the relaxed limit


def test_relaxed_switch_tends_to_closed():
    assert np.abs(t5s.switch_ptdf(1e8) - t5s.switch_ptdf(CLOSED)).max() < 1e-6


def test_closed_switch_row_moves_injection_across_the_station():
    """One MW from H1 to H2 crosses the switch in full."""
    H = t5s.switch_ptdf(CLOSED)
    q = np.zeros(6)
    q[4], q[5] = 1.0, -1.0
    assert np.isclose((H @ q)[-1], 1.0)


@pytest.fixture(scope="module")
def hours():
    """A fast two-hour instance: hours 1 and 2 at a 25% margin, hour 1 binding DH and SH."""
    return t5s.scenarios_from_injections(
        {"1": (CLOSED, t5s.Q[1], ["DH", "SH"]), "2": (OPEN, t5s.Q[2], ["NH", "ND"])}, margin=0.25
    )


def test_binding_lines_sit_at_rating_in_their_hour(hours):
    b_line, scenarios, cross = hours
    for sc in scenarios:
        assert cross[sc.label][sc.label]
    assert not cross["2"]["1"]  # hour 2's injection does not fit the closed network


def test_b2_has_no_exposure_and_the_unlimited_closed_switch_is_infinite(hours):
    b_line, scenarios, _ = hours
    b2 = intersection(t5s.model(CLOSED, b_line), t5s.model(OPEN, b_line))
    tbl = t5s.evaluate(b2, scenarios)
    assert np.allclose(tbl["U"].to_numpy(), 0.0, atol=1e-6)
    closed = t5s.evaluate(t5s.model(CLOSED, b_line), scenarios)
    assert np.isinf(closed.filter(closed["scenario"] == "2")["U"][0])  # finding 1: the open hour prices H1 != H2
    assert np.isclose(closed.filter(closed["scenario"] == "1")["U"][0], 0.0, atol=1e-6)


def test_enumeration_and_milp_agree(hours):
    b_line, scenarios, _ = hours
    template = t5s.model(CLOSED, b_line=b_line, b_switch=31.7)  # a switch limit the open hour can live with
    fence = t5s.rating_fence(b_line, b_switch=100.0)
    for switch in t5s.SWITCH_MODES:
        lp = t5s.design_limits(template, scenarios, 0.0, b_max=fence, switch=switch)
        milp = t5s.design_limits_milp(template, scenarios, 0.0, b_max=fence, switch=switch)
        assert np.isclose(lp.objective, milp.objective, rtol=1e-6), switch
        assert milp.status == "optimal" and milp.gap < 1e-6
    r = t5s.n_lines + t5s.n_switches
    fixed = t5s.design_limits_milp(template, scenarios, 0.0, b_max=fence, switch="fixed")
    assert fixed.b[t5s.n_lines] == 31.7 and fixed.b[r + t5s.n_lines] == 31.7  # the switch limit was not touched
    sym = t5s.design_limits_milp(template, scenarios, 0.0, b_max=fence, switch="symmetric")
    assert np.isclose(sym.b[t5s.n_lines], sym.b[r + t5s.n_lines])


def test_closed_switch_fixed_unlimited_has_no_design(hours):
    """Finding 1: with the FTR switch closed and its limit fixed at inf, an open hour
    makes U infinite, so no choice of line limits reaches tau = 0."""
    b_line, scenarios, _ = hours
    template = t5s.model(CLOSED, b_line=b_line, b_switch=np.inf)
    with pytest.raises(ValueError, match="no design"):
        t5s.design_limits_milp(template, scenarios, 0.0, b_max=t5s.rating_fence(b_line), switch="fixed")
