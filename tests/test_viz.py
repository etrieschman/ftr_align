"""Plot layer: thin checks that the drawing agrees with the geometry.

Not a rendering test -- what matters is that the coordinates the figure uses are
the ones the model actually implies.
"""

import matplotlib

matplotlib.use("Agg")  # no display in CI

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ftr_align import clear_dam, meet
from ftr_align.polytope import free_basis, polygon
from ftr_align.viz import (
    basis,
    _draw_line,
    draw_constraints,
    draw_optimum,
    draw_region,
    frame_axes,
    label_axes,
    to_plot,
)
from ftr_align.cases import toy

CLEAR = {"solver": "CLARABEL"}


@pytest.fixture
def ax():
    fig, ax = plt.subplots()
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    yield ax
    plt.close(fig)


def test_to_plot_inverts_the_basis():
    T = free_basis(3, 2)
    q = np.array([100.0, 25.0, -125.0])
    assert np.allclose(T @ to_plot(T, q), q)


def test_to_plot_is_slack_shift_free():
    """q is balanced, so it round-trips exactly; a non-balanced vector would not,
    which is the point of requiring 1^T T = 0."""
    T = free_basis(3, 1)  # eliminate C instead
    q = np.array([100.0, 25.0, -125.0])
    assert np.allclose(T @ to_plot(T, q), q)


def test_draw_region_returns_the_model_polygon(ax):
    _, g = toy.MODELS["derate"]
    drawn = draw_region(ax, g)
    assert np.allclose(drawn, polygon(g))
    assert len(ax.patches) + len(ax.lines) > 0


def test_draw_optimum_marks_a_point_on_the_boundary(ax):
    """The marker must sit at the plot image of the support maximiser, not at
    some separately computed point."""
    _, g = toy.MODELS["derate"]
    d = clear_dam(g, toy.SCENARIOS["(a)"], solver=CLEAR).direction
    T = basis(g)
    sol = draw_optimum(ax, g, d, solver=CLEAR)

    expected = to_plot(T, sol.q)
    marker = next(line for line in ax.lines if line.get_marker() == "o")
    assert np.allclose(marker.get_xydata()[0], expected, atol=1e-6)
    # and it is genuinely optimal
    assert d @ sol.q == pytest.approx(sol.value, rel=1e-7)


def test_draw_constraints_draws_one_line_per_enforced_row(ax):
    _, g = toy.MODELS["derate"]
    before = len(ax.lines)
    draw_constraints(ax, g)
    assert len(ax.lines) - before == int(np.isfinite(g.b).sum())


def test_vertical_constraints_are_not_a_special_case(ax):
    """A row with no dependence on the vertical axis plots as a vertical line;
    solving for whichever coordinate has the larger coefficient handles it.

    Against ``_draw_line`` directly: this is its behaviour, and the public
    ``draw_halfplane`` that used to be the way in here was deleted for having no
    callers -- it drew bid bounds and generation limits, which are not network
    feasibility and which no figure ever ended up drawing."""
    before = len(ax.lines)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    _draw_line(ax, np.array([1.0, 0.0]), 50.0, xlim, ylim)  # pure vertical
    _draw_line(ax, np.array([0.0, 1.0]), 50.0, xlim, ylim)  # pure horizontal
    assert len(ax.lines) - before == 2
    for line in ax.lines[before:]:
        assert np.all(np.isfinite(line.get_xydata()))


def test_a_full_figure_composes(ax):
    """The layers are meant to stack: DAM, FTR, their meet, and both optima."""
    f, g = toy.MODELS["mixed"]
    d = clear_dam(g, toy.SCENARIOS["(a)"], solver=CLEAR).direction
    draw_region(ax, g, label="Q(g)")
    draw_region(ax, f, label="Q(f)", color="C4", ls="dotted")
    draw_region(ax, meet(f, g), label="Q(f^g)", color="C1", ls="dashed", fill_alpha=0.0)
    draw_optimum(ax, g, d, solver=CLEAR)
    draw_optimum(ax, f, d, solver=CLEAR, color="C4", ls="dotted")
    label_axes(ax, g)
    assert ax.get_xlabel() == "$q_{S}$"
    assert ax.get_ylabel() == "$q_{C}$"


def test_frame_axes_sets_limits_before_the_layers_read_them(ax):
    """The framing layer has to come first: draw_constraints clips its lines to
    the *current* limits, so a figure that frames afterwards gets lines that stop
    short.  This pins the contract -- frame_axes sets what it is given, and
    defers to label_axes for the names it is not given."""
    _, g = toy.MODELS["derate"]
    frame_axes(ax, g, title="a title", xlim=(-10, 20), ylim=(-30, 40))
    assert ax.get_xlim() == (-10, 20)
    assert ax.get_ylim() == (-30, 40)
    assert ax.get_title() == "a title"
    assert ax.get_xlabel() == "$q_{S}$"  # no xlabel passed -> label_axes default


def test_frame_axes_leaves_out_what_it_is_not_given(ax):
    """Limits and title are optional, so a caller can frame only the names --
    and ``ylabel=""`` blanks the name on the inner column of a shared-y grid."""
    _, g = toy.MODELS["derate"]
    ax.set_xlim(-5, 5)
    frame_axes(ax, g, xlabel="L", ylabel="")
    assert ax.get_xlim() == (-5, 5)  # untouched
    assert ax.get_title() == ""
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("L", "")


def test_label_axes_refuses_a_positional_basis(ax):
    """``label_axes`` used to accept a ``T`` it never read.  Axis *names* depend
    only on which node was eliminated, so the parameter is gone -- keyword-only,
    so a stale ``label_axes(ax, model, T)`` fails loudly instead of being read as
    ``drop``."""
    _, g = toy.MODELS["derate"]
    T = free_basis(3, g.network.slack_idx)
    with pytest.raises(TypeError):
        label_axes(ax, g, T)

    label_axes(ax, g, drop=2)  # eliminate L -> the kept nodes are S and C
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("$q_{S}$", "$q_{C}$")
