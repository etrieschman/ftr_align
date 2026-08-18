# %%
"""Figures for the 3-node toy, written as PNG to ``notebooks/figures/``.

One file per case, scenarios as columns.  At three nodes the injection space is
2-D, so every picture is exact.

Axes are (load served, solar dispatch) -- a basis T with 1^T T = 0, which needs
no correction since q = T u turns row i into (T^T k_i)^T u <= b_i.

Colour is the element, dash is the contingency, width is the model.  Style is
not spent on the model because coincident rows would then be drawn twice.

Layers read the current axis limits, so frame the axis first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ftr_align import clear_dam, meet
from ftr_align.cases import toy
from ftr_align.metrics import row_labels
from ftr_align.polytope import basis_from_columns, faces
from ftr_align.viz import (
    draw_constraints,
    draw_halfplane,
    draw_optimum,
    draw_region,
    frame_axes,
    to_plot,
)

CLEAR = {"solver": "CLARABEL"}
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Axes: (total load served, solar dispatch).  Column one says "serve one more
# unit of load, from coal" -> q_C up, q_L down.  Column two says "shift one unit
# of that generation from coal to solar" -> q_S up, q_C down.  Both balanced.
T = basis_from_columns([(0.0, 1.0, -1.0), (1.0, -1.0, 0.0)])
XLABEL, YLABEL = r"$L$  (load served, MW)", r"$q_S$  (solar dispatch, MW)"
XLIM, YLIM = (-100, 200), (-100, 200)

DAM_LS, FTR_LS, MEET_LS = "solid", "dotted", "dashed"
DAM_LW, FTR_LW = 1.8, 0.9  # model = width, so dash is free to mean contingency


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png")


def width_key():
    """Proxy handles naming the width convention.

    Colour and dash each get a swatch for free, because every constraint entry
    carries them.  Width does not: the per-constraint entries are deduplicated to
    one handle apiece, so without these two nothing in the legend says which of a
    thick/thin pair is the FTR limit."""
    return [
        Line2D([], [], color="0.35", lw=DAM_LW, label="DAM limit  (thick)"),
        Line2D([], [], color="0.35", lw=FTR_LW, label="FTR limit  (thin)"),
    ]


def legend(ax, extra=(), **kw):
    """Legend with duplicate labels collapsed, first occurrence winning, on an
    opaque background.

    Both models are drawn, and where they enforce the same contingency at the
    same limit the rows coincide -- so the same ``base:SL`` label is registered
    twice for what is one visible line.  The background matters because these
    figures fill the window with lines that would otherwise run through the
    swatches and misrepresent them."""
    kw = {
        "fontsize": 8,
        "loc": "upper left",
        "frameon": True,
        "facecolor": "white",
        "framealpha": 0.9,
        "edgecolor": "none",
        **kw,
    }
    seen = {}
    for handle, text in zip(*ax.get_legend_handles_labels()):
        seen.setdefault(text, handle)
    for handle in extra:
        seen.setdefault(handle.get_label(), handle)
    ax.legend(seen.values(), seen.keys(), **kw)


# %%
# -------------------------------------
# One figure per case, scenarios as columns
# -------------------------------------
# The geometry is fixed by the case, so all three panels of a figure draw the
# same regions and the same constraint lines; only the direction -- and so the
# two maximisers and their supporting hyperplanes -- changes across the columns.
# Side by side is what makes that readable: the eye holds the region still and
# watches the optimum move.
#
# Shared axes, so the panels are directly comparable rather than each auto-scaled
# to its own contents, and only the left column carries the y label.
for case, (f_model, g_model) in toy.MODELS.items():
    fig, axes = plt.subplots(
        1,
        len(toy.SCENARIOS),
        figsize=(4.8 * len(toy.SCENARIOS), 5.0),
        sharex=True,
        sharey=True,
    )
    for col, (ax, scenario) in enumerate(zip(axes, toy.SCENARIOS)):
        direction = clear_dam(g_model, toy.SCENARIOS[scenario], solver=CLEAR).direction
        frame_axes(
            ax,
            g_model,
            title=f"scenario {scenario}",
            xlim=XLIM,
            ylim=YLIM,
            xlabel=XLABEL,
            ylabel=YLABEL if col == 0 else "",
            fontsize=11,
        )

        # Filled but not stroked: the coloured constraint lines below already
        # trace each region's boundary, and a second outline would hide which
        # element bounds where.
        draw_region(
            ax,
            g_model,
            T,
            label=r"$\mathcal{Q}(g)$  DAM",
            color="grey",
            ls=DAM_LS,
            fill_alpha=0.22,
            outline=False,
        )
        draw_region(
            ax,
            f_model,
            T,
            label=r"$\mathcal{Q}(f)$  FTR",
            color="C4",
            ls=FTR_LS,
            fill_alpha=0.22,
            outline=False,
        )

        # No `ls`: dash carries the contingency, so base and an outage row of the
        # same element are distinguishable.  The FTR pass is drawn second and
        # thinner, so a coincident pair reads as one line with a core.
        draw_constraints(ax, g_model, T, lw=DAM_LW, names=True)
        draw_constraints(ax, f_model, T, lw=FTR_LW, names=True)

        # The intersection is not the boundary of either model, so it keeps an
        # outline of its own.  Unlabelled: it is always the tighter of the two
        # regions already drawn, so the legend would add a third entry for
        # something the picture states.
        # draw_region(
        #     ax, meet(f_model, g_model), T, color="k", ls=MEET_LS, fill_alpha=0.0, lw=1.6
        # )

        draw_optimum(ax, g_model, direction, T, solver=CLEAR, color="grey", ls=DAM_LS)
        draw_optimum(ax, f_model, direction, T, solver=CLEAR, color="C4", ls=FTR_LS)

        # One legend for the figure, on the left panel: the three columns draw
        # the same models and the same rows, so the others would be copies.
        if col == 0:
            legend(ax, extra=width_key())

    fig.suptitle(case, fontsize=12)
    save(fig, f"toy_{case}")
    # to view in notebooks
    display(fig)

# %%
# -------------------------------------
# Market bounds against network feasibility
# -------------------------------------
# Q(b) is network feasibility only.  A market also obeys bounds on the axes --
# load served cannot be negative, solar cannot exceed nameplate -- which carry no
# PTDF row and are given in plot coordinates.  Passing them to draw_region cuts
# the fill; passing them to draw_halfplane draws their edges.
_, g_model = toy.MODELS["derate"]
SOLAR_CAP = 50.0

#   L >= 0      ->  -L <= 0   ->  a = (-1, 0), c = 0
#   q_S <= cap  ->                a = ( 0, 1), c = cap
BOUNDS = [((-1.0, 0.0), 0.0), ((0.0, 1.0), SOLAR_CAP)]
BOUND_STYLE = [
    ("k", r"$L \geq 0$  (load served)"),
    ("#C79000", rf"$q_S \leq {SOLAR_CAP:.0f}$  (solar nameplate)"),
]

fig, ax = plt.subplots(figsize=(6.4, 5.2))
frame_axes(
    ax,
    g_model,
    title="derate   g (DAM)   network feasibility under market bounds",
    xlim=XLIM,
    ylim=YLIM,
    xlabel=XLABEL,
    ylabel=YLABEL,
    fontsize=11,
)
# The network region, ghosted, so what the bounds remove stays visible.
draw_region(ax, g_model, T, color="grey", ls=DAM_LS, fill_alpha=0.10, outline=False)
draw_region(
    ax,
    g_model,
    T,
    label=r"$\mathcal{Q}(g)$ under bounds",
    color="grey",
    ls=DAM_LS,
    fill_alpha=0.35,
    outline=False,
    bounds=BOUNDS,
)
draw_constraints(ax, g_model, T, lw=DAM_LW, names=True)
for (a, c), (colour, text) in zip(BOUNDS, BOUND_STYLE):
    draw_halfplane(ax, a, c, label=text, color=colour, ls="dashdot", lw=1.6)

legend(ax)  # one model, so no width key
save(fig, "toy_market_bounds")
display(fig)


# %%
# -------------------------------------
# The regime map
# -------------------------------------
# Every vertex of Q(b) is a realizable active set, and faces() returns it
# labelled with the rows tight there -- so this is every congestion regime the
# model admits, laid out on the polytope.  No scenario: it holds for all of them.
regimes = [
    (r"$\mathcal{Q}(g)$ for dam_outage", toy.MODELS["dam_outage"][1]),
    (r"$\mathcal{Q}(f)$ for ftr_extra", toy.MODELS["extra_ftr"][0]),
]
for label, model in regimes:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    frame_axes(
        ax,
        model,
        title=f"{label}; every realizable active set",
        xlim=(-200, 200),
        ylim=(-200, 200),
        xlabel=XLABEL,
        ylabel=YLABEL,
        fontsize=11,
    )
    draw_region(ax, model, T, color="grey", ls="solid", fill_alpha=0.25)
    draw_constraints(ax, model, T, lw=0.8)
    for face in faces(model):
        u = to_plot(T, face.q)
        ax.plot(*u, marker="o", ms=7, color="k", zorder=6, ls="none")
        ax.annotate(
            "\n".join(row_labels(model, face.rows)),
            u,
            textcoords="offset points",
            xytext=(9, 9),
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
        )
    legend(ax)  # one model, so no width key
    display(fig)
    plt.close(fig)

# %%
