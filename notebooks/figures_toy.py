# %%
"""Figures for the 3-node toy, one plot per file, written as PNG to
``notebooks/figures/``.

At three nodes power balance leaves a 2-D injection space, so every picture here
is exact -- not a projection.  Each figure stacks layers from ``ftr_align.viz``
onto an axis, so a variant is a matter of adding or dropping a call.

**Axes.**  Not two node injections but *total load served* against *solar
dispatch*, which is how the conference figures read: the x-axis is how much load
is met and the y-axis is how it is split.  That is a basis ``T`` with
``1^T T = 0`` like any other -- see :func:`basis_from_columns` -- and the
constraint lines need no correction for it, since substituting ``q = T u`` turns
row ``i`` into ``(T^T k_i)^T u <= b_i``, i.e. just ``K T``.

**Colour is the element, style is the model.**  A line keeps its colour across
contingencies, models and figures; solid is the DAM model and dotted the FTR one.

Every figure names both the case and the scenario, in its title and its
filename, so nothing is fixed silently.

Note the layers read the *current* axis limits, so set ``xlim``/``ylim`` first.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from ftr_align import clear_dam, meet
from ftr_align.cases import toy
from ftr_align.metrics import row_labels
from ftr_align.polytope import basis_from_columns, faces
from ftr_align.viz import (
    draw_constraints,
    draw_optimum,
    draw_region,
    label_axes,
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
XLIM, YLIM = (-40, 260), (-60, 200)

DAM_LS, FTR_LS, MEET_LS = "solid", "dotted", "dashed"


def new_axis(model, title):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axhline(0, lw=0.4, c="k")
    ax.axvline(0, lw=0.4, c="k")
    label_axes(ax, model, xlabel=XLABEL, ylabel=YLABEL)
    ax.set_title(title, fontsize=11)
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png")


def slug(scenario):
    """'(a)' -> 'a', for filenames."""
    return scenario.strip("()")


# %%
# -------------------------------------
# One figure per (case, scenario)
# -------------------------------------
# The geometry is fixed by the case; only the direction -- and so the two
# maximisers and their supporting hyperplanes -- changes with the scenario.
for case, (f_model, g_model) in toy.MODELS.items():
    for scenario in toy.SCENARIOS:
        direction = clear_dam(g_model, toy.SCENARIOS[scenario], solver=CLEAR).direction
        fig, ax = new_axis(g_model, f"{case}   scenario {scenario}")

        # Filled but not stroked: the coloured constraint lines below already
        # trace each region's boundary, and a second outline would hide which
        # element bounds where.
        draw_region(ax, g_model, T, label=r"$\mathcal{Q}(g)$  DAM",
                    color="grey", ls=DAM_LS, fill_alpha=0.22, outline=False)
        draw_region(ax, f_model, T, label=r"$\mathcal{Q}(f)$  FTR",
                    color="C4", ls=FTR_LS, fill_alpha=0.22, outline=False)

        draw_constraints(ax, g_model, T, ls=DAM_LS, lw=1.0, names=False)
        draw_constraints(ax, f_model, T, ls=FTR_LS, lw=1.0, names=False)

        # The intersection is not the boundary of either model, so it keeps an
        # outline of its own.  Unlabelled: it is always the tighter of the two
        # regions already drawn, so the legend would add a third entry for
        # something the picture states.
        draw_region(ax, meet(f_model, g_model), T,
                    color="k", ls=MEET_LS, fill_alpha=0.0, lw=1.6)

        draw_optimum(ax, g_model, direction, T, solver=CLEAR, color="grey", ls=DAM_LS)
        draw_optimum(ax, f_model, direction, T, solver=CLEAR, color="C4", ls=FTR_LS)

        ax.legend(fontsize=8, loc="upper left", frameon=False)
        save(fig, f"toy_{case}_{slug(scenario)}")


# %%
# -------------------------------------
# One figure per model: which constraint bounds where
# -------------------------------------
# draw_region shows only the envelope.  Here every half-space is drawn and named,
# colour by element and style by contingency, including the ones that never bind
# -- CL at 300 MW only clips the corner of the window, which is itself the point.
for case in ("dam_outage",):
    f_model, g_model = toy.MODELS[case]
    for name, model in (("g (DAM)", g_model), ("f (FTR)", f_model)):
        fig, ax = new_axis(model, f"{case}   {name}   constraints")
        draw_region(ax, model, T, color="grey", ls="solid", fill_alpha=0.25)
        draw_constraints(ax, model, T, lw=1.1)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        save(fig, f"toy_constraints_{case}_{name.split()[0]}")


# %%
# -------------------------------------
# The regime map
# -------------------------------------
# Every vertex of Q(b) is a realizable active set, and faces() returns it
# labelled with the rows tight there -- so this is every congestion regime the
# model admits, laid out on the polytope.  No scenario: it holds for all of them.
_, g_model = toy.MODELS["dam_outage"]
fig, ax = new_axis(g_model, "dam_outage   g (DAM)   every realizable active set")
draw_region(ax, g_model, T, color="grey", ls="solid", fill_alpha=0.25)
draw_constraints(ax, g_model, T, lw=0.8)
for face in faces(g_model):
    u = to_plot(T, face.q)
    ax.plot(*u, marker="o", ms=7, color="k", zorder=6, ls="none")
    ax.annotate(
        "\n".join(row_labels(g_model, face.rows)),
        u, textcoords="offset points", xytext=(9, 9), fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
    )
ax.legend(fontsize=8, loc="upper left", frameon=False)
save(fig, "toy_regimes_dam_outage")
