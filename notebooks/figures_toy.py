# %%
"""Figures for the 3-node toy, written as PNG to ``notebooks/figures/``.

At three nodes power balance leaves a 2-D injection space, so every picture here
is exact -- not a projection.  Each figure stacks layers from ``ftr_align.viz``
onto an axis, so a variant is a matter of adding or dropping a call.

**Axes.**  Not two node injections but *total load served* against *solar
dispatch*, which is how the conference figures read: the x-axis is how much load
is met and the y-axis is how it is split.  That is a basis
``T`` with ``1^T T = 0`` like any other -- see :func:`basis_from_columns` -- and
the constraint lines need no correction for it, since substituting ``q = T u``
turns row ``i`` into ``(T^T k_i)^T u <= b_i``, i.e. just ``K T``.

**Colour is the element, style is the model.**  A line keeps its colour across
contingencies, models and figures; solid is the DAM model and dotted the FTR one.

Note the layers read the *current* axis limits, so set ``xlim``/``ylim`` first.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def frame(ax, model, title=None):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axhline(0, lw=0.4, c="k")
    ax.axvline(0, lw=0.4, c="k")
    label_axes(ax, model, xlabel=XLABEL, ylabel=YLABEL)
    if title:
        ax.set_title(title, fontsize=10)


def direction_for(g_model, scenario):
    return clear_dam(g_model, toy.SCENARIOS[scenario], solver=CLEAR).direction


def draw_pair(ax, f_model, g_model, direction, *, lines=True, legend=False):
    """Both regions, their intersection, the constraint lines that bound them,
    and each model's support maximiser.

    With ``lines=True`` the regions are filled but not stroked -- the coloured
    constraint lines already trace their boundaries, so an extra outline would
    just hide which element bounds where."""
    draw_region(ax, g_model, T, label=r"$\mathcal{Q}(g)$  DAM",
                color="grey", ls=DAM_LS, fill_alpha=0.22, outline=not lines)
    draw_region(ax, f_model, T, label=r"$\mathcal{Q}(f)$  FTR",
                color="C4", ls=FTR_LS, fill_alpha=0.22, outline=not lines)
    if lines:  # colour = element, style = model; drawn right across the window
        draw_constraints(ax, g_model, T, ls=DAM_LS, lw=1.0, names=False)
        draw_constraints(ax, f_model, T, ls=FTR_LS, lw=1.0, names=False)
    draw_region(ax, meet(f_model, g_model), T, label=r"$\mathcal{Q}(f \wedge g)$",
                color="k", ls=MEET_LS, fill_alpha=0.0, lw=1.6)
    draw_optimum(ax, g_model, direction, T, solver=CLEAR, color="grey", ls=DAM_LS)
    draw_optimum(ax, f_model, direction, T, solver=CLEAR, color="C4", ls=FTR_LS)
    if legend:
        ax.legend(fontsize=8, loc="upper left", frameon=False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=140, bbox_inches="tight")
    print(f"wrote {name}.png")


# %%
# -------------------------------------
# FIG 1: the four model differences, one scenario
# -------------------------------------
SCENARIO = "(a)"
fig, axes = plt.subplots(1, len(toy.MODELS), figsize=(4.6 * len(toy.MODELS), 4.2))
for ax, (case, (f_model, g_model)) in zip(axes, toy.MODELS.items()):
    frame(ax, g_model, f"{case}  {SCENARIO}")
    draw_pair(ax, f_model, g_model, direction_for(g_model, SCENARIO),
              legend=ax is axes[0])
save(fig, "toy_cases")


# %%
# -------------------------------------
# FIG 2: every case against every scenario
# -------------------------------------
# Only the direction changes down a column, so this separates what is geometry
# (fixed per column) from what is the realized certificate.
fig, axes = plt.subplots(
    len(toy.SCENARIOS), len(toy.MODELS),
    figsize=(4.5 * len(toy.MODELS), 4.1 * len(toy.SCENARIOS)),
)
for row, scenario in zip(axes, toy.SCENARIOS):
    for ax, (case, (f_model, g_model)) in zip(row, toy.MODELS.items()):
        frame(ax, g_model, f"{case}  {scenario}")
        draw_pair(ax, f_model, g_model, direction_for(g_model, scenario))
axes[0][0].legend(fontsize=8, loc="upper left", frameon=False)
save(fig, "toy_grid")


# %%
# -------------------------------------
# FIG 3: which constraint bounds where
# -------------------------------------
# draw_region shows only the envelope.  Here each half-space is drawn and named,
# colour by element and style by contingency, including the ones that never bind
# -- CL at 300 MW sits well outside the window, which is itself the point.
fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
f_model, g_model = toy.MODELS["dam_outage"]
for ax, (name, model) in zip(
    axes, [("g (DAM: base + SC outage)", g_model), ("f (FTR: base only)", f_model)]
):
    frame(ax, model, name)
    draw_region(ax, model, T, color="grey", ls="solid", fill_alpha=0.25)
    draw_constraints(ax, model, T, lw=1.1)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
save(fig, "toy_constraints")


# %%
# -------------------------------------
# FIG 4: the regime map
# -------------------------------------
# Every vertex of Q(b) is a realizable active set, and faces() returns it
# labelled with the rows tight there -- every congestion regime the model
# admits, laid out on the polytope.
fig, ax = plt.subplots(figsize=(8.5, 6.5))
_, g_model = toy.MODELS["dam_outage"]
frame(ax, g_model, "regimes of $\\mathcal{Q}(g)$ -- every realizable active set")
draw_region(ax, g_model, T, color="grey", ls="solid", fill_alpha=0.25)
draw_constraints(ax, g_model, T, lw=0.8, names=True)
for face in faces(g_model):
    u = to_plot(T, face.q)
    ax.plot(*u, marker="o", ms=7, color="k", zorder=6, ls="none")
    ax.annotate(
        "\n".join(row_labels(g_model, face.rows)),
        u, textcoords="offset points", xytext=(9, 9), fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
    )
ax.legend(fontsize=8, loc="upper left", frameon=False)
save(fig, "toy_regimes")
