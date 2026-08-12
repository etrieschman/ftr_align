# %%
"""Figures for the 3-node toy, written to ``notebooks/figures/``.

At three nodes power balance leaves a 2-D injection space, so every picture here
is exact -- not a projection.  Each figure is built by stacking layers from
``ftr_align.viz`` onto an axis, so a variant is a matter of dropping or adding a
call rather than editing a plotting routine.

Note the layers read the *current* axis limits (a constraint line is drawn across
whatever window is set), so set ``xlim``/``ylim`` before calling them.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ftr_align import clear_dam, meet
from ftr_align.cases import toy
from ftr_align.metrics import row_labels
from ftr_align.polytope import faces
from ftr_align.viz import (
    DAM_STYLE,
    FTR_STYLE,
    MEET_STYLE,
    draw_constraints,
    draw_optimum,
    draw_region,
    label_axes,
)

CLEAR = {"solver": "CLARABEL"}
XLIM, YLIM = (-170, 170), (-210, 210)
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def frame(ax, model, title=None):
    """Common axis furniture."""
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axhline(0, lw=0.4, c="k")
    ax.axvline(0, lw=0.4, c="k")
    label_axes(ax, model)
    if title:
        ax.set_title(title, fontsize=10)


def direction_for(g_model, scenario):
    """The realized DAM certificate's node-space direction."""
    return clear_dam(g_model, toy.SCENARIOS[scenario], solver=CLEAR).direction


def draw_pair(ax, f_model, g_model, direction, legend=False):
    """DAM, FTR and their intersection, with each model's support maximiser."""
    draw_region(ax, g_model, label=r"$\mathcal{Q}(g)$  DAM", **DAM_STYLE)
    draw_region(ax, f_model, label=r"$\mathcal{Q}(f)$  FTR", **FTR_STYLE)
    draw_region(ax, meet(f_model, g_model), label=r"$\mathcal{Q}(f \wedge g)$", **MEET_STYLE)
    draw_optimum(ax, g_model, direction, solver=CLEAR, **DAM_STYLE)
    draw_optimum(ax, f_model, direction, solver=CLEAR, **FTR_STYLE)
    if legend:
        ax.legend(fontsize=8, loc="upper left", frameon=False)


# %%
# -------------------------------------
# FIG 1: the four model differences, one scenario
# -------------------------------------
SCENARIO = "(a)"
fig, axes = plt.subplots(1, len(toy.MODELS), figsize=(4.4 * len(toy.MODELS), 4.2))
for ax, (case, (f_model, g_model)) in zip(axes, toy.MODELS.items()):
    frame(ax, g_model, f"{case}  {SCENARIO}")
    draw_pair(ax, f_model, g_model, direction_for(g_model, SCENARIO), legend=ax is axes[0])
fig.tight_layout()
fig.savefig(OUT / "toy_cases.pdf", bbox_inches="tight")
fig.savefig(OUT / "toy_cases.png", dpi=140, bbox_inches="tight")


# %%
# -------------------------------------
# FIG 2: every case against every scenario
# -------------------------------------
# Only the direction changes down a column, so this shows how much of the story
# is geometry (fixed per column) and how much is the realized certificate.
fig, axes = plt.subplots(
    len(toy.SCENARIOS), len(toy.MODELS),
    figsize=(4.3 * len(toy.MODELS), 4.1 * len(toy.SCENARIOS)),
)
for row, scenario in zip(axes, toy.SCENARIOS):
    for ax, (case, (f_model, g_model)) in zip(row, toy.MODELS.items()):
        frame(ax, g_model, f"{case}  {scenario}")
        draw_pair(ax, f_model, g_model, direction_for(g_model, scenario))
axes[0][0].legend(fontsize=8, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "toy_grid.pdf", bbox_inches="tight")
fig.savefig(OUT / "toy_grid.png", dpi=110, bbox_inches="tight")


# %%
# -------------------------------------
# FIG 3: which constraint bounds where
# -------------------------------------
# draw_region shows only the envelope; draw_constraints shows the individual
# half-spaces, including the ones that never bind (CL at 300 MW sits far outside).
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
f_model, g_model = toy.MODELS["dam_outage"]
for ax, (name, model) in zip(axes, [("g (DAM: base + SC outage)", g_model),
                                    ("f (FTR: base only)", f_model)]):
    frame(ax, model, name)
    draw_region(ax, model, **DAM_STYLE)
    draw_constraints(ax, model, lw=0.8)
    ax.legend(fontsize=7, loc="upper left", frameon=False, ncols=2)
fig.tight_layout()
fig.savefig(OUT / "toy_constraints.pdf", bbox_inches="tight")
fig.savefig(OUT / "toy_constraints.png", dpi=140, bbox_inches="tight")


# %%
# -------------------------------------
# FIG 4: the regime map
# -------------------------------------
# Each vertex of Q(b) is a realizable active set, and faces() returns it labelled
# with the rows tight there.  Annotating them turns the polytope into a map of
# every congestion regime the model admits.
fig, ax = plt.subplots(figsize=(7.5, 6.5))
_, g_model = toy.MODELS["dam_outage"]
frame(ax, g_model, "regimes of Q(g) -- every realizable active set")
draw_region(ax, g_model, **DAM_STYLE)
for face in faces(g_model):
    u = face.q[[0, 1]]  # drop-slack basis: the first two node coordinates
    ax.plot(*u, marker="o", ms=6, color="C3", mec="black", mew=0.6, zorder=5, ls="none")
    ax.annotate(
        "\n".join(row_labels(g_model, face.rows)),
        u, textcoords="offset points", xytext=(8, 8), fontsize=7, color="C3",
    )
fig.tight_layout()
fig.savefig(OUT / "toy_regimes.pdf", bbox_inches="tight")
fig.savefig(OUT / "toy_regimes.png", dpi=140, bbox_inches="tight")

print(f"wrote {len(list(OUT.glob('*.pdf')))} figures to {OUT}")
