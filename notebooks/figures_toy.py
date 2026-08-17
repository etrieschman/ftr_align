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

**Colour is the element, dash is the contingency, width is the model.**  A line
keeps its colour across contingencies, models and figures.  Style is *not* spent
on the model: where both models enforce the same contingency at the same limit
the two rows are the identical line, so a per-model style would be drawing one
line twice.  Width carries the model instead, which still separates the two base
lines in ``derate``/``mixed``, where the FTR limits are derated and the rows are
parallel rather than coincident.  The supporting hyperplanes in
:func:`draw_optimum` do stay solid/dotted -- one line per model, no contingency
to encode.

Every figure names both the case and the scenario, in its title and its
filename, so nothing is fixed silently.

Note the layers read the *current* axis limits, so set ``xlim``/``ylim`` first.
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
XLIM, YLIM = (-100, 200), (-100, 200)

DAM_LS, FTR_LS, MEET_LS = "solid", "dotted", "dashed"
DAM_LW, FTR_LW = 1.8, 0.9  # model = width, so dash is free to mean contingency


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

        legend(ax, extra=width_key())
        save(fig, f"toy_{case}_{slug(scenario)}")
        # to view in notebooks
        display(fig)


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
        u,
        textcoords="offset points",
        xytext=(9, 9),
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
    )
legend(ax)  # one model, so no width key
save(fig, "toy_regimes_dam_outage")
display(fig)

# %%
