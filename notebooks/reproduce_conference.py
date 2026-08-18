# %%
"""The PowerUp conference paper's tables and figures.

A fixed target, not analysis: these numbers are printed in the paper.

``net_dual`` and ``dual_summary`` live here rather than in ``ftr_align.metrics``
because they are shaped to one paper's table.  The signed collapse is a
reporting convention only -- the package keeps ``mu`` stacked-nonnegative.

Solve with CLARABEL: the realized ``y*`` is not unique on these patterns, and
the paper's numbers are the analytic-centre certificate.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D

from ftr_align import SupportProblem, clear_dam
from ftr_align.cases import toy
from ftr_align.metrics import row_labels, gap_summary
from ftr_align.network import NetworkModel, contingency_label, element_label
from ftr_align.polytope import basis_from_columns, faces
from ftr_align.solve import CENTER
from ftr_align.viz import (
    draw_constraints,
    draw_region,
    frame_axes,
    to_plot,
)

pl.Config.set_tbl_rows(40)

NET_DUAL_TOL = 0.5  # drop sub-dollar net duals from the reported table


def net_dual(model: NetworkModel, mu: np.ndarray) -> pl.DataFrame:
    """Collapse stacked ``mu`` to a signed net dual per (contingency, element):
    ``mu_upper - mu_lower``.  Rows with ~zero net are dropped.

    The column is ``mu_signed`` rather than ``mu`` because two different things
    travel under that letter: this one, and the raw stacked multipliers that
    ``metrics.constraint_table`` reports one row per side.  They are not interchangeable
    and the name should say which you are holding."""
    names = model.network.element_names
    records = []
    for c in model.contingencies:
        net = mu[model.rows_upper(c.key)] - mu[model.rows_lower(c.key)]
        for e in range(model.ell):
            if abs(net[e]) > NET_DUAL_TOL:
                records.append(
                    {
                        "contingency": contingency_label(c.key, names),
                        "element": element_label(names, e),
                        "mu_signed": float(net[e]),
                    }
                )
    return pl.DataFrame(
        records,
        schema={
            "contingency": pl.Utf8,
            "element": pl.Utf8,
            "mu_signed": pl.Float64,
        },
    )


def dual_summary(f_model, sol_f, g_model, sol_g, labels=None):
    """Table III: signed net duals ``mu_f`` (FTR) and ``mu_g`` (DAM) per
    (contingency, element), joined.  ``labels`` adds constant metadata columns."""
    left = net_dual(f_model, sol_f.mu).rename({"mu_signed": "mu_f"})
    right = net_dual(g_model, sol_g.mu).rename({"mu_signed": "mu_g"})
    out = left.join(right, on=["contingency", "element"], how="full", coalesce=True)
    if labels:
        out = out.with_columns(**{k: pl.lit(v) for k, v in labels.items()})
    return out


def _dollars(*names):
    """Round to whole dollars, with anything rounding to zero printed as ``0.0``.

    The explicit branch is not decoration: a failure mode that is structurally
    zero comes back as a solver's ``-1e-13``, and plain rounding renders that
    ``-0.0`` -- which reads as a signed quantity, and ``U``/``V`` are one-signed
    by construction."""
    return [
        pl.when(pl.col(c).abs() < 0.5)
        .then(pl.lit(0.0))
        .otherwise(pl.col(c).round(0))
        .alias(c)
        for c in names
    ]


# %%
# -------------------------------------
# TABLE II: FTR-DAM alignment
# -------------------------------------
# MS_DAM = h(g; y*) (prop:support -- the realized merchandising surplus), the gap
# Delta = h(f) - h(g), and the alignment ratio eta = h(f)/h(g).  The failure
# modes U and V come along for free: gap_summary already solves the intersection
# f ^ g, and Delta = U - V identically, so they are the paper's one number split
# into its two one-signed halves -- U underfunding exposure, V lost hedge value.
runs = []
for case, (f_model, g_model) in toy.MODELS.items():
    for scenario_name, scenario in toy.SCENARIOS.items():
        dam = clear_dam(g_model, scenario, solver=CENTER)
        runs.append(
            gap_summary(
                f_model,
                g_model,
                dam.direction,
                labels={"variation": case, "scenario": scenario_name},
                solver=CENTER,
            )
        )

table_ii = (
    pl.DataFrame(runs)
    .select(
        "variation",
        "scenario",
        pl.col("h_g").alias("MS_DAM"),
        pl.col("h_f").alias("MS_FTR"),
        "Delta",
        "relative_gap",
        "U",
        "V",
    )
    .with_columns(
        *_dollars("MS_DAM", "MS_FTR", "Delta", "U", "V"),
        pl.col("relative_gap").round(2),
    )
)
display(table_ii)

# %%
# -------------------------------------
# TABLE III: dual attribution
# -------------------------------------
# Per (contingency, element) net duals for both models, over every model
# difference and scenario -- on the plain 3-node and on the double-circuit
# variant, where the parallel SLa/SLb pair makes the optimal dual face
# non-singleton and mu trades between the two rows.
for models in (toy.MODELS, toy.REDUNDANT_MODELS):
    frames = []
    for case, (f_model, g_model) in models.items():
        for scenario_name, scenario in toy.SCENARIOS.items():
            dam = clear_dam(g_model, scenario, solver=CENTER)
            sol_f = SupportProblem(f_model, dam.direction).solve(solver=CENTER)
            sol_g = SupportProblem(g_model, dam.direction).solve(solver=CENTER)
            frames.append(
                dual_summary(
                    f_model,
                    sol_f,
                    g_model,
                    sol_g,
                    labels={"variation": case, "scenario": scenario_name},
                )
            )
    table = (
        pl.concat(frames)
        .unpivot(
            index=["variation", "scenario", "contingency", "element"],
            value_name="mu",
        )
        .pivot(
            index=["variation", "scenario", "variable"],
            on=["contingency", "element"],
            values="mu",
        )
        .sort(by=["variation", "scenario", "variable"])
    )
    display(table)


# %%
# -------------------------------------
# FIGURES: the 3-node injection polytopes
# -------------------------------------
# The same layers as ``figures_toy``, which stays the place that writes the PNGs
# for the paper.  Here they are inline alongside the tables they illustrate, and
# ``draw_optimum`` is deliberately left out: the paper's figures show the two
# *regions* and where they disagree, and the maximiser plus its supporting
# hyperplane belongs to the support-geometry figure (hand-drawn in TikZ).  The
# calls are kept, commented, so switching them on is uncommenting.  Because that
# is the only layer a scenario changes, there is one figure per *case* here, not
# one per (case, scenario).
#
# At three nodes power balance leaves a 2-D injection space, so every picture is
# exact -- not a projection.
#
# **Axes.**  Not two node injections but *total load served* against *solar
# dispatch*, which is how the conference figures read.  That is a basis ``T``
# with ``1^T T = 0`` like any other, and the constraint lines need no correction
# for it: substituting ``q = T u`` turns row ``i`` into ``(T^T k_i)^T u <= b_i``,
# i.e. just ``K T``.
#
# **Colour is the element, dash is the contingency, width is the model.**  Style
# is not spent on the model, because where both models enforce the same
# contingency at the same limit the two rows are the identical line and a
# per-model style would draw one line twice.  Width carries the model instead,
# which still separates the two base lines in ``derate``/``mixed``, where the FTR
# limits are derated and the rows are parallel rather than coincident.
#
# Note the layers read the *current* axis limits, so set ``xlim``/``ylim`` first.

# Column one says "serve one more unit of load, from coal" -> q_C up, q_L down.
# Column two says "shift one unit of that generation from coal to solar" -> q_S
# up, q_C down.  Both balanced.
T = basis_from_columns([(0.0, 1.0, -1.0), (1.0, -1.0, 0.0)])
XLABEL, YLABEL = r"$L$  (load served, MW)", r"$q_S$  (solar dispatch, MW)"
XLIM, YLIM = (-100, 200), (-100, 200)

DAM_LS, FTR_LS = "solid", "dotted"
DAM_LW, FTR_LW = 1.8, 0.9  # model = width, so dash is free to mean contingency


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
# One figure per case.  The geometry is fixed by the
# case; the scenario enters only through the direction ``d``, and the only layer
# that draws ``d`` is ``draw_optimum``.  With that off, as it is in the paper's
# figures, the three scenarios of a case are the same picture three times, so the
# loop is over cases alone.
for case, (f_model, g_model) in toy.MODELS.items():
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    frame_axes(
        ax,
        g_model,
        title=case,
        xlim=XLIM,
        ylim=YLIM,
        xlabel=XLABEL,
        ylabel=YLABEL,
        fontsize=11,
    )

    # Filled but not stroked: the coloured constraint lines below already trace
    # each region's boundary, and a second outline would hide which element
    # bounds where.
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

    # The maximiser and its supporting hyperplane, left off deliberately: the
    # paper's region figures do not draw them, and the support geometry is its
    # own figure (fig_support_vi, hand-drawn in TikZ).  To switch them back on,
    # add ``draw_optimum`` to the ``ftr_align.viz`` import above, uncomment these
    # three lines, and put the scenario loop back -- they are the only thing that
    # makes one scenario's picture differ from another's.
    # direction = clear_dam(g_model, toy.SCENARIOS["(a)"], solver=CENTER).direction
    # draw_optimum(ax, g_model, direction, T, solver=CENTER, color="grey", ls=DAM_LS)
    # draw_optimum(ax, f_model, direction, T, solver=CENTER, color="C4", ls=FTR_LS)

    legend(ax, extra=width_key())
    display(fig)
    plt.close(fig)


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
