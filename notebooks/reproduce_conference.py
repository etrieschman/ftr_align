# %%
"""Reproduce the PowerUp conference paper's tables from the current library.

Kept separate from ``explore_toy`` because this is not analysis -- it is a fixed
target.  The numbers here are printed in the paper, so this notebook either
matches them or something has regressed.

``dual_summary`` lives here rather than in ``ftr_align.metrics`` for the same
reason: it is a join shaped to one paper's table, not a general reporting tool.
The reusable half is ``metrics.net_dual``, which implements the signed-collapse
convention and is imported below.

**Solve with CLARABEL.**  The realized ``y*`` is not unique on these toy
patterns, and the paper's numbers are the analytic-centre certificate that an
interior-point solver produces; simplex gives an equally valid vertex dual with a
different split across rows.
"""

import polars as pl

from ftr_align import SupportProblem, clear_dam
from ftr_align.cases import toy
from ftr_align.metrics import net_dual
from ftr_align.solve import CENTER

pl.Config.set_tbl_rows(40)


def dual_summary(f_model, sol_f, g_model, sol_g, labels=None):
    """Table III: signed net duals ``mu_f`` (FTR) and ``mu_g`` (DAM) per
    (contingency, element), joined.  ``labels`` adds constant metadata columns."""
    left = net_dual(f_model, sol_f.mu).rename({"mu_signed": "mu_f"})
    right = net_dual(g_model, sol_g.mu).rename({"mu_signed": "mu_g"})
    out = left.join(right, on=["contingency", "element"], how="full", coalesce=True)
    if labels:
        out = out.with_columns(**{k: pl.lit(v) for k, v in labels.items()})
    return out


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
