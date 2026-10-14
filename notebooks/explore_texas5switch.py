# %% [markdown]
# # texas5 with a station switch: the ex-ante design ladder
#
# Station H is split into H1 and H2 joined by a switch.  Each hour's DAM ran with
# the switch closed or open; the FTR model is one network `Q(y, b)` that must
# serve every hour.  Every rung of the ladder is the same evaluation, `E[U]` and
# `E[V]` over the hours, with a different FTR model.  Findings: `findings_design.md`.
#
# The hours (`texas5switch.Q` and `four_hours`):
#
# | hour | DAM switch | binding lines | story |
# |---|---|---|---|
# | 1 | closed | DH, WD | evening: strong west wind, Houston load high |
# | 2 | open | NH, ND | midday: north wind and solar surging |
# | 3 | closed | SD, SH | evening: south gas up |
# | 4 | open | DH, WD | midday, south; shares hour 1's pattern |
#
# `U_t`, `V_t` are hour `t`'s failure modes.  The FTR model's limits are `b`
# (lines, and the switch in each direction); hour `t`'s DAM limits are `b_t`,
# here one rating vector for every hour.  Sections:
# 1. The instance: the PTDF at each switch state; the four hours and their one rating vector.
# 2. The ladder: B1, B2, S1 at both states, the tau frontier, S3 over the FTR switch susceptance.

import numpy as np
import polars as pl

from ftr_align.cases import texas5switch as t5s
from ftr_align.cases.texas5switch import (
    CLOSED,
    OPEN,
    ELEMENT_NAMES,
    LINE_NAMES,
    NODE_NAMES,
)
from ftr_align.network import intersection, with_limits

np.set_printoptions(precision=3, suppress=True, linewidth=120)
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_float_precision(2)


def ladder_row(label, ftr, scenarios, **extra):
    """One row of the ladder: E[U], E[V], then U_t and V_t per hour."""
    tbl = t5s.evaluate(ftr, scenarios)
    w = tbl["weight"].to_numpy()
    return {
        "design": label,
        **extra,
        "E[U]": float(w @ tbl["U"].to_numpy()),
        "E[V]": float(w @ tbl["V"].to_numpy()),
        **{f"U_{sc.label}": u for sc, u in zip(scenarios, tbl["U"])},
        **{f"V_{sc.label}": v for sc, v in zip(scenarios, tbl["V"])},
    }


def switch_limits(d):
    """A design's switch limits as 'H1->H2 / H2->H1'."""
    r = t5s.n_lines + t5s.n_switches
    return f"{d.b[t5s.n_lines]:.1f} / {d.b[r + t5s.n_lines]:.1f}"


def lines_below_rating(d, b_line):
    return {
        e: round(float(bb), 1)
        for e, bb, r in zip(LINE_NAMES, d.b[: t5s.n_lines], b_line)
        if bb < r - 1e-3
    }


# %% [markdown]
# ## 1. The instance
#
# ### 1.1 One PTDF, three switch regimes
#
# Columns: inject 1 MW at this node, withdraw it at the slack D.  The last row is
# the flow on the switch, positive H1 -> H2.  A relaxed switch at a large
# susceptance reproduces the closed PTDF row for row.

for name, y in (("closed", CLOSED), ("open", OPEN)):
    H = t5s.switch_ptdf(y)
    print(f"FTR switch {name} (y_H = {y:g})")
    print(
        pl.DataFrame(
            {
                "element": list(ELEMENT_NAMES),
                **{nm: H[:, i] for i, nm in enumerate(NODE_NAMES)},
            }
        )
    )
print(
    "max |relaxed(1e6) - closed| =",
    np.abs(t5s.switch_ptdf(1e6) - t5s.switch_ptdf(CLOSED)).max(),
)

# %% [markdown]
# ### 1.2 The four hours and their one rating vector
#
# The injections are chosen by hand (the table in `texas5switch`); the rest is
# computed.  A binding line's rating is its flow in its own hour; every other line
# gets its largest flow over the four hours plus 10%.  Each hour's direction `v_t`
# is the signed sum of its binding lines' PTDF rows under its own switch state.
#
# The flows table shows every hour's injection under every hour's switch state.
# `cross` says which injections are infeasible under the other switch state at
# these ratings: only hour 2's, under the closed switch.  That is the only place
# `V` can come from against B2 (finding 2).

B_LINE, SCENARIOS, cross = t5s.four_hours(margin=0.10)
FENCE = t5s.rating_fence(
    B_LINE, b_switch=100.0
)  # a line at most its rating; the switch at most 100

print(
    pl.DataFrame(
        {"node": list(NODE_NAMES), **{f"q_{sc.label}": sc.q for sc in SCENARIOS}}
    )
)
print(t5s.flows_table(SCENARIOS))
print(pl.DataFrame({"line": list(LINE_NAMES), "rating": B_LINE}))
print(
    "hour's injection infeasible under the switch state of hour:",
    {k: [j for j, ok in v.items() if not ok] for k, v in cross.items()},
)
for sc in SCENARIOS:
    state = "closed" if np.isinf(sc.y_switch) else "open"
    print(
        f"hour {sc.label} (DAM switch {state}, binds {', '.join(sc.binding)}): v =",
        np.round(sc.direction, 3),
    )

# %% [markdown]
# ## 2. The ladder
#
# ### 2.1 Baselines: B1 derates one switch state, B2 stacks both
#
# B1: the FTR model is one switch state with every line limit times `alpha` and
# the switch unlimited.  With the FTR switch closed and unlimited, `U` is
# infinite in the open-switch hours 2 and 4 (finding 1).  B2: the stacked
# intersection of both states' DAM models at the ratings, the "include every
# planned outage" practice; `U = 0` by construction, and `V` comes only from
# hour 2 (finding 2).

B2 = intersection(t5s.model(CLOSED, B_LINE), t5s.model(OPEN, B_LINE))
rows = [ladder_row("B2 stack", B2, SCENARIOS)]
for name, y in (("closed", CLOSED), ("open", OPEN)):
    for a in (1.0, 0.9, 0.8, 0.7):
        rows.append(
            ladder_row(
                f"B1 FTR switch {name}",
                t5s.model(y, b_line=a * B_LINE),
                SCENARIOS,
                alpha=a,
            )
        )
print(pl.DataFrame(rows))

# %% [markdown]
# ### 2.2 S1: design the limits at a fixed FTR switch state
#
# Line limits symmetric and fenced at the rating.  The dual vertex per hour is
# chosen by binaries (`design_limits_milp`, memo eq. design-b-vertex);
# `tests/test_texas5switch.py` checks it against full enumeration.
#
# What the design may do with the switch's limit is a toggle, `switch=`:
# `"symmetric"` designs one number for both directions; `"per_direction"`
# designs the two directions separately.  With the FTR switch open the switch is
# absent and the toggle changes nothing.  With it closed and fixed unlimited,
# hours 2 and 4 make `U` infinite (finding 1) and no line limits reach `tau = 0`.

rows = []
for name, y in (("closed", CLOSED), ("open", OPEN)):
    for switch, b_switch in (
        ("symmetric", np.inf),
        ("per_direction", np.inf),
    ):
        template = t5s.model(
            y, b_line=B_LINE, b_switch=b_switch
        )  # line limits are placeholders; the design replaces them
        try:
            d = t5s.design_limits_milp(
                template, SCENARIOS, tau=0.0, b_max=FENCE, switch=switch, time_limit=60
            )
        except ValueError as err:
            print(
                {
                    "design": f"S1 FTR switch {name}",
                    "switch": switch,
                    "result": str(err),
                }
            )
            continue
        rows.append(
            ladder_row(
                f"S1 FTR switch {name}",
                with_limits(template, d.b),
                SCENARIOS,
                switch=switch,
                switch_limits=switch_limits(d),
                lines_lowered=str(lines_below_rating(d, B_LINE)),
            )
        )
print(pl.DataFrame(rows, strict=False))

# %% [markdown]
# ### 2.3 The tau frontier at both FTR switch states
#
# A budget `tau > 0` lets limits sit above where `U = 0` held them.  FTR switch
# closed (switch limit designed, symmetric): `V` is all hour 2's topology-bounded
# `V`, and no budget touches it.  FTR switch open: `V` is all exposure-bounded,
# and the budget buys it back.

rows = []
for name, y in (("closed", CLOSED), ("open", OPEN)):
    template = t5s.model(y, b_line=B_LINE, b_switch=100.0)
    for tau in (0.0, 1.0, 2.0, 4.0, 8.0, 9.0):
        d = t5s.design_limits_milp(
            template, SCENARIOS, tau, b_max=FENCE, switch="symmetric", time_limit=60
        )
        rows.append(
            ladder_row(
                f"S1 FTR switch {name}",
                with_limits(template, d.b),
                SCENARIOS,
                tau=tau,
                switch_limits=switch_limits(d),
            )
        )
print(pl.DataFrame(rows))

# %% [markdown]
# ### 2.4 S3: the FTR switch at a susceptance between open and closed
#
# First: every line at its rating, the FTR switch at each susceptance.
# Hours 1 and 3 (DAM switch closed): `U` is the same at every `y_H` below
# infinity, and zero at infinity, where the FTR network is their own.  Hours 2
# and 4 (DAM switch open): `U` grows with `y_H` and is infinite at infinity
# (finding 1).  Hour 2's `V` is its topology-bounded `V`, rising along the curve
# of finding 7.  No other hour has `V`.
#
# Then: At each `y_H` the FTR switch is a line with reactance `1/y_H`, and the limits are
# designed at that susceptance (fenced, `tau = 0`, switch limit symmetric).  The ends are S1 open and S1
# closed.  Hour 2's topology-bounded `V` grows with `y_H`; hours 2 and 4's
# exposure-bounded `V` vanishes as soon as the switch exists, because the design
# limits the switch instead of the lines.  The lowest `E[V]` is at an interior
# susceptance.

yhs = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, np.inf)
# Every design at its rating
rows = []
for yh in yhs:
    rows.append(
        ladder_row("at ratings", t5s.model(yh, b_line=B_LINE), SCENARIOS, y_H=yh)
    )
print(pl.DataFrame(rows))

rows = []
for yh in yhs:
    template = t5s.model(yh, b_line=B_LINE, b_switch=100.0)
    d = t5s.design_limits_milp(
        template, SCENARIOS, 0.0, b_max=FENCE, switch="symmetric", time_limit=60
    )
    rows.append(
        ladder_row(
            f"S3 y_H={yh:g}",
            with_limits(template, d.b),
            SCENARIOS,
            y_H=yh,
            switch_limits=switch_limits(d),
            gap=d.gap,
        )
    )
print(pl.DataFrame(rows))

# %%
