# %% [markdown]
# # texas5 with a station switch: the ex-ante design ladder
#
# One network, two intervals with different switch states.  Every rung of the
# ladder is the same evaluation (E[U], E[V] over the scenarios) with a different
# FTR model `Q(y, b)`.

# %%
import numpy as np
import polars as pl

from ftr_align.cases import texas5switch as t5s
from ftr_align.cases.texas5switch import ELEMENT_NAMES, NODE_NAMES, switch_ptdf

np.set_printoptions(precision=3, suppress=True, linewidth=120)
pl.Config.set_tbl_rows(40)
pl.Config.set_float_precision(3)


def show(H, title):
    print(title)
    print(pl.DataFrame({"element": list(ELEMENT_NAMES), **{nm: H[:, i] for i, nm in enumerate(NODE_NAMES)}}))


# %% [markdown]
# ## Step 2: one PTDF, three switch regimes
#
# Columns are "inject 1 at this node, withdraw 1 at the slack D".  The switch row
# is the flow on the switch, positive H1 -> H2.

# %%
H_closed = switch_ptdf([np.inf])
H_open = switch_ptdf([0.0])
H_relaxed = switch_ptdf([1e6])
show(H_closed, "closed (y_H = inf)")
show(H_open, "open (y_H = 0)")
print("max |relaxed(1e6) - closed| =", np.abs(H_relaxed - H_closed).max())

# %% [markdown]
# Check against the bus-branch network: with the switch closed the line rows must
# equal the 5-bus PTDF lifted by `M`.

# %%
net, M = t5s.bus_network([1])
H_bus = net.ptdf()
print("max |H_lines - H_bus M| =", np.abs(H_closed[: t5s.n_lines] - H_bus @ M).max())

# %% [markdown]
# ## Step 3: scenarios
#
# Two hours of one contract week.  t1 (evening, switch closed): strong west wind,
# no solar, south gas up, data centre steady, Houston load high.  t2 (midday,
# switch open): north wind + solar surging, gas backed down, data centre ramped.
# Ratings: a binding line's rating is its flow in its own interval, every other
# line its largest flow over the intervals plus a margin.  `v_t` is the pattern's
# direction, constant on each DAM bus by construction.
#
# NOTE: midday and evening peak would be different FTR products; one contract here.
# `V_t > 0` needs interval t's injection to be infeasible under the other
# interval's state at the shared ratings -- `cross` reports exactly that.
# TODO: weight several injections / patterns later.  TODO: make SD bind in t2 too
# (completes the SDH triangle); held back over degeneracy worries.
#
# FINDING (version A): V_1 = 0 for B2 at ANY closed-interval injection.  The closed
# DAM prices H1 = H2, so the H1/H2 split is free; slide it until the switch flow is
# zero and the closed flow pattern is also an open flow pattern.  V can only come
# from the open interval, and only when its priced split is far from the closed
# network's natural one (H1 -80 / H2 -10 puts 29 MW on the closed switch).

# %%
CLOSED, OPEN = (1,), (0,)
#                    W     N     S     D      H1    H2
Q1 = np.array([120.,  40.,  40.,  -80.,  -60., -60.])
Q2 = np.array([ 90., 110.,  10., -120.,  -80., -10.])
INTERVALS = {  # label -> (state, injection, binding lines)
    "t1: H closed": (CLOSED, Q1, ["DH", "SH"]),
    "t2: H open": (OPEN, Q2, ["NH", "ND"]),
}
print(t5s.flows_table(INTERVALS))

# %%
B_LINE, SCENARIOS, cross = t5s.scenarios_from_injections(INTERVALS)
print(pl.DataFrame({"line": list(t5s.LINE_NAMES), "rating": B_LINE}))
print("cross-feasibility (injection row feasible under state column):", cross)
for sc in SCENARIOS:
    print(sc.label, "v =", np.round(sc.direction, 3))

# %% [markdown]
# ## Step 5: the baselines
#
# **B1** -- derate one physical state uniformly by `alpha`, switch limit left at
# infinity.  **B2** -- the stacked intersection of both intervals' physical models
# (the "include every planned outage" practice).

# %%
def ladder_row(label, ftr, scenarios=None, **extra):
    tbl = t5s.evaluate(ftr, SCENARIOS if scenarios is None else scenarios)
    EU, EV = t5s.expected(tbl)
    return {"design": label, **extra, "E[U]": EU, "E[V]": EV,
            **{f"U_{i+1}": u for i, u in enumerate(tbl["U"])},
            **{f"V_{i+1}": v for i, v in enumerate(tbl["V"])}}


ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
b1 = pl.DataFrame(
    [ladder_row(f"B1 {name}", t5s.dam_model(state, b_line=a * B_LINE), alpha=a)
     for name, state in (("closed", CLOSED), ("open", OPEN)) for a in ALPHAS]
)
print(b1)

# %%
from ftr_align.network import intersection

b2_model = intersection(t5s.dam_model(CLOSED, B_LINE), t5s.dam_model(OPEN, B_LINE))
b2 = pl.DataFrame([ladder_row("B2 stack", b2_model)])
print(b2)

# %% [markdown]
# ## Step 6: S1 -- design `b` at a fixed state
#
# Line limits symmetric (one rating per line) and FENCED at the physical rating;
# the switch cap one-directional and unfenced.  Unfenced, S1 reached (0, 0) at both
# states by selling ABOVE rating on ND and below on NH/WN -- unphysical, too risky.
# TODO S5: relax to asymmetric line limits (memo's general `(b+, b-)`).
#
# Vertices of `Lambda(v_t)` on the FTR rows, one set per scenario; then one LP per
# vertex pair (memo eq. design-b-vertex).  At `tau = 0` first.

# %%
FENCE = t5s.rating_fence(B_LINE)  # no FTR line limit above its physical rating (no-regrets rule)


def s1(label, ftr_template, tau, b_max=FENCE, scenarios=None):
    scenarios = SCENARIOS if scenarios is None else scenarios
    d = t5s.design_limits(ftr_template, scenarios, tau, b_max=b_max)
    ftr = t5s.with_designed_limits(ftr_template, d.b)
    row = ladder_row(label, ftr, scenarios, tau=tau, n_lp=d.n_lp, vertices=str(d.vertices))
    return row, d, ftr


for name, y in (("closed", [np.inf]), ("open", [0.0])):
    template = t5s.model(y, b_line=B_LINE, b_switch=100.0)  # b values are placeholders; design replaces them
    for sc in SCENARIOS:
        print(f"S1 {name} / {sc.label}: {len(t5s.dual_vertices(template, sc.direction))} dual vertices")

# %%
rows, designs = [], {}
for name, y in (("closed", [np.inf]), ("open", [0.0])):
    template = t5s.model(y, b_line=B_LINE, b_switch=100.0)
    row, d, ftr = s1(f"S1 {name}", template, tau=0.0)
    rows.append(row)
    designs[name] = (d, ftr)
    lab = template.labels()
    print(f"S1 {name}: designed limits (upper half)")
    print(pl.DataFrame({"element": list(t5s.ELEMENT_NAMES), "b_upper": d.b[: len(t5s.ELEMENT_NAMES)],
                        "b_lower": d.b[len(t5s.ELEMENT_NAMES):], "rating": list(B_LINE) + [np.inf]}))
print(pl.DataFrame(rows))

# %% [markdown]
# ## Step 7: S1 closed's tau frontier, and S3 -- the grid over the switch susceptance
#
# S3: at each `y_H` in (0, inf) the switch is a line with reactance `1/y_H`; design
# `b` (fenced S1) and record E[V] at tau = 0.  The ends are S1 open and S1 closed.

# %%
TAUS = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]
frontier = pl.DataFrame(
    [s1("S1 closed", t5s.model([np.inf], b_line=B_LINE, b_switch=100.0), tau=t)[0] for t in TAUS]
)
print(frontier.select("design", "tau", "E[U]", "E[V]", "U_1", "U_2", "V_1", "V_2", "vertices"))

# %%
Y_GRID = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0, np.inf]
grid_rows = []
for yh in Y_GRID:
    template = t5s.model([yh], b_line=B_LINE, b_switch=100.0)
    row, d, ftr = s1(f"S3 y_H={yh:g}", template, tau=0.0)
    row["y_H"] = yh
    row["b_switch"] = f"{d.b[t5s.n_lines]:.1f} / {d.b[t5s.n_lines + t5s.n_lines + t5s.n_switches]:.1f}"
    grid_rows.append(row)
s3 = pl.DataFrame(grid_rows)
print(s3.select("y_H", "E[U]", "E[V]", "U_1", "U_2", "V_1", "V_2", "b_switch", "n_lp", "vertices"))

# %% [markdown]
# ## Step 8: four hours, two-line patterns pinned, 10% margin
#
# Single-line bindings under a loose rating vector made every rung reach (0, 0)
# (finding 3).  Here every hour binds TWO lines, the wind evening (closed) and the
# south midday (open) share the SAME pair at equal flow -- `match_flows` adjusts
# the midday injection by the least change that reproduces the evening's DH and
# WD flows under the open state (and keeps its ND just under the north hour's) -- and the margin on unbound lines is 10%.

# %%
Q3 = np.array([ 40.,  30., 120.,  -80., -65., -45.])   # eve gas, closed
Q4 = np.array([100.,  90.,  20.,  -60., -30., -120.])  # mid south, open (story)
F1 = t5s._state_ptdf(CLOSED) @ Q1
Q4M = t5s.match_flows(OPEN, Q4, {"DH": F1[t5s.LINE_NAMES.index("DH")], "WD": F1[t5s.LINE_NAMES.index("WD")], "ND": 24.0})
INTERVALS4 = {
    "t1 eve wind":   (CLOSED, Q1,  ["DH", "WD"]),
    "t3 eve gas":    (CLOSED, Q3,  ["SD", "SH"]),
    "t2 mid north":  (OPEN,   Q2,  ["NH", "ND"]),
    "t4 mid south*": (OPEN,   Q4M, ["DH", "WD"]),   # * matched to t1's DH/WD flows
}
print(pl.DataFrame({"node": list(NODE_NAMES), **{k: v[1] for k, v in INTERVALS4.items()}}))
B4, SC4, cross4 = t5s.scenarios_from_injections(INTERVALS4, margin=0.10)
print(pl.DataFrame({"line": list(t5s.LINE_NAMES), "rating": B4}))
print("injection infeasible under state of:", {k: [j for j, ok in v.items() if not ok] for k, v in cross4.items()})

# %%
FENCE4 = t5s.rating_fence(B4)
rows4 = [ladder_row("B2 stack", intersection(t5s.dam_model(CLOSED, B4), t5s.dam_model(OPEN, B4)), SC4)]
designs4 = {}
for name, y in (("closed", [np.inf]), ("open", [0.0])):
    row, d, ftr = s1(f"S1 {name}", t5s.model(y, b_line=B4, b_switch=100.0), tau=0.0, b_max=FENCE4, scenarios=SC4)
    rows4.append(row)
    designs4[name] = d
    print(f"S1 {name} limits:", dict(zip(t5s.ELEMENT_NAMES, np.round(d.b[: len(t5s.ELEMENT_NAMES)], 1))),
          " switch lower:", round(d.b[-1], 1))
pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(rows4).select("design", "E[U]", "E[V]", "U_1", "U_2", "U_3", "U_4", "V_1", "V_2", "V_3", "V_4", "n_lp", "vertices"))

# %% [markdown]
# ## Step 9: S3 on the four hours (MILP)
#
# Neither corner is ideal here (closed 4.2, open 8.1).  The vertex choice per hour
# is made by binaries (`design_limits_milp`, validated against enumeration: 4.219
# and 8.124 reproduced), so each grid point is seconds, not minutes.

# %%
Y_GRID4 = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 2.0, 5.0, 20.0, np.inf]  # 0.05 and 1.0 hit the time limit
g4 = []
for yh in Y_GRID4:
    tpl = t5s.model([yh], b_line=B4, b_switch=100.0)
    # switch fenced at 100 (its flow never exceeds ~32): a 1000 cap makes big-M huge and the MILP crawl
    d = t5s.design_limits_milp(tpl, SC4, 0.0, b_max=t5s.rating_fence(B4, b_switch=100.0), time_limit=60)
    row = ladder_row(f"S3 y_H={yh:g}", t5s.with_designed_limits(tpl, d.b), SC4, y_H=yh, vertices=str(d.vertices))
    row["b_switch"] = f"{d.b[t5s.n_lines]:.1f} / {d.b[2 * t5s.n_lines + t5s.n_switches]:.1f}"
    g4.append(row)
s3_4 = pl.DataFrame(g4)
print(s3_4.select("y_H", "E[U]", "E[V]", "V_1", "V_2", "V_3", "V_4", "b_switch", "vertices"))

# %% [markdown]
# ## Step 10: the undesigned FTR model at every susceptance
#
# Every limit at its rating, switch unlimited, no design.  Per hour U and V across
# `y_H`.  The closed hours' U is the same at every finite `y_H` (an unlimited switch
# reproduces every open-network flow pattern); the north midday's V is the
# topology-bounded V; the open hours' U is finding 1 with a finite switch.  With a
# switch row present (y_H > 0) the design can cap the switch instead of lowering
# line limits -- that cap is why S3 beats both states (finding 7).

# %%
rows10 = []
for yh in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, np.inf]:
    ftr = t5s.model([yh], b_line=B4, b_switch=np.inf if yh == np.inf else 1e4)
    tbl = t5s.evaluate(ftr, SC4)
    r = {"y_H": yh}
    for sc, u, v in zip(SC4, tbl["U"], tbl["V"]):
        r[f"U {sc.label}"] = u
        r[f"V {sc.label}"] = v
    rows10.append(r)
at_ratings = pl.DataFrame(rows10)
print(at_ratings.select(["y_H"] + [c for c in at_ratings.columns if c.startswith("U ")]))
print(at_ratings.select(["y_H"] + [c for c in at_ratings.columns if c.startswith("V ")]))

# %% [markdown]
# ## Step 11: a capped closed switch is B2 (finding 5)
#
# FTR model with the switch closed, every line at its rating, only the switch
# capped.  Two-sided cap of (31.7, 0) reproduces B2's U and V in every hour; a
# tighter cap raises V without changing U; a zero cap is not the open switch.

# %%
def closed_with_cap(lo, hi):
    m = t5s.model([np.inf], b_line=B4, b_switch=np.inf)
    b = m.b.copy()
    r = t5s.n_lines + t5s.n_switches
    b[t5s.n_lines] = hi          # H1 -> H2
    b[r + t5s.n_lines] = lo      # H2 -> H1
    return t5s.with_designed_limits(m, b)


for lo, hi in [(np.inf, np.inf), (31.7, np.inf), (31.7, 0.0), (20.0, 0.0), (0.0, 0.0)]:
    tbl = t5s.evaluate(closed_with_cap(lo, hi), SC4)
    print(f"cap H2->H1 <= {lo:>5}, H1->H2 <= {hi:>5}:  U = {np.round(tbl['U'].to_numpy(), 2)}   "
          f"V = {np.round(tbl['V'].to_numpy(), 2)}   E[V] = {t5s.expected(tbl)[1]:.2f}")
