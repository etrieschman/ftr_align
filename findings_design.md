# Findings: ex-ante FTR design with a station switch (texas5)

**Question.** How much does each design choice buy: the limits `b`, the switch state `s`, a relaxed switch susceptance `y_H`?
Code: `notebooks/explore_texas5switch.py`, `ftr_align/cases/texas5switch.py`.
Queue at the bottom.

**Definitions.**
An **hour** `t` has a switch state (its DAM ran with the station switch open or closed), an injection `q_t`, and a direction `v_t` (its prices).
`U_t = h_FTR − h_∩`. `V_t = h_DAM − h_∩`.
`V_t = 0` iff an injection that is optimal for hour `t`'s DAM is feasible in the FTR model.
`U_t = 0` iff an injection that is optimal for the FTR model along `v_t` is feasible in hour `t`'s DAM.
**Fenced**: every FTR line limit is at most the line's rating.
**Topology-bounded `V`**: the `V_t` of the FTR model with every limit at its rating. No `b` within the fence goes lower.
**Exposure-bounded `V`**: any `V_t` above that. It exists because limits were lowered to hold `U = 0`.

**Setup.**
texas5 with station H split into H1 (NH side) and H2 (SH and DH side), joined by one switch.
One rating vector for all hours: a binding line's rating is its flow in its hour; every other line gets its largest flow plus a margin (25% in the two-hour case, 10% in the four-hour case).
Equally weighted hours.

| hour | DAM switch | injection (W, N, S, D, H1, H2) | binding lines |
|---|---|---|---|
| wind evening | closed | 120, 40, 40, −80, −60, −60 | DH, SH (two-hour case); DH, WD (four-hour case) |
| north midday | open | 90, 110, 10, −120, −80, −10 | NH, ND |
| gas evening | closed | 40, 30, 120, −80, −65, −45 | SD, SH |
| south midday | open | 115, 73, 24, −85, −47, −80 | DH, WD |

| design | FTR model | chosen |
|---|---|---|
| B1 | one switch state, every limit `× α` | `α` |
| B2 | rows of every hour's switch state, stacked | nothing |
| S1 | one switch state, limits designed | `b` |
| S2 | best S1 over switch states | `s`, `b` |
| S3 | switch at susceptance `y_H`, S1 at each `y_H` | `y_H`, `b` |

---

## 1. A closed switch with no limit means infinite exposure whenever H1 and H2 price differently

The closed FTR model can move injection from H2 to H1 without changing any line flow.
With no switch limit that move is unbounded, and it is worth `v_H1 − v_H2` per MW.
So `U_t = ∞` in every hour where `v_H1 != v_H2`. Opening the switch in the FTR model or capping its flow controls this.

## 2. Misalignment requires an hour whose injection is infeasible in another hour's network

If every hour's injection is feasible in every hour's network, B2 contains them all and `U = V = 0`.
This finding holds when no derate is used; a derate below rating produces `V`.

The condition is weaker than it looks, because `V_t = 0` needs only one of the hour's optimal injections to be feasible in the FTR model, not `q_t` itself.
The more optimal injections the hour's DAM has, the more chances the FTR model has to contain one.
With one binding line, every feasible injection that makes that line bind is optimal.
With two binding lines, an optimal injection must bind both, which is a subset of the one-line set.
With enough binding lines, the only optimal injection is the DAM `q_t` itself.
So `V_t > 0` at a direction binding line `j` implies `V_t > 0` at every direction binding `j` and more.

Takeaway: `V` most likely comes from hours with several binding constraints.
An hour with one binding line leaves the DAM indifferent among many injections.


## 3. If every constraint of every state is a realized direction, no FTR model has lower `V` than B2

`U_t = 0` means `h_FTR(v_t) = h_∩(v_t)`, and `h_∩(v_t) <= h_DAM(v_t)` because the intersection sits inside the DAM polytope.
So `U_t = 0` implies `h_FTR(v_t) <= h_DAM(v_t)`; that is the FTR model inside the halfspace `v_t^T q <= h_DAM(v_t)`.

Suppose hour `t` binds line `j` alone, so `v_t = k_j`.
Then `h_DAM(v_t)` is the largest flow line `j` can carry in that state, which is `b_j` when `j` can reach its rating there: the halfspace is line `j`'s own constraint in that state, `k_j^T q <= b_j`.
One such hour per constraint of a state forces the FTR model inside that state's DAM polytope.
One such hour per constraint of every state forces it inside B2.

A smaller FTR model has a smaller intersection with the DAM, so a lower `h_∩` and a higher `V`.
So at `τ = 0`, with every constraint of every state realized, no FTR model has lower `V` than B2.
With fewer realized directions the containment is weaker: the FTR model must lie inside one halfspace per hour, not inside B2, and a lower `V` may be achieved.




## 4. Each switch state has a `V` that `b` cannot remove

With one topology, the FTR model at ratings is the DAM in every hour and `U = V = 0`.
With two, hours disagree, in two directions.
A DAM injection that does not fit the FTR network is topology-bounded `V`: no fenced `b` admits it.
An FTR portfolio that does not fit the DAM is `U`: fenced, it is removed only by lowering a line, and every hour whose optimal injection needs that line at rating loses `V`.
So exposure-bounded `V` in one hour is always caused by `U` in another, through a line both need.

**FTR switch closed.**
An open-hour injection can put load at H1 that the open network served through NH alone; closed, part of it crosses the switch and reroutes through the loop.
The north midday's 80 MW at H1 sends 29 MW across the switch and pushes ND past its rating, so `V = 16.9` whatever `b` is: topology-bounded.
In closed hours this FTR model is the DAM's own network and loses nothing.

**FTR switch open.**
A closed-hour DAM priced H1 and H2 equally, so its injection can be rebalanced between them at no loss until the switch flow is zero, and that flow pattern is an open-network pattern.
So this FTR model has no topology-bounded `V` in any hour, for any number of switches.
It has `U` in closed hours: it sells H1 and H2 loads independently, and the closed network couples them.
In the wind evening it sells NH, DH and WD at rating; closed, 23.6 MW of H1's load crosses the switch and DH and WD exceed rating, `U = 22.4`.
Removing that lowers NH and DH, the midday hours' binding lines, so the midday injections fall out: `V = 8.1`, all exposure-bounded.

In the table below we show numerically that no single state is the right decision: we create cases where `E[V]` is lower under closed than open and vice versa

| case | B2 | S1 closed | S1 open |
|---|---|---|---|
| two hours | 8.4 | 8.4 | 0 |
| four hours | 4.2 | 4.2 | 8.1 |



## 5. Fenced, `V` has two sources, and the budget reaches only one

Limits are fenced, `b ≤ b_t` on every line: an unfenced design sells above rating, which is exposure in any hour outside the sample (the two-hour case did so on ND).
`V_t > 0` means hour `t` has no optimal injection inside the FTR model, and under the fence there are two ways that happens.
The FTR network cannot carry the injection at ratings: no fenced `b` admits it, only a different state or susceptance does. This is topology-bounded `V`.
The FTR network can carry it at ratings, but a limit was lowered because at rating that line sold an injection some other hour cannot carry. This is exposure-bounded `V`, and a larger `τ` lets the limit back up, so `E[V]` cannot rise with `τ`.
Fenced, `U` is always removable (zero limits remove it), so the problem is feasible at `τ = 0` and is about `V`.

Four-hour case.
S1 closed: `E[V] = 4.2` at every `τ` from 0 to 16, all topology-bounded (the north midday), none bought back.
S1 open: `E[V] = 8.1` at `τ = 0`, all exposure-bounded, `E[V] = 0` by `τ = 6`.


## 6. The switch enters the flows through `y_H / (y_H + y_th)`; `V` follows a different curve because it is an optimum

With `y_th` the Thevenin susceptance between H1 and H2 through the lines, every line flow of a fixed injection is `f_open + (f_closed − f_open) · y_H / (y_H + y_th)`.
`V_t(y_H)` is an LP value, so it is piecewise, one piece per active set, and on each piece a ratio of two affine functions of `y_H`.
The constants differ from the flow formula because the LP moves the injection to where the limiting line is exactly at rating, and that point moves with `y_H`.
Two-hour case: one active set over the whole grid, `V = 16.9 · y_H / (y_H + 0.407)` against `y_th = 0.266`.

| y_H | 0 | 0.05 | 0.2 | 0.5 | 1 | 2 | 5 | 20 | ∞ |
|---|---|---|---|---|---|---|---|---|---|
| V, north midday | 0 | 1.8 | 5.6 | 9.3 | 12.0 | 14.0 | 15.6 | 16.5 | 16.9 |

## 7. On the four-hour case, the FTR model with lowest `E[V]` had the switch at an intermediate susceptance

S3 varies `y_H`, the susceptance of the switch in the FTR model, from 0 (open) to ∞ (closed), and designs the limits at each value.
The hours do not change.

Before design, with every limit at its rating and no limit on the switch:
in the two closed-switch hours `U` is 22.4 and 6.9 at every finite `y_H` and 0 at `y_H = ∞`; in the two open-switch hours `U` is positive at every `y_H > 0` and infinite at `y_H = ∞` (finding 1).
The design must bring all of this `U` to zero.

At `y_H = 0` the switch is not an element of the FTR model, so the design can only lower line limits.
The lowered limits exclude the two midday injections, and `E[V] = 8.1`, all of it exposure-bounded.

At every `y_H > 0` the switch is an element with a flow limit, and the design brings `U` to zero mainly by limiting the switch flow (6.3 at `y_H = 0.1`, 31.7 at `y_H = ∞`).
The line limits stay near their ratings and the midday injections remain feasible.
What remains is the north midday's topology-bounded `V`: 2.8 at `y_H = 0.1`, 16.9 at `y_H = ∞`.

| y_H | 0 | 0.1 | 0.2 | 0.5 | 2 | 5 | ∞ |
|---|---|---|---|---|---|---|---|
| E[V], designed, fenced, τ = 0 | 8.1 | 1.2 | 1.7 | 2.5 | 3.6 | 3.9 | 4.2 |

MILP at a 0.5% gap, so each value carries about ±0.6.
The lowest `E[V]` is at `y_H = 0.1`, below both switch states.
Where the lowest value sits depends on the instance; here it was the smallest grid point above zero because the topology-bounded `V` starts at zero and the switch limit was available at any `y_H > 0`.

## 8. A limit on a closed switch controls exposure in open-switch hours

In the closed FTR model, exposure in an open hour is injection moved across the switch and sold at that hour's H1–H2 spread, and a switch limit bounds that move directly, so no line limit needs lowering; each direction needs its own limit because open hours price the spread with either sign.
On the four-hour case the two limits fitted to the two open hours reproduced B2's `U` and `V` in every hour, and a third open hour would have no limit of its own.
In the auction a binding switch limit prices an H1–H2 spread that closed-switch hours of the DAM never pay, and at a zero limit that spread is set by bids alone.

---

## Next

- Fairness: per-constraint and per-block shares of `V` at a design (`constraint_table`, `block_table`); max or CVaR across hours.
- Facet directions with `τ > 0`: the budget chooses which facets of the other state to violate.
- Vertex directions: S1 and S3 have room above B2 there; use the memo's iteration heuristic, the MILP is too large.
- Other hour mixes: does the S3 optimum move; finding 4's open question. Fix the two timed-out grid points.
- Line-outage hours. Asymmetric line limits. Settlement-point weights.
