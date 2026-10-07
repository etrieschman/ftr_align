# Findings: ex-ante FTR design with a station switch (texas5)

**Question.** How much does each design choice buy: the limits `b`, the switch state `s`, a relaxed switch susceptance `y_H`?
Code: `notebooks/explore_texas5switch.py`, `ftr_align/cases/texas5switch.py`.
Queue at the bottom.

**Definitions.**
An **hour** `t` has a switch state (its DAM ran with the station switch open or closed), an injection `q_t`, and a direction `v_t` (its prices).
`v_t` is the sum of the shift-factor rows of the lines that bind in hour `t`, each with the sign of its flow.
`U_t = h_FTR − h_∩`. `V_t = h_DAM − h_∩`.
`V_t = 0` iff an injection that is optimal for hour `t`'s DAM is feasible in the FTR model.
`U_t = 0` iff an injection that is optimal for the FTR model along `v_t` is feasible in hour `t`'s DAM.
**Fenced**: every FTR line limit is at most the line's rating.
**Topology-bounded `V`**: the `V_t` of the FTR model with every limit at its rating. No `b` within the fence goes lower.
**Exposure-bounded `V`**: any `V_t` above that. It exists because limits were lowered to hold `U = 0`.

**Setup.**
texas5 with station H split into H1 (NH side) and H2 (SH and DH side), joined by one switch.
One rating vector for all hours: a binding line's rating is its flow in its hour; every other line gets its largest flow plus a margin (25% in the two-hour case, 10% in the four-hour case).
Equal weights.

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
So `U_t = ∞` in every hour whose DAM priced H1 and H2 differently.
A derate does not help. Opening the switch or capping its flow does.

## 2. Misalignment requires an hour whose injection is infeasible in another hour's network

If every hour's injection is feasible in every hour's network, B2 contains them all and `U = V = 0`.
An hour with one binding line almost never produces `V`: its optimal injections form a whole facet, and some point of it fits the other network.
On this network B2 has `V = 0` along all 32 facet normals of both DAM polytopes.
`V > 0` needed two binding lines and a lopsided H1/H2 split.

## 3. If every facet normal is a realized direction, no FTR model has lower `V` than B2

Suppose the realized directions include every facet normal of every hour's DAM polytope.
`U = 0` along a facet normal keeps the FTR model inside that facet's halfspace, so `U = 0` along all of them keeps it inside every DAM polytope, hence inside B2.
A smaller set inside the same DAM has `V` at least as large.
So at `τ = 0` no FTR model has lower `V` than B2.
The argument uses only polytope facts, so it holds for any network and any hours, but only under its premise: every facet normal realized.
Realized directions are few, which is the room design has.

## 4. Each FTR switch state has a `V` the other does not, and `b` removes neither

**FTR switch open.**
In an hour whose DAM had the switch closed, that DAM priced H1 and H2 equally, so its injection can be rebalanced between H1 and H2 with no change in value or line flows until the switch flow is zero.
A flow pattern with zero switch flow is also an open-network flow pattern.
So this FTR model has no topology-bounded `V` in any hour.
It has exposure-bounded `V`: in the wind evening at ratings `U = 22.4`, the design lowers NH, DH, WS and WN to remove it, and the lowered limits exclude the midday injections.

**FTR switch closed.**
The north midday's injection puts 80 MW of load at H1 and 10 at H2.
On the closed network 29 MW cross the switch, the loop redistributes it, and ND exceeds its rating.
No limit at or below rating admits that injection, so `V = 16.9` of 104.6 whatever `b` is.
In closed-switch hours this FTR model is the DAM's own network, so it lowers nothing.

Each closed-switch hour can add `V` only to the open FTR model. Each open-switch hour can add `V` only to the closed one.

| case | B2 | S1 closed | S1 open |
|---|---|---|---|
| two hours | 8.4 | 8.4 | 0 |
| four hours | 4.2 | 4.2 | 8.1 |

**Open question: is the open FTR model's `V` monotone in the number of closed-switch hours?**

## 5. On this instance, a two-sided limit on the closed switch reproduced B2

FTR switch closed, every line at rating, switch flow limited to 31.7 (H2→H1) and 0 (H1→H2): `U` and `V` equal B2's in every hour.
The memo's dual suggests why: for a closed switch, the value of moving injection across it lands on the switch's own limit price, so the switch limit is where exposure in open-switch hours is controlled.
Each direction needed its own limit because the two open-switch hours price H1 against H2 with opposite signs.
A tighter limit raised `V` without changing `U` (limit 20: `V = 22.4`).
A zero limit is not the open switch (`V = 31.7` against 0).
This is one switch and four hours; whether a switch limit always reproduces B2 is untested.

## 6. Fenced, exposure is controllable, and a budget buys back only exposure-bounded `V`

Unfenced, in the two-hour case, S1 reached `U = V = 0` at both states by setting ND's limit above its rating. Too risky, so limits are fenced.
Fenced, `U` can always be driven to zero by lowering `b`, so the design problem is about `V`.
This assumes the hours are known; realized underfunding is then a forecast error.
A budget `τ > 0` allows limits above where `U = 0` held them, so it removes exposure-bounded `V` only.
S1 closed has `E[V] = 8.4` at every `τ` from 0 to 8.

## 7. The switch enters the flows through `y_H / (y_H + y_th)`; `V` follows a different curve because it is an optimum

With `y_th` the Thevenin susceptance between H1 and H2 through the lines, every line flow of a fixed injection is `f_open + (f_closed − f_open) · y_H / (y_H + y_th)`.
`V_t(y_H)` is an LP value, so it is piecewise, one piece per active set, and on each piece a ratio of two affine functions of `y_H`.
The constants differ from the flow formula because the LP moves the injection to where the limiting line is exactly at rating, and that point moves with `y_H`.
Two-hour case: one active set over the whole grid, `V = 16.9 · y_H / (y_H + 0.407)` against `y_th = 0.266`.

| y_H | 0 | 0.05 | 0.2 | 0.5 | 1 | 2 | 5 | 20 | ∞ |
|---|---|---|---|---|---|---|---|---|---|
| V, north midday | 0 | 1.8 | 5.6 | 9.3 | 12.0 | 14.0 | 15.6 | 16.5 | 16.9 |

## 8. On the four-hour case, the FTR model with lowest `E[V]` had the switch at an intermediate susceptance

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

---

## Next

- Fairness: per-constraint and per-block shares of `V` at a design (`constraint_table`, `block_table`); max or CVaR across hours.
- Facet directions with `τ > 0`: the budget chooses which facets of the other state to violate.
- Vertex directions: S1 and S3 have room above B2 there; use the memo's iteration heuristic, the MILP is too large.
- Other hour mixes: does the S3 optimum move; finding 4's open question. Fix the two timed-out grid points.
- Line-outage hours. Asymmetric line limits. Settlement-point weights.
