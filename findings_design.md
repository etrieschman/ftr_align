# Findings: ex-ante FTR design with a station switch (texas5)

**Question.** How much does each design lever buy: the limits `b`, the switch state `s`, a relaxed switch susceptance `y_H`?
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

## 3. Against every direction, B2 is the best FTR model

If the realized directions include every facet normal of every hour's DAM polytope, `U = 0` forces the FTR model inside every DAM polytope, hence inside B2.
A smaller set inside the same DAM has larger `V`.
So at `τ = 0` nothing beats B2.
Today's practice is the design that refuses exposure in every direction.
Design can gain only because realized directions are few.

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

## 5. A two-sided cap on the closed switch equals B2

FTR switch closed, every line at rating, switch capped at 31.7 (H2→H1) and 0 (H1→H2): `U` and `V` equal B2's in every hour.
The memo's dual explains it: for a closed switch, the whole value of moving injection across it lands on the switch's own limit price, so the cap is the exact lever for exposure.
Each direction needs its own cap because the two open-switch hours price H1 against H2 with opposite signs.
A tighter cap raises `V` without changing `U` (cap 20: `V = 22.4`).
A zero cap is not the open switch (`V = 31.7` against 0).

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

## 8. A switch susceptance between open and closed can beat both states

Four-hour case.
With every limit at rating and the switch uncapped, the closed-switch hours' `U` is 22.4 and 6.9 at every finite `y_H`, because an unlimited switch reproduces every open-network flow pattern.
At `y_H = 0` the switch has no row, so only lowered line limits can remove that `U`, and they exclude the midday injections (exposure-bounded `V = 7.9`).
At any `y_H > 0` the switch has a row, the design caps it instead (6.3 at `y_H = 0.1`, 31.7 at ∞), the line limits stay near rating, and the exposure-bounded `V` vanishes.
The topology-bounded `V` is the north midday's, 0 at `y_H = 0` and rising to 16.9.
The sum is smallest at a small positive `y_H`.

| y_H | 0 | 0.1 | 0.2 | 0.5 | 2 | 5 | ∞ |
|---|---|---|---|---|---|---|---|
| E[V], designed, fenced, τ = 0 | 8.1 | 1.2 | 1.7 | 2.5 | 3.6 | 3.9 | 4.2 |

MILP at a 0.5% gap, so each value carries about ±0.6.
The north midday's 21.5 at `y_H = 0` is exposure-bounded; its 3.3 at `y_H = 0.1` is topology-bounded. That is the jump.

---

## Next

- Fairness: per-constraint and per-block shares of `V` at a design (`constraint_table`, `block_table`); max or CVaR across hours.
- Facet directions with `τ > 0`: the budget chooses which facets of the other state to violate.
- Vertex directions: S1 and S3 have room above B2 there; use the memo's iteration heuristic, the MILP is too large.
- Other hour mixes: does the S3 optimum move; finding 4's open question. Fix the two timed-out grid points.
- Line-outage hours. Asymmetric line limits. Settlement-point weights.
