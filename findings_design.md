# Findings: ex-ante FTR design with a station switch (texas5)

**Guiding question.** What is the relative value of each design lever: the limits `b`, the switch state `s`, a relaxed susceptance `y`?
Answered by showing what each lever can and cannot fix.
Code: `notebooks/explore_texas5switch.py`, `ftr_align/cases/texas5switch.py`.
The queue of next steps is at the bottom.

**Words.**
`t` indexes **hours** of one contract.
Hour `t` has a **switch state** (its DAM runs with the station switch open or closed), an **injection** `q_t` (what its DAM cleared), and a **direction** `v_t` (its prices).
`U_t = h_FTR − h_∩` and `V_t = h_DAM − h_∩`.
From the memo: `V_t = 0` iff some injection optimal for hour `t`'s DAM is feasible in the FTR model, and `U_t = 0` iff some injection optimal for the FTR model along `v_t` is feasible in hour `t`'s DAM.
**Fenced** means every FTR line limit is at most the line's physical rating.

**Two kinds of `V`.**
**Topology-bounded `V`** is the smallest `V_t` the FTR topology allows: the `V_t` with every limit at its rating.
No choice of `b` within the fence goes below it.
**Exposure-bounded `V`** is whatever a design has on top of that.
It appears only because some limits were set below their ratings to hold `U = 0`, and those lower limits exclude the injection the hour needs.

**Setup.**
texas5 with station H split into H1 (the NH side) and H2 (the SH/DH side) joined by a switch; WD1/WD2 merged.
One physical rating vector serves all hours: a line that binds in hour `t` gets its hour-`t` flow as its rating, every other line gets its largest flow over the hours plus a margin.
`v_t = Kᵀ 1_t`, where `1_t` has a 1 at each binding row (upper or lower side by flow direction) and 0 elsewhere.
In words: `v_t` is the sum of the binding lines' shift-factor rows under hour `t`'s switch state, with the flow's sign.
That makes `v_t` constant on every bus hour `t`'s DAM sees.
Equal weights over hours.

The hours used below:

| hour | switch | injection (W, N, S, D, H1, H2) | binding lines |
|---|---|---|---|
| wind evening | closed | 120, 40, 40, −80, −60, −60 | DH, SH (two-hour instance); DH, WD (four-hour instance) |
| north midday | open | 90, 110, 10, −120, −80, −10 | NH, ND |
| gas evening | closed | 40, 30, 120, −80, −65, −45 | SD, SH |
| south midday | open | 115, 73, 24, −85, −47, −80 (fitted so DH and WD carry the wind evening's flows and ND stays under the north midday's) | DH, WD |

The two-hour instance is the first two rows with a 25% margin.
The four-hour instance is all four with a 10% margin.

**Experiment setup.**

| rung | FTR model `Q(y, b)` | what is chosen |
|---|---|---|
| B1 | one switch state, every line limit `× α` | `α` |
| B2 | rows of every hour's switch state, stacked ("model every planned outage") | nothing |
| S1 | one switch state, limits `b` designed | `b` |
| S2 | best S1 over switch states | state and `b` |
| S3 | switch given a continuous susceptance `y_H`, S1 at each grid point | `y_H` and `b` |
| S4 | every element's susceptance free | `y` and `b` (not reached) |

---

## 1. A closed switch with no limit is infinite exposure the moment prices split

The closed FTR polytope contains the direction "move injection from H2 to H1".
No line flow changes along it, only the switch flow, and with `b_H = ∞` nothing stops it.
The intersection with an open-switch hour is bounded, since the open network can only move power from H1 to H2 over lines.
So `h_FTR(v_t) = ∞` while `h_∩` is finite, and `U_t = ∞` as soon as `v_H1 ≠ v_H2`.
A derate `α` scales line limits and never touches this direction.
In money: the SFT accepts an unbounded FTR from H2 to H1 worth `v_H1 − v_H2` per MW.
Two repairs, both design choices: open the switch in the FTR model, or give it a finite one-way cap.
(B1 closed: `U = ∞` in every open hour at every `α`.)

## 2. Misalignment requires an hour whose injection is infeasible in another hour's network

If every hour's injection is feasible in every hour's network, then B2 contains every hour's injection, and `V = 0` and `U = 0` in every hour.
So the most any lever can earn is set by how much the hours disagree.
Disagreement is easy to miss.
An hour with one line binding has a whole facet of optimal injections, and some point of that facet is almost always feasible in the other network.
On this network B2 has `V = 0` along every facet normal of both DAM polytopes, 32 of 32.
`V_t > 0` needed two lines binding together and a lopsided station split.

## 3. Against every direction at once, B2 is the best possible FTR model

Suppose the realized directions include every facet normal of every hour's DAM polytope.
`U_t = 0` along a facet normal keeps the FTR polytope inside that facet's halfspace.
Along all of them it stays inside the whole polytope.
Inside every hour's polytope means inside B2, and a smaller set inside the same DAM has larger `V`.
So at `τ = 0` nothing beats B2.
Refusing exposure in every direction is today's practice, derived.
Design gains exist only because realized directions are few and multi-line.

## 4. Each switch state of the FTR model has a `V` the other does not, and `b` removes neither

Two FTR models: the FTR model with the switch **open**, and the FTR model with the switch **closed**.
Two kinds of hour: an hour whose DAM ran with the switch **closed** (the wind evening, the gas evening), and an hour whose DAM ran with the switch **open** (the north midday, the south midday).

**The FTR model with the switch open has no topology-bounded `V` in any hour.**
Take an hour whose DAM ran with the switch closed.
That DAM priced H1 and H2 equally, so moving withdrawal from H2 to H1 changes neither the value `v_t·q` nor any line flow, only the switch flow.
Move it until the switch flow is zero.
The result is a flow pattern with no flow on the switch, so it is also a flow pattern of the open network, and the hour's injection (so moved) is feasible in the FTR model with the switch open.
An hour whose DAM ran with the switch open is feasible in that FTR model trivially.

**The FTR model with the switch open does have exposure-bounded `V`, and it comes from the closed hours.**
In the wind evening, with every FTR limit at its rating, the FTR model with the switch open has `U = 22.4`: along that hour's direction it can sell more than the closed DAM collected, because the open network routes the same injections differently and reaches further along `v_t` before any rating binds.
To bring `U` to zero the design must lower some line limits below their ratings (NH to 58.5, DH to 40.3, WS to 12.7, WN to 2.7).
Those lowered limits apply in every hour.
When the injection of an open hour no longer fits under them, that hour gets `V`.
That `V` exists only because of the lowered limits, so it is exposure-bounded.

**The FTR model with the switch closed has topology-bounded `V`, and it comes from the open hours.**
Take the north midday, whose DAM ran with the switch open and priced H1 below H2.
Its injection puts 80 MW of load at H1 and 10 at H2.
On the closed network that split sends 29 MW through the switch, the loop redistributes it, and ND's flow exceeds its rating.
No limit at or below rating admits that injection, so the FTR model with the switch closed has `V = 16.9` of 104.6 in that hour whatever `b` is.
In the closed hours this FTR model has `U = 0` at ratings (it is the DAM's own network), so it lowers no limits and has no exposure-bounded `V`.
This is the `V` that the "model every planned outage" rule `s_j = min_t s_{j,t}` avoids, and the `U` it does not.

**Each hour whose DAM had the switch closed can only add `V` to the FTR model with the switch open.
Each hour whose DAM had the switch open can only add `V` to the FTR model with the switch closed.
`b` removes neither.**
Fenced, `τ = 0`:

| instance | B2 | S1, FTR switch closed | S1, FTR switch open |
|---|---|---|---|
| two hours (wind evening, north midday) | 8.4 | 8.4 | **0** |
| four hours | 4.2 | 4.2 | 8.1 |

In the two-hour instance the limits lowered for the wind evening exclude nothing the north midday needs, so the FTR model with the switch open has `V = 0`.
In the four-hour instance the limits lowered for the two closed hours exclude the injections of the gas evening (1.3), the north midday (21.5) and the south midday (9.7), so it has `E[V] = 8.1`.
The FTR model with the switch closed equals B2 in both instances, since its only `V` is the north midday's 16.9.
**Open question: is the open FTR model's `V` monotone in the number of closed hours as a theorem, or only typically?**

## 5. A finite cap on the closed switch buys B2's exposure guarantee, and nothing more

Take the FTR model with the switch closed, every line at its rating, and the switch uncapped.
In each hour whose DAM ran with the switch open, `U = ∞` (finding 1).
Now cap the switch flow in both directions.
Four-hour instance:

| switch cap (H2→H1, H1→H2) | U per hour (wind, gas, north, south) | V per hour | E[V] |
|---|---|---|---|
| none | 0, 0, ∞, ∞ | 0, 0, 16.9, 0 | 4.2 |
| 31.7, none | 0, 0, 0, ∞ | 0, 0, 16.9, 0 | 4.2 |
| 31.7, 0 | 0, 0, 0, 0 | 0, 0, 16.9, 0 | 4.2 |
| 20, 0 | 0, 0, 0, 0 | 0, 0, 22.4, 0 | 5.6 |
| 0, 0 | 0, 0, 0, 0 | 0, 0, 31.7, 0 | 7.9 |
| B2 | 0, 0, 0, 0 | 0, 0, 16.9, 0 | 4.2 |

With the cap at 31.7 and 0, no line limit lowered, the FTR model with the switch closed has exactly B2's `U` and `V` in every hour.
The reason is in the memo's dual: for a closed switch the whole value `|v_H1 − v_H2|` of moving injection across it lands on the switch's own limit price, so the switch cap is the exact lever for exposure in open hours, and the line limits never need to move.
Each direction needs its own cap, because the two open hours price H1 against H2 with opposite signs.
The cap should be as loose as exposure allows.
Tightening it below 31.7 leaves `U` at zero and raises `V`, and a cap of zero is not the open switch: the two nodes still share an angle, and `V` is 31.7 against 0 for the FTR model with the switch open.
So a capped closed switch and B2 are the same design on these hours, and what neither can remove is the topology-bounded `V` of finding 4.

## 6. Fenced limits make exposure controllable; the budget buys back only exposure-bounded `V`

Unfenced, in the two-hour instance, S1 reached `(0, 0)` at both switch states by setting ND's limit at 35.6 against a rating of 24.6 and lowering other limits to compensate.
Legal in total, too risky per line, so limits are fenced from here on.
The fence is the ISO's rule, not a theorem, and it leaves the problem structure unchanged.

**6a.** Fenced, `U` can always be driven to zero by lowering `b`, so `E[U] = 0` is reachable on every rung and the design problem is about `E[V]`.
This assumes the hour set is known.
Realized underfunding in practice is then a forecast of the wrong hours, a different problem.

**6b.** A budget `τ > 0` does not necessarily lower `V`.
It allows limits above where `U = 0` had held them, so it can remove exposure-bounded `V` and nothing else.
Topology-bounded `V` is the same at every limit within the fence.
S1 closed sits at `E[V] = 8.4` for every `τ` from 0 to 8.
A `τ` frontier exists only for a model whose `V` is exposure-bounded.

## 7. The switch enters the flows through `y_H / (y_H + y_th)`, and `V` follows a different curve because it is an optimum

Let `y_th` be the Thevenin susceptance of the rest of the network between H1 and H2.
For a fixed injection, every line flow at switch susceptance `y_H` is

    f(y_H) = f_open + (f_closed − f_open) · y_H / (y_H + y_th).

This is the LODF formula.
In the angle formulation `y_H` is just one entry of the susceptance matrix; the ratio is the same fact seen from the flows.

`V_t(y_H)` is the value of an LP, and it is piecewise.
Each piece is one active set of the intersection LP.
Within a piece `V_t` is a ratio of two affine functions of `y_H`, the same shape as the formula above but with its own constants.
The constants differ because the formula holds the injection fixed, while the LP moves the injection to the point where the limiting line is exactly at its rating under the relaxed flows, and that point moves with `y_H`.
If the LP kept the DAM's injection, the curve would be the Thevenin one.

Two-hour instance, `τ = 0`, `U = 0` throughout.
One active set covered the whole grid, and the north midday's `V` fit `16.9 · y_H / (y_H + 0.407)` against `y_th = 0.266`:

| y_H | 0 | 0.05 | 0.2 | 0.5 | 1 | 2 | 5 | 20 | ∞ |
|---|---|---|---|---|---|---|---|---|---|
| V, north midday | 0 | 1.8 | 5.6 | 9.3 | 12.0 | 14.0 | 15.6 | 16.5 | 16.9 |

The curve only rises here because in this instance the FTR model with the switch open has `V = 0`, so nothing falls as `y_H` grows.

## 8. A switch susceptance strictly between open and closed can give the FTR model lower `E[V]` than either state

S3 varies one thing: the susceptance `y_H` of the switch **in the FTR model**, from 0 (open) to ∞ (closed), and designs `b` at each value.
The hours and their DAMs do not change.

First, the FTR model with every limit at its rating and the switch unlimited, before any design.
Four-hour instance:

| y_H | 0 (open) | 0.1 | 0.2 | 0.5 | 1 | 2 | 5 | 20 | ∞ (closed) |
|---|---|---|---|---|---|---|---|---|---|
| U, wind evening (closed) | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 22.4 | 0 |
| U, gas evening (closed) | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 | 0 |
| U, north midday (open) | 0 | 10.2 | 19.6 | 45.5 | 85.4 | 205 | 567 | 2370 | ∞ |
| U, south midday (open) | 0 | 24.4 | 48.7 | 122 | 244 | 487 | 1218 | 4871 | ∞ |
| V, north midday (open) | 0 | 2.8 | 4.8 | 8.4 | 11.2 | 13.5 | 15.3 | 16.5 | 16.9 |
| V, every other hour | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Three things to read off.
The north midday's `V` row is the topology-bounded `V`: it is the only one, and it rises with `y_H` along the curve of finding 7.
The closed hours' `U` is the same 22.4 and 6.9 at every finite `y_H`: an unlimited switch lets the FTR model reproduce every open-network flow pattern, so coupling H1 and H2 weakly or strongly changes nothing along a direction that prices them equally.
The open hours' `U` grows without bound: this is finding 1 with a finite switch.

Now the design.
At `y_H = 0` the switch has no row, so the only way to remove the closed hours' `U` of 22.4 and 6.9 is to lower line limits, and the lowered limits exclude the midday injections.
At any `y_H > 0` the switch has a row, and the design caps the switch flow instead: a one-way cap of 6.3 at `y_H = 0.1`, rising to 31.7 at `∞`.
The cap removes the closed hours' `U` and the open hours' `U` at once, and leaves the line limits near their ratings, so the midday injections stay feasible.
So the exposure-bounded `V` is 7.9 at `y_H = 0` (all of the 8.1) and at most the MIP tolerance at every `y_H > 0`.
The topology-bounded `V` is 0 at `y_H = 0` and rises to 16.9.
The sum is smallest where the topology-bounded part is still small and the switch cap is already available: a small positive `y_H`.

Designed, fenced, `τ = 0`, `U = 0` at every point; MILP at a 0.5% gap, so each `E[V]` carries about ±0.6, and two grid points timed out:

| y_H | 0 (open) | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 2 | 5 | 20 | ∞ (closed) |
|---|---|---|---|---|---|---|---|---|---|---|
| E[V] | 8.1 | **1.2** | 1.7 | 2.2 | 2.5 | 2.9 | 3.6 | 3.9 | 4.7 | 4.7 (4.2 exact) |
| V, north midday | 21.5 | 3.3 | 5.6 | 7.8 | 9.3 | 10.9 | 14.0 | 15.6 | 16.5 | 16.9 |
| V, gas evening | 1.3 | 1.5 | 1.2 | 1.0 | 0.8 | 0.7 | 0.3 | 0.1 | ~0 | ~0 |
| V, south midday | 9.7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| switch cap | none | 6.3 | 10.5 | 14.7 | 17.5 | 20.6 | 26.4 | 29.3 | 31.0 | 31.7 |

The north midday's 21.5 at `y_H = 0` is exposure-bounded (its topology-bounded value there is 0); its 3.3 at `y_H = 0.1` is topology-bounded plus tolerance (2.8 at ratings).
That is the jump between the first two columns: a different cause, not a discontinuity of one curve.
None of the curves is linear.
Each is a ratio of two affine functions of `y_H` per active set (finding 7).
The relaxed susceptance matters because it gives the switch a row and so a cap.
The memo's point that switch limits belong in the design is what the interior optimum is made of.

---

## Next (the queue)

- **Fairness.** Per-constraint and per-block shares of `V` at a design (existing `constraint_table` / `block_table`); max or CVaR across hours. At a vertex direction many rows bind, so this is also where dual multiplicity meets the design.
- **Facet directions with a budget.** With every facet normal realized and `τ > 0`, the budget buys "which facets of the other state to violate": a frontier that is purely exposure-bounded, the complement of 6b.
- **Vertex directions.** Dominance along exposing directions does not force containment, so S1 and S3 have room above B2 there. The MILP will be large (80 hours × 16–30 vertices); use the memo's iteration heuristic.
- **Other hour mixes.** Does the S3 optimum move; is the `V` of the FTR model with the switch open monotone in the number of closed hours (finding 4's open question). Fix the two timed-out grid points.
- Line-outage hours. SD as a third binder in the north midday. Asymmetric line limits (memo's `(b⁺, b⁻)`). Settlement-point weights.
