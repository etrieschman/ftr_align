# Findings: ex-ante FTR design with a station switch (texas5)

**Guiding question.** What is the bang for the buck of each design lever: the
limits `b`, the switch state `s`, a relaxed susceptance `y`? Answered by showing
what each lever can and cannot fix. Short by design; one entry per finding that
changed how we think. Code: `notebooks/explore_texas5switch.py`,
`ftr_align/cases/texas5switch.py`. Validation of the attribution machinery is
in `findings_validation.md`. The queue of next steps is at the bottom.

**Words.** `t` indexes **hours** of one contract. Hour `t` has a **switch
state** (its DAM runs with the station switch open or closed), a **direction**
`v_t` (its prices), and **clearings**: the injections its DAM could have
cleared at prices `v_t`, a face of the DAM polytope rather than one point.
`U_t = h_FTR − h_∩`, `V_t = h_DAM − h_∩`. From the memo: `V_t = 0` iff some
clearing of hour `t` is feasible in the FTR model; `U_t = 0` iff some
FTR-optimal injection along `v_t` is feasible in hour `t`'s DAM.

**Fenced** means every FTR line limit is at most the line's physical rating.
**Unfenced** means no such cap.

**Where `V` comes from.** Fix the FTR topology and fence the limits. Making a
limit larger never removes a clearing, so the FTR polytope with every limit at
its rating is the largest allowed and has the smallest `V_t` of any `b`. Call
that smallest value the **topology-forced `V`**: it is there at the fence, and
no choice of `b` removes it. Whatever a design has above it is **`U`-induced
`V`**: the design pulled some limits below their ratings, the only reason to do
so is to hold `U = 0` somewhere, and the pulled limits excluded clearings. So
`V_t(design) = V_t(fence) + [V_t(design) − V_t(fence)]`, topology-forced plus
`U`-induced, and these are the only two sources.

**Setup.** texas5 with station H split into H1 (the NH side) and H2 (the
SH/DH side) joined by a switch; WD1/WD2 merged. One physical rating vector for
all hours, set so chosen lines bind in each hour; `v_t` is the sum of those
binding rows, hence constant on every bus that hour's DAM sees. Equal weights.

**The ladder.** Rungs of FTR-model design, each scored by `E[U]`, `E[V]` over
the hours. From finding 4 on, designed limits are fenced.

| rung | FTR model `Q(y, b)` | what is chosen |
|---|---|---|
| B1 | one switch state, every line limit `× α` | `α` |
| B2 | rows of every hour's switch state, stacked ("model every planned outage") | nothing |
| S1 | one switch state, limits `b` designed | `b`, via the dual vertex of `Λ(v_t)` per hour (enumeration, or MILP with one binary per vertex) |
| S2 | S1 at every switch state, best kept | state and `b` |
| S3 | switch given a continuous susceptance `y_H`, S1 at each grid point | `y_H` and `b` |
| S4 | every element's susceptance free | `y` and `b` (not reached) |

---

## 1. A closed switch with no limit is infinite exposure the moment prices split

The closed FTR polytope contains the direction "move injection from H2 to H1":
no line flow changes, only the switch flow, and with `b_H = ∞` nothing stops
it. The intersection with an open-switch hour is bounded, since the open
network can only move power H1→H2 over lines. So `h_FTR(v_t) = ∞` while `h_∩`
is finite, and `U_t = ∞` as soon as `v_H1 ≠ v_H2`. A derate `α` scales line
limits and never touches this direction. In money: the SFT accepts an
unbounded FTR from H2 to H1 worth `v_H1 − v_H2` per MW. Two repairs, both
design choices: open the switch in the FTR model, or give it a finite one-way
cap. (B1 closed: `U = ∞` in every open hour at every `α`.)

## 2. Misalignment requires an hour whose clearings are all infeasible in another hour's network

If some clearing of every hour is feasible under every hour's network, then B2
(the stacked physical rows) contains all of them, so `V = 0` and `U = 0` in
every hour: perfect alignment along those directions, with no design at all.
So the most any lever can earn is set by how much the hours disagree. Two
things hide disagreement. One injection that is a clearing of several hours
and feasible everywhere settles them all at once. And an hour with one line
binding has a whole facet of clearings, which nearly always meets the other
network somewhere: B2 has `V = 0` along **every** facet normal of both DAM
polytopes here, 32 of 32. `V_t > 0` took two lines pinned together with a
lopsided station split; four hours with single-line bindings under a loose
rating vector gave `(0, 0)` on every rung.

## 3. Against every direction at once, B2 is the best possible FTR model

Suppose the realized directions include every facet normal of every hour's
DAM polytope. `U_t = 0` along a facet normal says the FTR polytope stays inside
that facet's halfspace. Along all of them it stays inside the whole polytope,
since a polytope is the intersection of its facet halfspaces. Inside every
hour's polytope means inside B2, and a smaller set inside the same DAM has
larger `V`. So at `τ = 0` no `(y, b)` beats B2. This is how today's practice
(stack every planned outage) falls out of the ideal formulation: it is the
answer when you refuse exposure in every direction. Design gains exist only
because realized directions are few and multi-line, and finding 2 says that
on this network B2 is not just optimal but perfectly aligned against facets.

## 4. Each switch state has a `V` the other does not; `b` cannot remove either

Two facts, then the rule.

**Opening a switch in the FTR model costs no `V` in any hour whose DAM has it
closed.** That DAM does not price across the switch, so `v_t` is equal on H1
and H2 and `v_t·q` does not change when withdrawal moves between them. Closed
network line flows do not change either, only the switch flow (the H1 and H2
columns of the PTDF agree on every line row). Slide any clearing's split until
the switch carries zero: same value, same line flows, and a closed flow pattern
with zero switch flow is a legal open-network flow pattern. So the clearing is
feasible in the open model.

**Closing a switch in the FTR model can cost `V` in an hour whose DAM has it
open.** That hour's clearing has a split fixed by prices. On the closed
network the imbalance rides the switch, the loop redistributes it, and a line
can exceed its rating (H1 −80 / H2 −10: 29 MW on the switch, ND over,
`V = 16.9` of 104.6). This `V` is topology-forced: it is there with every
limit at its rating.

So the closed model's `V` is topology-forced and comes from open hours. The
open model has no topology-forced `V` at all; its `V` is `U`-induced and comes
from closed hours: in a closed hour the open network can carry more along
`v_t` than the closed DAM can (at ratings, `U = 36.6` in the wind evening), so
limits must be pulled below rating to hold `U = 0`, and each pull can exclude
clearings that open hours need. **Each closed hour can only add cost to the
open model; each open hour can only add cost to the closed model; `b` removes
neither.** Two instances, fenced, `τ = 0`:

| hours | B2 | S1 closed | S1 open |
|---|---|---|---|
| wind evening (closed) + north midday (open) | 8.4 | 8.4 | **0** |
| + gas evening (closed) + south midday (open), two lines pinned each | 4.2 | 4.2 | 8.1 |

One closed hour: the open model's pulls cost nothing and it wins. Two closed
hours pricing different lines: the pulls exclude clearings of the three other
hours and it loses. S1 closed equals B2 in both: its only cost is the forced
one. So `b` chooses how to pay, `s` chooses which hours to pay for, and
neither can pay less than the cheaper of the two. **Open question: is the open
model's cost monotone in the number of closed hours as a theorem, or only
typically?**

## 5. Fenced limits make exposure controllable; the budget buys back only `U`-induced `V`

Unfenced, S1 reached `(0, 0)` at both states by selling ND at 35.6 against a
rating of 24.6 and paying with tighter limits elsewhere: legal in total, too
risky per line. So limits are fenced from here on (the ISO's rule, not a
theorem); the problem structure is unchanged.

**5a.** Fenced, `U` can always be driven to zero by shrinking `b`, so
`E[U] = 0` is reachable on every rung and the design problem is about `E[V]`.
This assumes the hour set is known; realized underfunding in practice is then a
forecast of the wrong hours, a different problem.

**5b.** A budget `τ > 0` does not necessarily lower `V`. It permits limits
above where `U = 0` had pulled them, so it can undo `U`-induced `V` and nothing
else. Topology-forced `V` is unchanged by any limit within the fence. S1 closed
sits at `E[V] = 8.4` for every `τ` from 0 to 8. A `τ` frontier is a property
of models whose `V` is `U`-induced; for a model whose `V` is topology-forced
the frontier is a single point.

## 6. A relaxed switch enters the model through one number between 0 and 1

Write `s = y_H / (y_H + y_th)`, where `y_th` is the Thevenin susceptance
between H1 and H2 through the lines (a constant of the network). As `y_H`
runs from 0 (open) to ∞ (closed), `s` runs from 0 to 1. Every entry of the
relaxed PTDF is `H_open + (H_closed − H_open) · s`: a straight line in `s`.
That is the LODF formula, and it means a relaxed switch adds exactly one
degree of freedom, `s`, to the design.

`V_t(s)` is then the value of an LP whose coefficients are straight lines in
`s`. Such a value is **piecewise**: on each stretch of `s` where the same rows
bind, the solution is `A(s)⁻¹ b` with `A(s)` linear in `s`, so the value is a
ratio of two polynomials in `s`; where the binding rows change, the formula
changes and the curve can kink. For a fixed injection the Thevenin line gives
the exceedance directly, but `V` is a minimum over the intersection's
injections, which moves within the hour's clearings to shed part of it, so `V`
lies below the fixed-injection curve. In the two-hour instance one set of rows
bound across the whole grid, and the north midday's `V` fit
`16.9 · y_H / (y_H + c)` with `c = 0.407` against `y_th = 0.266`: same family
as Thevenin, different constant, because it is an optimum and not a flow.

| y_H | 0 | 0.05 | 0.2 | 0.5 | 1 | 2 | 5 | 20 | ∞ |
|---|---|---|---|---|---|---|---|---|---|
| V (north midday), two-hour instance | 0 | 1.8 | 5.6 | 9.3 | 12.0 | 14.0 | 15.6 | 16.5 | 16.9 |

The curve rises monotonically here because the open model had no `U`-induced
`V` in this instance, so only the topology-forced part moves. What this buys
the design: S3 is a search over one bounded number per switch.

## 7. A susceptance strictly between open and closed can have lower `E[V]` than either state

When does this happen? The open model's `V` is `U`-induced and falls as the
switch couples, because even a weak coupling brings the FTR network's
capability along the closed hours' directions down toward the closed DAM's,
so fewer limits need pulling. The closed model's `V` is topology-forced and
rises along the curve of finding 6. If both are positive, a falling curve plus
a rising one can have its minimum in between. The two-hour instance had the
open model at zero, so nothing fell and the curve was monotone. The four-hour
instance has both positive. Fenced, `τ = 0`, `U = 0` at every point; MILP at a
0.5% gap, so each `E[V]` carries about ±0.6 and two grid points timed out:

| y_H | 0 (open) | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 2 | 5 | 20 | ∞ (closed) |
|---|---|---|---|---|---|---|---|---|---|---|
| E[V] | 8.1 | **1.2** | 1.7 | 2.2 | 2.5 | 2.9 | 3.6 | 3.9 | 4.7 | 4.7 (4.2 exact) |
| north midday (topology-forced) | 21.5 | 3.3 | 5.6 | 7.8 | 9.3 | 10.9 | 14.0 | 15.6 | 16.5 | 16.9 |
| gas evening (`U`-induced) | 1.3 | 1.5 | 1.2 | 1.0 | 0.8 | 0.7 | 0.3 | 0.1 | ~0 | ~0 |
| south midday (`U`-induced) | 9.7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The minimum is near `y_H ≈ 0.1`, below `y_th`, at about a quarter of the
better state. Two things to see. The north midday row equals finding 6's
table at every `y_H`: topology-forced `V` is a property of that hour alone and
adds across hours. The `U`-induced rows collapse at the first grid point: at
open the model pulled NH to 58.5 to hold the closed hours; at `y_H = 0.1` it
needs only 73.7, and the south midday's clearings survive. The relaxed
susceptance is not a blend of two topologies; it is a second lever, the
strength of the loop coupling, which the two states fix at their extremes.

---

## Next (the queue)

- **Fairness.** Per-constraint and per-block shares of `V` at a design
  (existing `constraint_table` / `block_table`); max or CVaR across hours.
  At a vertex direction many rows bind, so this is also where dual
  multiplicity meets the design.
- **Facet directions with a budget.** With every facet normal realized and
  `τ > 0`, the budget buys "which facets of the other state to violate": a
  frontier that is purely `U`-induced, the complement of 5b.
- **Vertex directions.** Dominance along exposing directions does not force
  containment, so S1 and S3 have room above B2 there. The MILP will be large
  (80 hours × 16–30 vertices); use the memo's iteration heuristic.
- **Other hour mixes.** Does the S3 optimum move; is the open model's cost
  monotone in the number of closed hours (finding 4's open question). Fix the
  two timed-out grid points.
- Line-outage hours. SD as a third binder in the north midday. Asymmetric
  line limits (memo's `(b⁺, b⁻)`). Settlement-point weights.
