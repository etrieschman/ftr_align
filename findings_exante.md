# Findings: the station-switch ladder on texas5

Short by design: one entry per finding that changed how we think, with the
reasoning spelled out. Setup and code: `notebooks/explore_texas5switch.py`,
`ftr_align/cases/texas5switch.py`. Notation follows the ex-ante memo.

**Words.** `t` indexes **hours** of one contract. Each hour has a **switch
state** (the DAM runs with the station switch open or closed), an injection the
DAM clears, and a **direction** `v_t` (its prices). A **clearing** of hour `t`
is any injection the DAM could have cleared at prices `v_t`; it is a face of
the DAM polytope, not one point. `U_t = h_FTR − h_∩`, `V_t = h_DAM − h_∩`,
and from the memo: `V_t = 0` iff some clearing of hour `t` is feasible in the
FTR model; `U_t = 0` iff the FTR model's reach along `v_t` is no more than the
intersection's.

**Setup.** texas5 with station H split into H1 (the NH side) and H2 (the
SH/DH side) joined by a switch; WD1/WD2 merged. One physical rating vector
for all hours, set so chosen lines bind in each hour; `v_t` is the sum of
those binding rows, hence constant on every bus that hour's DAM sees. Equal
weights.

**The ladder.** Rungs of FTR-model design, each scored by `E[U]`, `E[V]` over
the hours. From finding 4 on, designed limits are fenced at the physical ratings.

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

## 2. Misalignment needs an hour none of whose clearings fit the other switch state

A model built from physical rows (B2) has `V_t = 0` as soon as one clearing
of hour `t` is feasible under every switch state. Two things make that easy.
One injection that is a clearing of several hours and feasible under every
state settles all of them at once. And an hour with one line binding has a
whole facet of clearings, which nearly always meets the other state's polytope
somewhere. `V_t > 0` needs every clearing of the hour to fall outside: here
two lines pinned together with a lopsided station split. Four hours with
single-line bindings under a loose rating vector gave `(0, 0)` on every rung.

## 3. Opening a switch in the FTR model never costs V in hours the DAM has it closed; closing one can cost V in hours the DAM has it open

Take an hour with the switch closed. Its DAM does not price across the switch,
so `v_t` is equal on H1 and H2, and `v_t·q` does not change when withdrawal
moves between them. Closed-network line flows do not change either; only the
switch flow does (the H1 and H2 columns of the PTDF agree on every line row).
So slide any clearing's H1/H2 split until the switch carries zero: same value,
same line flows, still feasible. A closed flow pattern whose switch carries
zero is a legal flow pattern of the open network. So that clearing is feasible
in the open-switch model, and `V_t = 0` there.

The reverse fails. A clearing of an open hour has a split fixed by prices. Put
it on the closed network and the imbalance rides the switch, the loop
redistributes it, and a line can go over its rating (H1 −80 / H2 −10 put
29 MW on the switch, ND overloaded, `V = 16.9` of 104.6). So the "model every
planned outage" rule, `s_j = min_t s_{j,t}`, is exact for containment: it
never loses a clearing to a switch. What it costs is finding 4. Needs only that
the DAM leaves closed switches unlimited; a line outage has no such indifference.

## 4. The two switch states fail in two different ways, and which is cheaper depends on the mix of hours

Unfenced, S1 reached `(0, 0)` at both states by selling ND at 35.6 against a
rating of 24.6 and paying with tighter limits elsewhere: legal in total, too
risky per line. So `b` is fenced at the ratings; the problem structure is
unchanged.

The **closed** FTR model fails by **containment**. In an open hour with a
lopsided split, loop redistribution puts a line over its rating (finding 3),
and no limit choice fixes it. This cost is set by topology, once per such
hour, and does not grow with the number of closed hours.

The **open** FTR model never fails by containment (finding 3). It fails by
**reach**: in every closed hour it can carry more along `v_t` than the closed
DAM can (B1 open at `α = 1`: `U = 36.6` in the wind evening), so it must pull
limits below rating to hold `U_t = 0`, and each pull removes clearings that
open hours need. This cost accumulates with the number and diversity of
closed hours.

Both sides, `τ = 0`:

| hours | B2 | S1 closed | S1 open |
|---|---|---|---|
| wind evening (closed) + north midday (open) | 8.4 | 8.4 | **0** |
| + gas evening (closed) + south midday (open), two lines pinned each | 4.2 | 4.2 | 8.1 |

In the first, one closed hour costs the open model five limit cuts and it
still contains every clearing. In the second, the gas evening prices SD and SH,
the cuts to hold reach there remove clearings of all three other hours, and
`V` appears in each. S1 closed equals B2 in both: its one failure is the
forced one. So with one closed hour the open state wins; as closed hours
multiply, the closed state catches up. The question for S3 is whether a
susceptance between the two pays part of each cost and less than either.

## 5. Exposure is controllable; the budget can only be spent by raising a limit

Fenced, `U` can always be driven to zero by shrinking `b`, so `E[U] = 0` is
reachable on every rung and the design problem is about `E[V]`. A budget
`τ > 0` lets the FTR model's reach exceed the intersection's by `τ`, which
means raising some limit. S1 closed sits at `E[V] = 8.4` for every `τ` from 0
to 8: its limits are already at their ratings, the fence leaves nothing to
raise, and its remaining `V` is containment. `τ` helps only where a limit was
pulled below rating to hold reach, so it can come back up.

Two caveats. The fence is the ISO's rule, not a theorem, and it is what makes
`U` fully controllable here. And "controllable" assumes the hour set is known:
realized underfunding in practice is then a forecast of the wrong hours, a
different problem from this one.

## 6. The relaxed switch interpolates by Thevenin in the flows, and `V` lags it

For a fixed injection, the flow on any line at switch susceptance `y_H` is
`f_open + (f_closed − f_open) · s`, `s = y_H / (y_H + y_th)`, with `y_th = 0.266`
the Thevenin susceptance between H1 and H2 through the lines, as an LODF would
say (checked on ND: 24.6 → 40.2). `V` is a minimum over the intersection's
injections, so the minimizer moves within the hour's clearings to shed part of
the exceedance, and `V` is never more than the fixed-injection loss. The whole
relaxed PTDF is affine in the one scalar `s`, so while the binding set of the
LP stays fixed its value is a ratio of affine functions of `s`, which is again
`y_H / (y_H + c)` with a different `c`; a change of binding set gives a kink.
Two-hour instance, `τ = 0`, `U = 0` throughout, one binding set all the way,
`c = 0.407`:

| y_H | 0 | 0.05 | 0.2 | 0.5 | 1 | 2 | 5 | 20 | ∞ |
|---|---|---|---|---|---|---|---|---|---|
| V (north midday) | 0 | 1.8 | 5.6 | 9.3 | 12.0 | 14.0 | 15.6 | 16.5 | 16.9 |

Monotone and concave when one corner is ideal. The four-hour instance, where
neither corner is, is in finding 7.

## 7. Between the two switch states there is a susceptance that beats both

Four-hour instance, fenced, `τ = 0`, `U = 0` at every point. MILP with a 0.5%
gap, so each `E[V]` is uncertain by about ±0.6; two points hit the time limit.

| y_H | 0 (open) | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 2 | 5 | 20 | ∞ (closed) |
|---|---|---|---|---|---|---|---|---|---|---|
| E[V] | 8.1 | **1.2** | 1.7 | 2.2 | 2.5 | 2.9 | 3.6 | 3.9 | 4.7 | 4.7 (4.2 exact) |
| V north midday | 21.5 | 3.3 | 5.6 | 7.8 | 9.3 | 10.9 | 14.0 | 15.6 | 16.5 | 16.9 |
| V gas evening | 1.3 | 1.5 | 1.2 | 1.0 | 0.8 | 0.7 | 0.3 | 0.1 | ~0 | ~0 |
| V south midday | 9.7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The minimum is near `y_H ≈ 0.1`, below the Thevenin value 0.266, at roughly a
quarter of the better corner. The two costs of finding 4 move in opposite
directions in `y_H`:
- The **containment** cost (north midday) rises with `y_H` along the Thevenin
  curve of finding 6, and its values are identical to the two-hour instance at
  every `y_H`: it is a property of that hour alone, additive across hours.
- The **reach** cost (gas evening, south midday) falls fast from the open
  corner. At `y_H = 0` the model has no switch row and must cut five lines to
  hold reach in the closed hours (NH 58.5, DH 40.3). Even a weak coupling
  brings the FTR network's reach along the closed hours' directions down toward
  the closed DAM's, so fewer cuts are needed (NH 73.7 at `y_H = 0.1`), and the
  south midday's clearings survive.

A rising curve plus a falling one gives an interior minimum. The relaxed
susceptance is not a compromise between two topologies; it is a second lever
(the strength of the loop coupling) that the two corners fix at their extremes.

## Open questions

- Per-constraint and per-block shares of U and V (fairness): existing
  `constraint_table` / `block_table` on a designed model; max or CVaR across
  hours on top.
- Line-outage hours. SD as a third binder in the north midday. Asymmetric line
  limits (memo's `(b⁺, b⁻)`), parked as S5. Settlement-point weights.
