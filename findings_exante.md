# Findings: the station-switch ladder on texas5

Short by design: one entry per finding that changed how we think, with the
reasoning spelled out. Setup and code: `notebooks/explore_texas5switch.py`,
`ftr_align/cases/texas5switch.py`. Notation follows the ex-ante memo.

**Setup.** texas5 with station H split into H1 (the NH side) and H2 (the SH/DH
side) joined by a switch; WD1/WD2 merged. Hours of one contract, some with the
switch closed, some open. One physical rating vector for all hours, set so
chosen lines bind in each. Each hour's direction `v_t` is the sum of its
binding rows, so it is constant on every bus that hour's DAM sees.
`U_t = h_FTR − h_∩`, `V_t = h_DAM − h_∩`, equal weights.

**The ladder.** Rungs of FTR-model design, each scored by `E[U]`, `E[V]` over
the hours. From finding 5 on, designed limits are fenced at the physical ratings.

| rung | FTR model `Q(y, b)` | what is chosen |
|---|---|---|
| B1 | one physical switch state, every line limit `× α` | `α` |
| B2 | rows of every hour's physical state, stacked ("model every planned outage") | nothing |
| S1 | one physical switch state, limits `b` designed | `b`, via the dual vertex of `Λ(v_t)` per hour (enumeration, or MILP with one binary per vertex) |
| S2 | S1 at every switch state, best kept | state and `b` |
| S3 | switch given a continuous susceptance `y_H`, S1 at each grid point | `y_H` and `b` |
| S4 | every element's susceptance free | `y` and `b` (not reached) |

---

## 1. A closed switch with no limit is infinite exposure the moment prices split

The closed FTR polytope contains the direction "move injection from H2 to H1":
no line flow changes, only the switch flow, and with `b_H = ∞` nothing stops
it. The intersection with an open-switch DAM is bounded, since the open network
can only move power H1→H2 over lines. So `h_FTR(v_t) = ∞` while `h_∩` is
finite, and `U_t = ∞` as soon as `v_H1 ≠ v_H2`. A derate `α` scales line
limits and never touches this direction. In money: the SFT accepts an
unbounded FTR from H2 to H1 worth `v_H1 − v_H2` per MW. Two repairs, both
design choices: open the switch in the FTR model, or give it a finite one-way
cap. (B1 closed: `U_2 = ∞` at every `α`.)

## 2. U and V are the two excess reaches along a direction

Along `v_t` each polytope has a reach, its support value, and the intersection
never reaches further than either parent. `V_t` is how far the **DAM** reaches
beyond the intersection: rent the market collects that the auction never sold.
`U_t` is how far the **FTR model** reaches beyond it: payouts the auction
promises at prices `v_t` that the market cannot back. The zero conditions are
mirror images: `V_t = 0` iff some DAM-optimal point is FTR-feasible, `U_t = 0`
iff some FTR-optimal point is DAM-feasible. The design asymmetry: `U` is fixed
by shrinking the FTR model's reach, and shrinking is also what breaks `V`.

## 3. Misalignment needs an hour whose whole optimal face leaves the other state

By finding 2, a model built from physical rows (B2) has `V_t = 0` as soon as
one DAM-optimal point of hour `t` is feasible in every state. An injection
that is optimal in several hours and feasible in every state settles all of
them at once. An hour whose optimum is a whole facet (one line binding) nearly
always has some point inside the other state's polytope. `V_t > 0` needs the
hour's entire optimal face outside it: here two lines pinned together with a
lopsided station split. Four hours with single-line bindings under a loose
rating vector gave `(0, 0)` on every rung.

## 4. In version A, a closed hour never produces V for B2

Holds for any closed-hour injection.
- The closed DAM prices H1 and H2 equally, so `v_t·q` does not change when
  withdrawal moves between them.
- Closed-network line flows do not change either; only the switch flow does
  (the H1 and H2 columns of the PTDF agree on every line row).
- Slide the closed optimum's split until the switch flow is zero: same value,
  same line flows, still feasible.
- That point is feasible on the open network too: a closed flow pattern whose
  switch carries zero is a valid open flow pattern.
- So it lies in B2 and reaches the closed DAM's value. `V_t = 0`.

Only open hours can produce V, and only when the priced split is far from the
closed network's natural one (H1 −80 / H2 −10 put 29 MW on the closed switch,
ND overloaded under the closed state, `V = 16.9` of 104.6). A line outage
(version B) has no such indifference.

## 5. Fenced at the ratings, the switch state matters, and which state wins depends on the hours

Unfenced, S1 reached `(0, 0)` at both states by selling ND at 35.6 against a
rating of 24.6 and paying with tighter limits elsewhere: unphysical per line,
legal in total by finding 2, too risky to rely on. So `b` is fenced at the
ratings; the problem structure is unchanged.

Two hours (wind evening closed; lopsided north midday open), `τ = 0`:

| rung | E[U] | E[V] |
|---|---|---|
| B2 stack | 0 | 8.4 |
| S1 closed | 0 | 8.4 |
| S1 open | 0 | 0 |

S1 closed equals B2 and this is forced: every open-hour optimum has ND's
closed-state flow above its rating. S1 open is ideal: by finding 4 it contains
the closed hour's optimum, and the open hour is its own physics; it pulled
five limits below rating to hold its reach in the closed hour. V is about
containing the right points, not polytope size.

Four hours (two closed pricing DH, WD and SD, SH; two open pricing NH, ND and
DH, WD), `τ = 0`:

| rung | E[U] | E[V] | where V lands |
|---|---|---|---|
| B2 stack | 0 | 4.2 | north midday only |
| S1 closed | 0 | 4.2 | north midday only |
| S1 open | 0 | 8.1 | three of four hours |

The second closed hour forces the open model to hold its reach there as well,
and the limits it pulls down cut off the open hours' optima. What generalizes
is the mechanism, not the winner: each closed hour costs the open model
reach, each lopsided open hour costs the closed model containment, and the
balance depends on the hours. With few hours `b` has slack to spare and the
rung above S1 cannot matter; with more hours or a fence it can.

## 6. The exposure budget can only be spent by raising a limit

`τ > 0` lets the FTR model reach beyond the intersection by `τ`, which means
raising some limit. Fenced, a model whose limits already sit at their ratings
has nothing to raise, and its remaining V is a containment failure (finding 3)
that a higher limit, if allowed, would fix. So S1 closed sits at `E[V] = 8.4`
for every `τ` from 0 to 8. A `τ` frontier exists only where the reach
condition, not the fence, is what pulled a limit down.

## 7. The relaxed switch interpolates by Thevenin in the flows, and the optimizer lags it in V

For a fixed injection, the flow on any line at switch susceptance `y_H` is
`f_open + (f_closed − f_open) · y_H / (y_H + y_th)` with `y_th = 0.266` the
Thevenin susceptance between H1 and H2 through the lines, exactly as an LODF
would say (checked on ND: 24.6 → 40.2). The value `V` follows the same shape,
`V_closed · y_H / (y_H + c)`, but with `c = 0.407`: the intersection
re-optimizes its injection at each `y_H` (no generators or costs here, only
the support LP moving `q`), recovering part of the exceedance, so V grows more
slowly than the physics. Two-hour instance, `τ = 0`, `U = 0` throughout:

| y_H | 0 | 0.05 | 0.2 | 0.5 | 1 | 2 | 5 | 20 | ∞ |
|---|---|---|---|---|---|---|---|---|---|
| V_2 | 0 | 1.8 | 5.6 | 9.3 | 12.0 | 14.0 | 15.6 | 16.5 | 16.9 |

Monotone and concave between the corners when one corner is ideal. The
four-hour instance, where neither is, is the test of whether a middle `y_H`
beats both (see below once run).

## Open questions

- Per-constraint and per-block shares of U and V (fairness): existing
  `constraint_table` / `block_table` on a designed model; max or CVaR across
  scenarios on top.
- Version B (line outage). SD as a third t2 binder. Asymmetric line limits
  (memo's `(b⁺, b⁻)`), parked as S5. Settlement-point weights.
