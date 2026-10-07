# Findings: the station-switch ladder on texas5

Kept deliberately short. One entry per finding that changed how we think, with
the reasoning spelled out. Setup and code: `notebooks/explore_texas5switch.py`,
`ftr_align/cases/texas5switch.py`. Notation follows the ex-ante memo.

**Setup.** texas5 with station H split into H1 (the NH side) and H2 (the SH/DH
side) joined by a switch; WD1/WD2 merged. Two hours of one contract: t1 evening,
switch closed; t2 midday, switch open. One physical rating vector for both,
set so chosen lines bind in each hour (t1: DH, SH; t2: NH, ND). Each hour's
direction `v_t` is the sum of its binding rows, so it is constant on every bus
the DAM sees. `U_t = h_FTR - h_∩`, `V_t = h_DAM - h_∩`, equal weights.

**The ladder** (rungs of FTR-model design, each evaluated by `E[U]`, `E[V]`):
B1 one physical state, every limit derated by `α`. B2 the stacked rows of
every hour's physical state ("model every planned outage"). S1 limits `b`
designed at a fixed state, by enumeration over the dual vertices of `Λ(v_t)`.
S2 S1 at every switch state, best kept. S3 the switch given a continuous
susceptance `y_H`, S1 at each. S4 every susceptance free (not reached).
Designed limits are fenced at the physical ratings from finding 5 on.

---

## 1. A closed switch with no limit is infinite exposure the moment prices split

The _closed_ FTR polytope contains the direction "move injection from H2 to H1".
No line flow changes, only the switch flow, and with `b_H = ∞` nothing stops
it. The intersection with the open-switch DAM is bounded, because the open
network can only move power H1→H2 over lines. So `h_FTR(v_t) = ∞` and
`h_∩(v_t) < ∞`, hence `U_t = ∞`, as soon as `v_H1 ≠ v_H2`. A derate `α`
scales line limits and never touches this direction. In money: the SFT accepts
an unbounded FTR from H2 to H1 worth `v_H1 − v_H2` per MW. Two repairs, both
design choices: open the switch in the FTR model, or give it a finite
cap. (B1 closed: `U_2 = ∞` at every `α`.)

## 2. U and V are the two excess reaches along a direction

Along `v_t` each polytope has a reach, its support value. The intersection
never reaches further than either parent: `h_∩ ≤ h_FTR`, `h_∩ ≤ h_DAM`.
- `V_t` is how far the **DAM** reaches beyond the intersection along `v_t`:
  rent the market collects that the auction never sold.
- `U_t` is how far the **FTR model** reaches beyond the intersection along
  `v_t`: payouts the auction promises at prices `v_t` that the market cannot
  back.

The zero conditions are mirror images. `V_t = 0` iff some DAM-optimal point is
FTR-feasible. `U_t = 0` iff some FTR-optimal point is DAM-feasible. Each is
containment of the *other* model's best point. The asymmetry that matters for
design: `U` is fixed by shrinking the FTR model's reach, and shrinking is also
what breaks `V`.

## 3. Misalignment needs an hour whose whole optimal face leaves the other state

By finding 2, a model built from physical rows (B2) has `V_t = 0` as soon as
*one* DAM-optimal point of hour `t` is feasible in every state. Two things
make that easy. An injection that is optimal in several hours and feasible in
every state settles all of them at once. And an hour whose optimum is a whole
facet (one line binding) nearly always has some point of it inside the other
state's polytope. `V_t > 0` needs the hour's entire optimal face to lie
outside: two lines pinned together, in this network with a lopsided station
split. One run put four hours with single-line bindings under one rating
vector and every rung reached `(0, 0)`.

## 4. In version A, the closed hour never produces V for B2

Holds for *any* closed-hour injection.
- The closed DAM prices H1 and H2 equally (it never prices across a switch it
  closed). So `v_1·q` does not change when withdrawal moves between H1 and H2.
- Closed-network *line* flows do not change either; only the switch flow does.
  (PTDF table: the H1 and H2 columns are identical on every line row.)
- Take the closed DAM's optimum and slide the H1/H2 split until the switch flow
  is exactly zero. Same value, same line flows, still feasible.
- That point is also feasible on the *open* network: the open network is the
  closed one with the switch removed, and a closed flow pattern whose switch
  carries zero is a valid open flow pattern (same angles, same line flows).
- So it lies in B2 and reaches the closed DAM's value. `V_1 = 0`.

Only the open hour can produce V, and only when its priced split is far from
the closed network's natural one. H1 −50 / H2 −40 put 0.6 MW on the closed
switch and gave `V_2 = 0`; H1 −80 / H2 −10 put 29 MW on it, the redistribution
overloaded ND under the closed state, and `V_2 = 16.9` of 104.6. Binding NH
alone was not enough (the facet left room to dodge); NH and ND together pinned
it. Version B (a line outage) has no such indifference and both hours can
produce V.

## 5. Unfenced, limits alone reach the ideal by selling above rating; fenced, the state matters

Unfenced S1 reached `(E[U], E[V]) = (0, 0)` at both states. The closed-state
design sold ND at 35.6 against a rating of 24.6, exactly the line B2 tripped
over, and paid for it with NH 59 (rating 80), WN 10.5 (12) and a 20.7 one-way
switch cap. Per line unphysical; in total never more along either direction
than the DAM collects, which is all `U = 0` asks (finding 2). Too risky to
rely on, so `b` is fenced at the ratings from here on. That is a no-regrets
rule and leaves the problem structure unchanged.

Fenced, at `τ = 0`:

| rung | E[U] | E[V] |
|---|---|---|
| B1 closed | ∞ | any |
| B1 open, α = 0.6 | 0 | 26.6 |
| B2 stack | 0 | 8.4 |
| S1 closed | 0 | 8.4 |
| S1 open | 0 | 0 |
| S2 (best state) | 0 | 0 |

S1 closed equals B2 to the decimal, and this is forced: every open-hour optimum
has ND's *closed-state* flow above 24.6, so no fenced closed model can contain
one. S1 open reaches the ideal: by finding 4 it can contain the closed hour's
optimum (slid to zero switch flow), and the open hour is its own physics; it
tightened WN, WS, WD, SH, DH below rating to hold the ceiling in the closed
hour. So the rung that matters here is S2, and it picks the *open* state: the
more constrained topology, the "model every planned outage" rule
`s_j = min_t s_{j,t}`, but with designed limits rather than a derate.

Why the smaller polytope wins: V is not about size, it is about containing the
right points (finding 2). The closed polytope is larger but shaped by loop
redistribution that excludes the open hour's optimum; the open polytope is
smaller and holds both hours' optima.

With two scenarios, eight line limits and a switch cap have slack to spare, so
`b` alone saturates. Upper rungs can only matter when scenarios outnumber the
freedom in `b` or `b` is fenced.

**The open state's advantage is not general.** Four hours, two lines pinned in
each (wind evening DH, WD; gas evening SD, SH; north midday NH, ND; south
midday DH, WD), 10% margin, fenced, `τ = 0`:

| rung | E[U] | E[V] | where V lands |
|---|---|---|---|
| B2 stack | 0 | 4.2 | north midday only (16.9) |
| S1 closed | 0 | 4.2 | north midday only (16.9) |
| S1 open | 0 | 8.1 | gas evening 1.3, north midday 21.5, south midday 9.7 |

A second closed hour pricing different lines (SD, SH) forces the open model
to tighten five limits to hold its reach in that hour (finding 2), and those
tightenings cut off the open hours' optima. The closed model's one failure is
the forced one from above. So neither corner is ideal here, and this is the
first instance where S3 has something to show.

## 6. Fenced, the exposure budget buys the closed model nothing

S1 closed has `E[V] = 8.4` at every `τ` from 0 to 8. `V_2` is a containment
failure (finding 5), every closed-model limit already sits at its rating, and
spending budget means selling more, which the fence forbids. A `τ` frontier
exists only where the ceiling (`U`) forced limits below rating; here that is
the open model, which already has `V = 0`. So in this instance `τ` matters
nowhere.

## 7. The relaxed switch interpolates monotonically; no middle is worse than both ends

S3, fenced S1 at each switch susceptance `y_H`, `τ = 0`, `U = 0` and `V_1 = 0`
throughout:

| y_H | 0 | 0.05 | 0.2 | 0.5 | 1 | 2 | 5 | 20 | 100 | ∞ |
|---|---|---|---|---|---|---|---|---|---|---|
| V_2 | 0 | 1.8 | 5.6 | 9.3 | 12.0 | 14.0 | 15.6 | 16.5 | 16.8 | 16.9 |

Monotone and concave, saturating toward the closed value. The one-way switch
cap the design picks grows with it (3.5 → 31.7). Empirically the curve is
exactly `V_2 = 16.87 · y_H / (y_H + c)` with `c = 0.407` at every point: the
shape of a path in parallel with the rest of the network. But `c` is not the
H1–H2 Thevenin susceptance through the lines (0.266), because the intersection
re-optimizes its injection at each `y_H`. Shape understood, constant open.

Consequence for the ladder: with the open corner already ideal, the relaxed
susceptance cannot beat S2 here and is strictly worse in between. Testing S3's
claim needs an instance where *neither* corner is ideal, which means more
scenarios than `b` can fit (several closed hours pricing differently, so the
open model must tighten lines, plus open hours where those tightenings cost V).

## Open questions

- The constant `c = 0.407` in finding 7.
- Does "open dominates" survive more scenarios? Its cost is the ceiling: the
  open model tightened five lines to hold `U_1 = 0`, and more closed-hour
  scenarios may make those tightenings cost V in the open hour.
- Per-constraint and per-block shares of U and V (fairness): existing
  `constraint_table` / `block_table` on a designed model; max or CVaR across
  scenarios on top.
- SD as a third t2 binder (completes the SDH triangle). Version B (line
  outage). Asymmetric line limits (memo's `(b⁺, b⁻)`), parked as S5.
