# Structure and results worth putting in the paper

General claims, each with the mechanism that makes it true. They were arrived at
by running the cases in `notebooks/`, but the figures themselves belong in the
runs, not here — what a specific design measured today is not a claim.

Engineering notes live in `CLAUDE.md`; what is still open lives in `PLAN.md`.

---

## 1. The ambient geometry

**Every PTDF row acts on a space of dimension `n − 1`.** Balanced injections
satisfy `1ᵀq = 0`. On that subspace `kᵢᵀq = k̄ᵢᵀq`, where `k̄ᵢ = kᵢ − mean(kᵢ)`.
So a row only ever acts through its mean-removed part, and every `k̄ᵢ` lies in
`1⊥`, a space of dimension `n − 1`.

Four consequences, each a direct count.

**(a) `m` rows carry `m − rank` independent dependencies, and `rank ≤ n − 1`.**
For the base case of a connected graph with every element monitored, the rank is
exactly `n − 1`, so there are `E − (n − 1)` dependencies. That count coincides
with the graph's cycle space, `E − V + 1`, and KVL supplies the dependencies
themselves: `Σ_{e ∈ cycle} xₑ kₑ = 0` around any cycle.

**(b) A contingency adds rows to the *same* space, never a dimension.** The rank
is already saturated at `n − 1`, so each new row adds exactly one dependency.
Adding a single outage therefore multiplies the number of dependencies several
times over, without introducing any new geometry.

> **This is the paper's structural claim about contingencies.** Almost every one
> of those new dependencies mixes base rows with post-outage rows. Circulation
> spanning a contingency boundary is *generic*, not exotic — and the cycle space
> of the network graph, which describes the base case alone, does not see it. The
> object is the stacked system.

**(c) Minimal dependent sets have at most `n` rows.** Any `n` vectors in an
`(n − 1)`-dimensional space are dependent, so a circuit never exceeds `n` rows.

**(v) Circuit size determines the face it exposes.** If exactly the rows of a
circuit `S` are tight, the optimal face is `{q : k̄ᵢᵀq = bᵢ, i ∈ S}`, of dimension

```
(n − 1) − rank(k̄_S) = (n − 1) − (|S| − 1) = n − |S|
```

A maximal circuit (`|S| = n`) exposes a vertex; the smallest (`|S| = 2`) exposes
a face of dimension `n − 2`. **Small circuits are the ones that leave room to
move**, which is what §4 needs.

**The slack bus is a labelling convention.** PTDF rows under two slack
conventions differ by a multiple of `1ᵀ`, which annihilates balanced injections.
Changing it shifts `v` by a constant vector and leaves `Q(b)`, `h` and `μ`
untouched. Choosing the slack to suit a figure's axes is free.

---

## 2. When attribution rises above a single constraint

**The block condition.** Rows `S` share an attribution block iff some `d`
supported on `S` satisfies **both**

```
(a)  Σ_{i∈S} dᵢ k̄ᵢ = 0        the trade moves no flow
(b)  Σ_{i∈S} dᵢ bᵢ = 0        and costs no value
```

**(a) is limit-free.** It is pure linear algebra on the stacked PTDF, so it can
be checked before any limits are chosen. A row set failing it can never be a
block, at any `b`. This is the cheap screen, and it is what makes "which sets
*could* be blocks" a finite enumeration rather than a search.

**(b) is automatic whenever the whole circuit binds.** If every row of `S` is
tight at the optimum `q`, then

```
Σ dᵢ bᵢ = Σ dᵢ (Kq)ᵢ = (Σ dᵢ kᵢ)ᵀ q = 0
```

by (a). So **a circuit that binds is a block** — condition (b) is a consequence,
not an extra requirement. Nothing needs to be imposed on the limits.

**A nondegenerate vertex has only singleton blocks.** There `J*` holds `n − 1`
independent rows, so `Σ dᵢ k̄ᵢ = 0` forces `d = 0`. Non-trivial attribution
structure lives on faces, not corners.

**Block count is not ambiguity; corank is.** Blocks *partition* the priced rows,
so `n_blocks = n_priced` is the fully-identified case — two rows that cannot
trade give **two** singleton blocks, not one ambiguous pair. The ambiguity is
`dim ker C`, equivalently `max_block > 1`.

**A block's trade space is its true corank, not `|B| − 1`.** The shortcut assumes
every block has corank exactly 1, and large blocks routinely have corank well
below `|B| − 1`. The identity that does hold is that per-block coranks sum to
`dim ker C` over all of `J*`, because the trade space splits as a direct sum over
blocks. That is the check worth running.

**A uniform derate preserves block structure; a selective one destroys it.**
Scaling every limit by `α` leaves (b) satisfied: `Σ dᵢ(α bᵢ) = α · 0 = 0`.
Derating only some rows of a circuit breaks exactly the block it was built to
make.

**A binding cross-contingency block need not be a funded one.** Take `f` base-only
and `g` = base + one outage, and let a circuit `S` bind, split into base rows
`S_b` and contingency rows `S_c` that `f` cannot price at all. Underfunding is a
difference of *support values*, not of binding counts, and

```
U = h(f;y) − h(g;y) = 0   ⟺   v ∈ cone{kᵢ : i ∈ S_b} + span{1}
```

**This is Farkas, so it is the criterion and not a proxy.** The design makes `S`
the only tight set, so at `q` the model `f` improves iff some `Δq` satisfies

```
kᵢᵀ Δq ≤ 0   for i ∈ S_b        stay feasible
vᵀ Δq > 0                        strictly improve
```

and Farkas says no such `Δq` exists exactly when `v ∈ cone(S_b) + span{1}` — the
cone, not the span, because `f`'s multipliers must be non-negative. The value
then matches by (b): the weights `w` are a dual-feasible certificate for `f`, and
`Σ wᵢ bᵢ = Σ_S bᵢ` is precisely what (b) asserts.

So a contingency row can be tight, priced with `μᵢ > 0`, and still cut nothing
off `Q(f)`: condition (b) places its hyperplane exactly through the corner the
base rows already form. **The relation that makes the rows one block is the same
relation that can make the contingency rows redundant.**

**The weights are unique, so the test needs no LP.** A *proper subset* of a
circuit is independent, so `K_{S_b}` has full rank and the `w` solving
`v = K_{S_b}ᵀ w + t·1` is unique wherever it exists. Least squares finds it;
the test is then the residual and the signs. Two failures, and **both carry
content**:

- **no solution** — `v` leaves `span(S_b)`. Vacuous when `|S_c| = 1`, since any
  `|S| − 1` rows of a circuit span it; the usual outcome when `|S_c| ≥ 2`, which
  is why the contingency-row count is such a strong empirical proxy (976 of 1009).
- **a solution with a negative weight** — spanned, but still outside the cone.

**So a span test is not the criterion.** It sees the first failure and is blind
to the second, and it errs in the dangerous direction: it calls a *live* design
dead. On the 5-node it is wrong on **173 of 1683** realizable designs, 165 of
them with a single contingency row — exactly the regime where spanning is
automatic and only the signs carry information. Span is a statement about row
spaces and does not involve `v`; the cone is where the direction re-enters.
Checked against the cone LP on all 1683: no disagreements.

**The smallest such block is an LODF triple, and it is always in the cone.**
`{base:o, base:e, c_o:e}` is dependent because its `d` **is** the outage-transfer
identity

```
k_e^{c_o} = k_e + L(e,o) · k_o
```

Take the three rows on sides `s_o, s_e, s_c ∈ {±1}`. The base rows `f` can price
are `s_o k_o` and `s_e k_e`, and the recombination weights come out in closed
form:

```
w_e = 1 + s_c·s_e   ∈ {0, 2}          always ≥ 0
w_o = 1 + s_c·s_o·L                    ≥ 0  ⟺  |L| ≤ 1
```

So the cone test collapses to **`|L| ≤ 1`, independent of the sides** — the
`w_e` branch can never fail, and the sign choice only picks which of `1 ± L` the
other weight is. `|L| ≥ 1` means the outage more than fully reverses the flow, a
near-radial situation excluded with the bridges, so **every LODF triple carries
`U = 0`, on any side assignment** — the outage row is priced and worth nothing.
Verified on the 5-node across all 70 genuine triples × 8 side assignments, zero
failures, `|L| ∈ [0.043, 0.769]`, giving `min w_o = 1 − max|L| = 0.23`. (The
parallel pair `WD1`/`WD2` is excluded: `k_o` and `k_e` are collinear there, so it
is rank 1, not a circuit of three independent directions.)

**So escaping the cone is a statement about topology, not sides.** A spanning
circuit escapes only by *not* being an LODF triple — by not containing the
outaged element's own base row, so that the contingency row is tied to base rows
other than `o`. That is necessary but not sufficient: of the 133 realizable
non-LODF size-3 spanning designs, 75 escape and 58 do not, while all 116 LODF
ones stay in. Which is why the cone LP is run rather than pattern-matched.

**Block size is not a usable proxy for the test.** Over 1683 realizable
spanning-circuit designs on the 5-node the cone test partitioned `U` exactly —
1141 out-of-cone every one with `U > 0`, 542 in-cone every one with `U = 0`, no
disagreements. Restricting to `|S| ≥ 4` still leaves 368 of 1434 in the cone;
`|S_c| ≥ 2` is the better structural proxy (976 of 1009) but is also not exact.
The test itself is one small LP, cheaper than the design LP it screens.

**And it does not need the block broken.** Detuning the contingency limit off its
designed value restores `U`, but only by destroying the circuit — the row above
about selective derates. Out-of-cone designs need no such thing: they realize
`J*` equal to the pattern, as a single size-3 cross-contingency block with
`dim ker C = 1`, *and* `U > 0` at the same limits.

---

## 3. The gap between the floor and the failure mode is a duality gap

**The claim in one line: `loss − floor` is the duality gap of the model's
certificate against the target.** So the floor is tight exactly when that
certificate is still optimal for the target, and slack by however much it is not.

**Why the floor bounds the mode.** With `μ` the model's certificate,
`h(model) = bᵀμ` by strong duality. The target shares `K` and `v`, so `μ` is
dual-*feasible* for it, and weak duality gives `h(target) ≤ b_targetᵀμ`.
Subtracting,

```
loss = h(model) − h(target) ≥ bᵀμ − b_targetᵀμ = Σᵢ μᵢ(bᵢ − bᵢ^target) = floor
```

**Why the slack is a duality gap.**

```
loss − floor = b_targetᵀμ − h(target)
```

which is exactly the gap of `μ` evaluated against the target's problem. Hence

> `floor_ratio = 1` ⟺ the model's certificate is also a certificate for the target.

**What makes it slack, row by row.** By complementary slackness `μ` stays optimal
for the target only if every row it prices is tight at the target's optimum *at
the target's limit*. So among priced rows:

| priced row | contributes to `floor` | contributes to `h(model)` |
|---|---|---|
| models **disagree** (`bᵢ > bᵢ^target`) | `μᵢ(bᵢ − bᵢ^target) > 0` | `μᵢbᵢ` |
| models **agree** (`bᵢ = bᵢ^target`) | **0** — the terms cancel | `μᵢbᵢ` |

**The second row is the whole story.** A priced row where the models agree is
part of why the model is worth more than the target, but the floor cannot see it,
because the floor measures only disagreement.

**Hence the sharpened form of `cor:canonical` item 1.** A derate uniform over
*every* row gives `Q(target) = α·Q(model)`; the optimum scales, the same rows
bind, the same `μ` stays optimal, and the ratio is exactly 1. The uniformity must
hold over the **whole stacked system**, not over one contingency.

**And hence the floor is a gauge, not a switch.** The reading that the ratio is
always 0 or 1 is an artifact of cases where the disagreement happens to cover the
entire priced support. Split the derate across the stack and it does not: with
`f` enforcing only the base case and `g` enforcing base plus an outage, the
intersection inherits `g`'s outage limits untouched, so every direction whose
certificate prices an outage row lands strictly inside `(0, 1)`. That is the
typical case, not the exception.

A pure coverage difference still gives exactly 0: an unmonitored row forces
`μᵢ = 0`, so it can carry no floor at all.

---

## 4. Identification is primal multiplicity of the intersection

**`identified` is a statement about `Q(f ∧ g)` having more than one optimum.**
Four steps.

**(i) The share is read at a point, and the point is a choice.**

```
share(B) = Σ_{i∈B} μᵢ [ bᵢ − (Kq)ᵢ ]
```

`μ` is the **model's** certificate (`f`'s for U, `g`'s for V). `q` is a maximiser
for the **target** `f ∧ g`. The number is the model's dual weights evaluated at a
point of the intersection.

**(ii) It is affine in `q`.**

```
share(B) = const − wᵀq,    w = Σ_{i∈B} μᵢ kᵢ,    const = Σ_{i∈B} μᵢ bᵢ
```

Only `wᵀq` depends on the choice.

**(iii) So the question is whether `wᵀq` is constant on the target's optimal
face.** A unique maximiser forces `q`, and the share is a number — `identified`
holds, but *vacuously*. On a face of positive dimension you can move by any `Δq`
with `1ᵀΔq = 0` and `K_{J*(f∧g)} Δq = 0`, and the share is invariant to all of them
exactly when

```
w ∈ span{1} + row(K_{J*(f∧g)})
```

`False` means the share is genuinely a **different number at different optima**.
It is an interval, not a value.

**(iv) Where it fails, and why that is structural.** Point `v` along a single
row's normal, `v = kᵢ`. The maximiser set is then the whole facet, and
`J*(f ∧ g)` is essentially `{i}`, so the test subspace is only `span{1, kᵢ}` —
two dimensions inside `Rⁿ`.

Now the asymmetry between the modes, which is the interesting part:

- **V** (`model = g`): `g` enforces row `i`, so at `v = kᵢ` its certificate
  concentrates there, `w ∝ kᵢ`, and the test passes.
- **U** (`model = f`): `f` is base-only and does not have row `i` at all. Its
  certificate lands on whichever *base* rows bind, so `w` is a combination of
  base normals with no reason to lie in `span{1, kᵢ}`.

> **The rule: a block's share stops being identified exactly where the direction
> points along a constraint the model is blind to.** Failures therefore appear in
> `U` at the facets of contingencies the FTR model omits, and never in `V`, whose
> model enforces them. The witness is a structural property of a model missing a
> whole contingency — not a knife-edge coincidence, and not something that
> disappears under perturbation.

**Two independent computations agree.** `primal_invariant` is a span test;
`block_share_range` is two LPs over the same face. They agree block for block,
and the widths they report separate by orders of magnitude between the identified
and unidentified cases. The threshold must scale with `h`: an identified block
still shows a width of order the face-construction leak, so a fixed absolute
tolerance misclassifies it.

---

## 5. The regime map replaces the derate search

**Faces of `Q(b)` and cones of its normal fan are dual.** A vertex corresponds to
a full-dimensional cone of directions, so enumerating the vertices of `Q(f ∧ g)`
and taking one direction interior to each cone gives a **complete, non-redundant
list of regimes** — every qualitatively distinct binding pattern the pair admits,
with no price certificate posited and no market cleared.

This turns "does this pair exhibit both failure modes at once?" from a search
over derates into an enumeration — and on a network with contingencies the answer
is *nearly every regime*, with no tuning at all. That is worth stating, because
on a small network without contingencies both modes coexist only at a
hand-picked derate and in one scenario.

**Why both modes appear at nearly every vertex.** Let `q*` be a vertex of
`Q(f ∧ g)` and `v` a direction interior to its normal cone, so `h(f∧g;v) = vᵀq*`.
Then

```
U = h(f;v) − vᵀq* = 0   ⟺   q* maximizes v over Q(f)   ⟺   v ∈ N_f(q*)
V = h(g;v) − vᵀq* = 0   ⟺   q* maximizes v over Q(g)   ⟺   v ∈ N_g(q*)
```

because `Q(f∧g) ⊆ Q(f)` makes `vᵀq* ≤ h(f;v)` always, with equality exactly when
`q*` is already `f`-optimal. For polyhedra the normal cone of an intersection is
the Minkowski sum of the normal cones,

```
N_{f∧g}(q*) = N_f(q*) + N_g(q*)
```

so the two statements compose into one:

> **Both failure modes are strictly positive exactly when
> `v ∈ (N_f(q*) + N_g(q*)) \ (N_f(q*) ∪ N_g(q*))`** — the exposing direction needs
> generators from *both* models and lies in neither cone alone.

**This is why it is generic rather than tuned.** The sum of two cones fills the
entire wedge between them while the union is only the two cones, so whenever both
cones are proper the sum is strictly larger and most of it lies outside both. And
`faces` builds `v = Σ_{i tight} kᵢ`, a positive combination of *every* tight row,
which takes generators from both models by construction and therefore lands
between them by construction.

**The only way to fail is a vertex inherited whole from one parent.** If one
model has no binding row at `q*` its normal cone is trivial, `N_{f∧g}(q*)` reduces
to the other model's, `v` lies inside it, and that model's mode is exactly zero
while the other stays positive. So a vertex fails to carry both modes iff its
tight set is single-model — a classification, not a statistic, and the reading to
check a sweep against.

**At a vertex of `Q(f)` the sign of `U` is a feasibility test, not an
optimization.** Take `q*` a vertex of `Q(f)` and `v` interior to `N_f(q*)`, so `q*`
is the *unique* `f`-maximizer and `h(f;v) = vᵀq*`. Then

> **`U > 0`  ⟺  `q* ∉ Q(g)`.**

Both directions are immediate. If `q* ∈ Q(g)` then `q* ∈ Q(f∧g)`, so
`vᵀq* ≤ h(f∧g;v) ≤ h(f;v) = vᵀq*` and `U = 0`. If `q* ∉ Q(g)` then the `f∧g`
maximiser is some `q ≠ q*` in `Q(f)`, and uniqueness of `q*` gives `vᵀq < vᵀq*`,
so `U > 0` strictly. Uniqueness is what makes the second case strict, which is
why `v` must be *interior* to the cone — the direction `faces` returns.

Three consequences.

- **The sign of `U` costs a matrix-vector product**, `K_g q* ⪯ b_g`, instead of
  three support solves. The *magnitude* still needs one solve on the meet.
- **The `U > 0` region of direction space is characterised, not searched**: it is
  the union of the normal cones of the `g`-infeasible vertices of `Q(f)`. Every
  such direction arises this way, since a `v` whose `f`-optimal face lies wholly
  inside `Q(g)` gives `U = 0`.
- **Globally, `U > 0` for some direction iff `Q(f) ⊄ Q(g)`** — a polytope is the
  hull of its vertices, so if `Q(g)` cuts anything off `Q(f)` it cuts off a
  vertex.

The symmetric statements hold for `V` at the vertices of `Q(g)`.

**Open direction.** The feasibility reading suggests limits could be designed to
target `U > 0` directly, rather than positing a binding pattern and checking
afterwards. Not worked out here.

**Scale.** By the Upper Bound Theorem a polytope of dimension `n − 1` with `m`
facets has on the order of `m^⌊(n−1)/2⌋` vertices. **Contingencies are survivable;
buses are not** — adding contingencies grows `m` linearly, while adding buses
grows the exponent. Past a small `n` the *answer* is too large, not the
computation, and the well-posed question becomes a sample of realized directions
rather than an enumeration. Enumeration's role is **completeness at small `n`**:
it establishes that both modes coexist across essentially the whole regime space
without tuning, which licenses the sampling design used where enumeration is
impossible.

---

## 6. Construction (methods)

**Positing a binding pattern is positing `y`.** Every proposition except
`prop:support` holds at an arbitrary certificate `y ⪰ 0`, so a case needs no bid
data: choose the rows that should bind and solve for limits that make them bind
exactly. Maximising the margin at every unbound row makes `J*` **equal** the
chosen pattern by construction, rather than something to filter for afterwards.

**One `b`, several patterns — not a fresh `b` per pattern.** A fresh `b` makes
each cell a different network engineered to produce its own answer. Which
patterns can coexist on one limit vector is a real structural fact about the
network, and it is what the multi-pattern design solves for.

**One limit per element across contingencies removes the ratings confound.**
Imposing `b[base, e] = b[c, e]` means no *level* difference can hide between two
contingencies, so any cross-contingency block that appears is topology rather
than a ratings artifact. The cost is realism — real post-contingency ratings are
higher — and margin, since pinning a base row now pins its twin.

**A designed pattern is a claim about one direction only.** The design pins each
pattern at its own optimum for its own direction. The union of two patterns is
not itself a designed pattern: `v = Kᵀ(1_A + 1_B)` exposes a third face where
`J*` is neither, and structure built for `A` is simply absent there.
