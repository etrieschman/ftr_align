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

**(d) Circuit size determines the face it exposes.** If exactly the rows of a
circuit `S` are tight, the optimal face is `{q : k̄ᵢᵀq = bᵢ, i ∈ S}`, of dimension

```
(n − 1) − rank(k̄_S) = (n − 1) − (|S| − 1) = n − |S|
```

A maximal circuit (`|S| = n`) exposes a vertex; the smallest (`|S| = 2`) exposes
a face of dimension `n − 2`. **Small circuits are the ones that leave room to
move**, which is what §4 needs.

**The slack bus is a labelling convention.** PTDF rows under two slack
conventions differ by a multiple of `1ᵀ`, which annihilates balanced injections.
Changing it shifts `d` by a constant vector and leaves `Q(b)`, `h` and `μ`
untouched. Choosing the slack to suit a figure's axes is free.

---

## 2. When attribution rises above a single constraint

**The block condition.** Rows `S` share an attribution block iff some `z`
supported on `S` satisfies **both**

```
(a)  Σ_{i∈S} zᵢ k̄ᵢ = 0        the trade moves no flow
(b)  Σ_{i∈S} zᵢ bᵢ = 0        and costs no value
```

**(a) is limit-free.** It is pure linear algebra on the stacked PTDF, so it can
be checked before any limits are chosen. A row set failing it can never be a
block, at any `b`. This is the cheap screen, and it is what makes "which sets
*could* be blocks" a finite enumeration rather than a search.

**(b) is automatic whenever the whole circuit binds.** If every row of `S` is
tight at the optimum `q`, then

```
Σ zᵢ bᵢ = Σ zᵢ (Kq)ᵢ = (Σ zᵢ kᵢ)ᵀ q = 0
```

by (a). So **a circuit that binds is a block** — condition (b) is a consequence,
not an extra requirement. Nothing needs to be imposed on the limits.

**A nondegenerate vertex has only singleton blocks.** There `J*` holds `n − 1`
independent rows, so `Σ zᵢ k̄ᵢ = 0` forces `z = 0`. Non-trivial attribution
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
Scaling every limit by `α` leaves (b) satisfied: `Σ zᵢ(α bᵢ) = α · 0 = 0`.
Derating only some rows of a circuit breaks exactly the block it was built to
make.

---

## 3. The gap between the floor and the failure mode is a duality gap

**The claim in one line: `loss − floor` is the duality gap of the model's
certificate against the target.** So the floor is tight exactly when that
certificate is still optimal for the target, and slack by however much it is not.

**Why the floor bounds the mode.** With `μ` the model's certificate,
`h(model) = bᵀμ` by strong duality. The target shares `K` and `d`, so `μ` is
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
holds, but *vacuously*. On a face of positive dimension you can move by any `v`
with `1ᵀv = 0` and `K_{J*(f∧g)} v = 0`, and the share is invariant to all of them
exactly when

```
w ∈ span{1} + row(K_{J*(f∧g)})
```

`False` means the share is genuinely a **different number at different optima**.
It is an interval, not a value.

**(iv) Where it fails, and why that is structural.** Point `d` along a single
row's normal, `d = kᵢ`. The maximiser set is then the whole facet, and
`J*(f ∧ g)` is essentially `{i}`, so the test subspace is only `span{1, kᵢ}` —
two dimensions inside `Rⁿ`.

Now the asymmetry between the modes, which is the interesting part:

- **V** (`model = g`): `g` enforces row `i`, so at `d = kᵢ` its certificate
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

**Scale.** By the Upper Bound Theorem a polytope of dimension `n − 1` with `m`
facets has on the order of `m^⌊(n−1)/2⌋` vertices. **Contingencies are survivable;
buses are not** — adding contingencies grows `m` linearly, while adding buses
grows the exponent. Past a small `n` the *answer* is too large, not the
computation, and the well-posed question becomes a sample of realized directions
rather than an enumeration.

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
not itself a designed pattern: `d = Kᵀ(1_A + 1_B)` exposes a third face where
`J*` is neither, and structure built for `A` is simply absent there.
