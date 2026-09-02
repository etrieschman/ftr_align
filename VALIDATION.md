# Validation plan

Claims and their mechanisms live in `LEARNINGS.md`; measurements live in
`notebooks/`. This file maps between them: which case shows which claim, with
what instrument, and what is still open.

Each rung shows something the rung below structurally cannot.

| rung | role | what only it can do |
|---|---|---|
| **5-node** | attribution *structure* | blocks need ≥ 4 dimensions to be non-trivial |
| **RTS-GMLC** | *frequency* | do these structures occur in a real network, how often |
| **ERCOT** | *scale*, Assumption 1 | different PTDFs between the FTR and DAM models |

**No 3-node rung.** The toy stays in the codebase and its tests still run — it
reproduces results derived independently by hand, which is what makes the
implementation trustworthy. That is code verification, not an argument the note
makes, and its geometry can be drawn abstractly.

---

## 1. The 5-node: attribution structure

### 1.1 Circulation, not co-pricing, is the criterion

**Claim.** Rows are attributed together when a circulation links them, not when
one certificate happens to price them.

Precisely: a **circuit** is a minimal dependent set of the stacked rows `K̄`, and
a circuit whose rows all bind lies inside a single block (`LEARNINGS` §2). That
is *sufficient*, not necessary. A **block** is a connected component of the
matroid on `J*` — the transitive closure of "shares a circuit with" — so a block
is in general a **union of overlapping circuits**, not one circuit. `two_blocks`
below is the witness: corank 3 spread over 2 blocks means at least one block
carries more than one independent circulation.

Circuits of the stacked system coincide with graph cycles only in the base case.
Across a contingency boundary there is no cycle behind them (§1.2).

**Why this rung.** Below 4 dimensions block structure reports the dimension, not
the network: a 3-bus balanced subspace is 2-D, so any three rows are dependent
and a non-singleton block is forced rather than discovered. And a block among
*parallel* elements — the only kind a small case reliably offers — dissolves
under equivalent-circuit reduction.

**Shown by** four binding patterns on one shared limit vector (`explore_texas5`,
`PATTERN_ELEMENTS`), all base-case:

| pattern | rows | priced | blocks | max block | `dim ker C` | shows |
|---|---|---|---|---|---|---|
| `parallel_wd` | WD1, WD2 | 2 | 1 | 2 | 1 | parallel elements: the degenerate block |
| `no_loop` | WN, SH | 2 | **2** | **1** | **0** | priced together, no circulation, separately attributable |
| `outer_loop` | WN, NH, SH, WS | 4 | 1 | 4 | 1 | one circulation binds four rows |
| `two_blocks` | WN, ND, WD1, WD2, SD, DH, SH | 7 | 2 | 4 | 3 | two blocks on one certificate; a block ≠ a circuit |

Structure is the claim; `h` depends on the design solved that run.

**`no_loop` is the result.** Two constraints priced by the same certificate,
split into two singleton blocks with an empty trade space, because no circulation
joins them. **Co-pricing is not a reason to aggregate.** Unreachable at 3 nodes:
in 2-D two priced rows at a vertex already span the space.

**`parallel_wd` is why the rung exists.** A real block, but an artifact of
representation — merge the parallel pair and it disappears. `outer_loop` cannot
be merged away.

### 1.2 Contingencies multiply dependencies without adding geometry

**Claim.** A contingency adds rows to a space already saturated at rank `n − 1`,
so every added row is a new dependency. Most of the resulting circulations mix
base with post-outage rows: **cross-contingency blocks are generic**, and the
base graph's cycle space cannot see any of them.

**Why it matters.** Pure counting, so it transfers to RTS and ERCOT unchanged.
This is the bridge to rung 2.

**The count** (texas5: `E = 9`, `n = 5`, balanced dimension 4):

| system | usable rows | rank | dependencies |
|---|---|---|---|
| base only | 9 | 4 | **5** |
| base + one outage | 17 | 4 | **13** |

The base's 5 is the graph cycle space `E − V + 1`, with KVL supplying the
dependencies. One outage takes it to 13, of which only 5 are spanned by base-only
cycles: **at least 8 dependency dimensions must involve post-contingency rows.**

**Shown by** `circuits_for` over every outage, with `spans` counting circuits
that touch both the base and a contingency.

### 1.3 A block can be fully realized and still be worth nothing

**Claim.** A cross-contingency block can bind exactly as designed — `J*` equal to
the pattern, one block, `dim ker C = 1` — and contribute **zero** to `U`.
Underfunding is a difference of support values, not a count of binding rows.

**The criterion** (`LEARNINGS` §2), with `f` base-only and the circuit split into
base rows `S_b` and contingency rows `S_c`:

```
U = 0   ⟺   v ∈ cone{kᵢ : i ∈ S_b} + span{1}
```

This is Farkas, so it is the condition rather than a proxy: `U > 0` needs a `Δq`
with `kᵢᵀΔq ≤ 0` on the tight base rows and `vᵀΔq > 0`, and no such `Δq` exists
exactly when `v` lies in their cone. Pinning a whole circuit forces `Σ dᵢbᵢ = 0`,
which puts the contingency hyperplane through the corner the base rows already
form — it touches `Q(f)` and slices off nothing.

**The sharp case.** The smallest cross-contingency circuit is the **LODF triple**
`{base:o, base:e, c_o:e}`, whose dependence *is* the outage-transfer identity
`k_e^{c_o} = k_e + L(e,o)·k_o`. Its weights are `w_e = 1 + s_c s_e ∈ {0,2}` and
`w_o = 1 + s_c s_o L`, so the test reduces to `|L| ≤ 1` **independent of sides**:
every LODF triple is worth zero to `U`. A single-loop network has `|LODF| = 1`
identically, so a 3-node has no interior LODFs to show.

**Shown by** two spanning circuits at the same designed limits — an LODF triple
(in-cone, `U = 0`) beside a non-LODF spanning circuit (out-of-cone, `U > 0`).
Same size, same `dim ker C`, both cross-contingency, both with `J*` equal to
their pattern; only the economics differ. Over 1683 realizable spanning-circuit
designs the criterion partitions `U` exactly, with no disagreements.

### 1.4 Both failure modes at a meet vertex

**Claim.** At a vertex `q*` of `Q(f∧g)` exposed by an interior direction `v`:

```
U = 0  ⟺  v ∈ N_f(q*)          V = 0  ⟺  v ∈ N_g(q*)
```

and since `N_{f∧g}(q*) = N_f(q*) + N_g(q*)` for polyhedra,

> both modes are strictly positive exactly when
> `v ∈ (N_f + N_g) \ (N_f ∪ N_g)` — the direction needs generators from both
> models and lies in neither cone alone.

**Why it is generic.** The sum of two cones fills the wedge between them while
the union is only the two cones. And the exposing direction is `Σ_{i tight} kᵢ`,
a positive combination of every tight row, so it takes generators from both
models by construction.

**When it cancels.** Only at a vertex whose tight set belongs to one model alone.
The other's normal cone is trivial, the sum collapses, `v` sits inside it, and
that model's mode is exactly zero. A failure is a classification — an inherited
vertex — not a statistic.

**Shown by** `faces(f ∧ g)` plus `gap_summary` at each exposing direction: both
modes live at 58 of 60. The prediction to check the sweep against is that the two
exceptions have single-model tight sets.

### 1.5 The sign of `U` is a feasibility test

**Claim.** At a vertex `q*` of `Q(f)` with `v` interior to its normal cone:

> **`U > 0` ⟺ `q* ∉ Q(g)`** — the FTR model's own optimal dispatch is not
> DAM-feasible.

**Why.** Interiority makes `q*` the unique `f`-maximizer, so `h(f;v) = vᵀq*`. If
`q* ∈ Q(g)` then `q* ∈ Q(f∧g)` and `vᵀq* ≤ h(f∧g;v) ≤ h(f;v) = vᵀq*`, so `U = 0`.
Otherwise the meet's maximizer is some other `q ∈ Q(f)`, and uniqueness forces
`vᵀq < vᵀq*`, so `U > 0` strictly.

**Consequences.** The sign of `U` costs a matrix-vector product rather than three
support solves; the magnitude still needs one. The `U > 0` region of direction
space is characterised — the union of normal cones of the `g`-infeasible vertices
of `Q(f)` — rather than searched. Globally, `U > 0` for some direction iff
`Q(f) ⊄ Q(g)`.

**Open.** This suggests designing limits to target `U > 0` directly rather than
positing a binding pattern and checking afterwards. Logged, not worked out.

### 1.6 Where block attribution stops being a number

**Claim.** Blocks are the unit at which attribution is well posed — settled
elsewhere. What remains is the residual failure: even at block granularity, a
block's *share* of a failure mode can be an interval rather than a value.

**Why.** The share is read at a maximiser `q` of the target and is affine in it:

```
share(B) = const − wᵀq ,     w = Σ_{i∈B} μᵢ kᵢ
```

so it is a number only when `wᵀq` is constant on the target's optimal face, i.e.
`w ∈ span{1} + row(K_{J*(f∧g)})`. A vertex satisfies this vacuously. **Facet
normals** are where it bites, because there the optimal face is
positive-dimensional and `q` is a genuine choice.

**The condition is model-symmetric.** Nothing in it privileges `U` or `V`; `w` is
built from whichever model's certificate is in play. Which mode shows a failure
is decided by which model is blind to the row the direction points along. On this
pair that is always `f`, so every observed failure lands in `U`.

**Shown by** 20 of 94 probed blocks failing, all in `U`, all at the normal of a
contingency row `f` cannot price. Two independent computations agree on all 94:
`primal_invariant` is a span test, `block_share_range` is two LPs over the same
face.

**The separation is economic, not numerical.** Identified widths top out at
`6.7e-4`; unidentified ones start at `48.1`. So the multiplicity is real rather
than a tolerance artifact — and the cheap span test is therefore sufficient as
the detector, with the LPs corroborating rather than instrumenting.

**Open, for rung 2.** The failure may be small relative to what the block is
worth, in which case attribution degrades gracefully rather than collapsing. The
ratio `width / block value` is question 12 below, and it decides whether
unidentified blocks are a caveat or a problem.

---

## 2. RTS-GMLC: frequencies

Rung 1 says these structures exist and why. Rung 2 asks how often, on a 73-bus
network with real N-1, heat-rate bids and limits. Several are *predictions*
computable from `K` before they are measured.

**Prerequisite.** RTS has a DAM instance but no FTR/DAM *model pair*. One must be
defined — most naturally an FTR model enforcing a reduced contingency set —
before anything below runs.

| # | question | instrument | predicted by |
|---|---|---|---|
| 1 | share of dependency dimensions involving contingency rows | rank counts on `K` | §1.2 |
| 2 | circuit-size distribution; share spanning the contingency boundary | `circuits_for` | §1.2 |
| 3 | how often attribution is ambiguous — `n_blocks` vs `n_priced` | `block_table` | §1.1 |
| 4 | block-size and `dim ker C` distributions | `block_table` | §1.1 |
| 5 | share of realized `J*` holding a cross-contingency block | `attribution_blocks` + `spans` | §1.2 |
| 6 | of those, the share in-cone — **binding but unfunded** | cone test vs `U` | §1.3 |
| 7 | LODF distribution, and mass near `\|L\| = 1` | PTDF rows | §1.3 |
| 8 | how often both modes co-occur at *realized* directions | `gap_summary` over `clear_dam` | §1.4 (58/60 enumerated) |
| 9 | how often block shares fail to identify | `primal_invariant`, `block_share_range` | §1.6 (20/94) |
| 10 | floor-ratio distribution — a gauge, strictly inside `(0,1)`? | `gap_summary` floors | §3 of `LEARNINGS` |
| 11 | merchandising surplus `= h(g;y*)` on a real clearing | `clear_dam` + `SupportProblem` | first realistic test of Prop 1 |
| 12 | **`width / block value`** for unidentified blocks | `block_share_range` vs `value` | §1.6 — open |

**Sampling replaces enumeration, and asks the better question.** `faces` is
intractable at `n = 73`, but realized directions from actual clearings are what
matters economically — a market visits a small, structured subset of the normal
fan. Rung 1's complete enumeration is what licenses the sampling design: it
establishes that both modes coexist across essentially the whole regime space, so
a frequency measured on realized directions estimates something real.

---

## 3. ERCOT: scale, and Assumption 1

**Scale.** Everything above at production size. The levers are in place —
`J_star` as one CLARABEL solve where only the support is needed, and candidate
restriction plus a compiled Parameter-objective LP for the robust bounds. Dense
`K` stays correct; the scale lever is active-set / column generation, not sparse
storage.

**Assumption 1 breaks.** Everything above assumes the FTR and DAM models share
PTDFs. ERCOT is the case where `K_f ≠ K_g`, and the code raises
`NotImplementedError` there rather than returning a wrong number. Two things to
establish:

- **`f ∧ g` needs the stack.** The intersection is always the polytope of
  `[K_f; K_g] q ⪯ [f; g]`. Under Assumption 1 that stack has identical row pairs
  and collapses exactly to `min(fᵢ, gᵢ)` after `align`, which is why the
  elementwise min is correct today.
- **`v` was built to survive this.** Support is parametrised by the node-space
  direction `v = Kᵀy*`, not a row-space certificate. Under `K_f ≠ K_g`, `v` is
  still a node-space vector and both support problems stay well-posed, so values
  and the gap need no alignment. Carrying `y` instead would not survive — it is
  meaningless without the model that indexes it. This is an architectural claim
  ERCOT can make good on, and the reason the extension is a fallback rather than
  a rewrite.

---

## Register

✅ established · 🔨 instrument exists, not yet run · ⬜ open

| claim | rung | instrument | status |
|---|---|---|---|
| circulation, not co-pricing; four-pattern taxonomy | 5-node | `explore_texas5`, `block_table` | ✅ |
| co-priced ≠ aggregable (`no_loop`) | 5-node | `explore_texas5` | ✅ |
| contingencies multiply dependencies (§1.2 count) | 5-node | `circuits_for` + rank | ✅ |
| binding ≠ funded; cone criterion; LODF triples | 5-node | `in_base_cone` vs `U` | ✅ 1683/1683 |
| both modes at meet vertices; normal-cone sum | 5-node | `faces` + `gap_summary` | ✅ 58/60 |
| inherited-vertex classification of the 2 exceptions | 5-node | tight-set inspection | 🔨 |
| `U > 0` ⟺ `f`-vertex is `g`-infeasible | 5-node | `faces(f)` + feasibility | 🔨 |
| block shares can fail; span test suffices to detect | 5-node | `primal_invariant` vs `block_share_range` | ✅ 0/94 disagree |
| floor is a gauge, not a switch | 5-node | `gap_summary` floors | ✅ 56/60 interior |
| block total invariant over the dual face | codebase (toy) | `test_block_total_is_face_invariant` | ✅ code-verified |
| RTS FTR/DAM model pair | RTS | — | ⬜ prerequisite |
| frequency battery (12 questions) | RTS | see §2 | ⬜ |
| `width / block value` for unidentified blocks | RTS | `block_share_range` | ⬜ |
| scale | ERCOT | — | ⬜ |
| Assumption 1 / stack fallback | ERCOT | `meet` guard | ⬜ |
