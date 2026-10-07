# Validation

What each case shows, and what is open. Engineering notes are in `CLAUDE.md`.

**Notation** (journal draft). `Q^FTR`, `Q^DAM`, intersection `Q^∩` (rows of both,
stacked). `v` a congestion direction. `U = h_FTR − h_∩`, `V = h_DAM − h_∩`. `J*`
the priced set, `𝒮 = ker K̄_{J*}ᵀ` the shift space, `B` a block, `N_Q(q)` the
normal cone of `Q` at `q`. Paper hooks cite TeX labels.

| case | shows |
|---|---|
| 5-node (`notebooks/explore_texas5.py`) | attribution structure exists, and why |
| RTS-GMLC | how often, at realized directions with real N-1 and bids |
| ERCOT | scale, and the funding-gap decomposition on market data |

The 3-node is the code oracle only: below 4 dimensions any three rows are
dependent, so block structure reports the dimension, not the network.

---

## 1. 5-node: attribution structure

Cases are built by positing a binding pattern, which is positing `y`: every
proposition but `prop:cr_support` holds at any `y ⪰ 0`, so no bids are needed.
`solve_limit_design` makes the pattern bind and maximizes the margin at every
other row, so `J*` equals the pattern. One limit vector carries several patterns;
one limit per element across contingencies (`b[base,e] = b[c,e]`) makes every
cross-contingency block topology rather than ratings.

### 1.1 Types of attribution blocks

Rows share a block because a circulation links them, not because one certificate
prices them. The 5-node exhibits every type the paper names:

| type | pattern | rows | blocks | max block | `dim 𝒮` | shows |
|---|---|---|---|---|---|---|
| parallel elements | `parallel_wd` | WD1, WD2 | 1 | 2 | 1 | a block that is a representation artifact: merge the pair and it is gone |
| co-priced, independent | `no_loop` | WN, SH | **2** | **1** | **0** | priced together, separately attributable |
| one circulation | `outer_loop` | WN, NH, SH, WS | 1 | 4 | 1 | a loop binds four rows into one block |
| two blocks, one certificate | `two_blocks` | WN, ND, WD1, WD2, SD, DH, SH | 2 | 4 | 3 | a block is a union of overlapping circuits, not one circuit |
| cross-contingency, LODF triple | `{base:o, base:e, c_o:e}` | 3 | 1 | 3 | 1 | the outage-transfer identity `k_e^{c_o} = k_e + L·k_o` is the circulation; no cycle in the graph behind it |
| cross-contingency, spanning circuit | non-LODF, size 3–4 | — | 1 | 3–4 | 1 | same shape as the triple, different economics (1.2) |

Reading the columns: blocks partition `J*`, so `n_blocks = n_priced` is the fully
identified case. Ambiguity is `dim 𝒮`, equivalently `max block > 1`. A block's
corank is usually below `|B| − 1` (a 9-row block here has corank 5); per-block
coranks sum to `dim 𝒮`.

Hooks: `thm:blocks`(iii), the three bullets of §5.1, `fig_texas5_blocks`.

### 1.2 Priced but costless

FTR enforces the base case, DAM adds one outage, and a circuit
`S = S_b ∪ S_c` (base rows, post-outage rows) binds at the intersection. Then

```
U = 0   ⟺   v ∈ cone{kᵢ : i ∈ S_b} + span{1}
```

A post-outage row can bind, be priced, and cut nothing off `Q^FTR`: the
dependence that puts it in a block also puts its hyperplane through the corner
the base rows already form. This is Farkas, so it is the criterion, not a
proxy. A proper subset of a circuit is independent, so the weights solving
`v = K_{S_b}ᵀw + t·1` are unique: least squares, then read the residual (span)
and the signs (cone). **Cone, not span.** A span test is blind to negative
weights and calls live designs dead.

In general, with `q∩` in the relative interior of the intersection's optimal
face, `U = 0` iff `v` lies in the cone of the FTR rows priced by the intersection,
and symmetrically for `V`. That is `cor:binding` in cone form, and the form the
paper should state.

- **Over 1,683 realizable spanning-circuit designs** the criterion partitions `U`
  exactly: 1,141 out of cone, all `U > 0`; 542 in cone, all `U = 0`. The span test
  is wrong on 173, 165 of them with one post-outage row.
- **Every LODF triple is in cone.** With sides `s_o, s_e, s_c ∈ {±1}`,
  `w_e = 1 + s_c s_e ∈ {0, 2}` and `w_o = 1 + s_c s_o L`, so the test is
  `|L| ≤ 1` on every side assignment. `|L| > 1` is the near-radial case excluded
  with bridges. All 70 triples × 8 sides pass; `|L| ∈ [0.043, 0.769]`.
- **Escaping the cone is topology, not sides.** A spanning circuit can carry
  `U > 0` only if it is not an LODF triple: of 133 non-LODF size-3 designs, 75
  escape and 58 do not. No proxy is exact (`|S_c| ≥ 2` is the best, 976 of 1,009),
  so run the test. The block need not be broken: out-of-cone designs realize `J*`
  as one size-3 cross-contingency block with `dim 𝒮 = 1` and `U > 0` at once.
- **Open: N-2.** An N-2 row `(o₁,o₂):e` combines base rows `e, o₁, o₂`, a
  4-row analogue of the triple. Whether a bound like `|L| ≤ 1` keeps it in cone
  decides if targeted N-2 coverage is priced but free, and is the N-2 selection
  rule for the RTS pair (ii).

Hook: `cor:binding`.

### 1.3 Identification of block shares

A block's share is read at an intersection optimum `q∩` and is affine in it,

```
share(B) = Σ_{i∈B} μᵢ bᵢ − wᵀq∩ ,      w = Σ_{i∈B} μᵢ kᵢ ,
```

so it is one number iff `w ∈ span{1} + row(K̄∩_{J∩*})` and an interval otherwise
(`thm:failure_blocks`(ii)). Vertices satisfy this vacuously; facet normals are
where it bites, because there the optimal face has dimension `n − |S|` for the
tight circuit `S` and `q∩` is a genuine choice.

- The span test (`primal_invariant`) and the two face LPs (`block_share_range`)
  agree on all 88 blocks probed at facet normals. Widths separate by four orders
  of magnitude (identified ≤ 5e-4, unidentified ≥ 8.6), so the span test alone
  detects. The threshold must scale with `h`.
- **Where it fails.** 28 of 88: 24 in `U` at facets of contingencies the FTR
  model omits, and 4 in `V` at the `base:SH` facet, where the DAM has the row but
  looser than the FTR's derated copy, so its certificate prices `base:SD` and
  `DH:SH` instead. Rule to state: a share is unidentified where the direction
  points along a row the intersection binds but the model does not price, either
  because it lacks the row (coverage) or holds it looser (level). Symmetric in
  the two models.
- RTS witness (random 26-contingency pair, interval 5235): `U = 21.8` on the
  parallel pair CA-1/CB-1, FTR lacking their mutual contingencies, each block's
  share ranging over `[0, 21.8]`.
- **Open.** `width / block value`: whether attribution degrades gracefully or
  collapses when unidentified.

### 1.4 Both failure modes at once

At any `q∩` attaining `h_∩(v)`, `U = 0 ⟺ v ∈ N_FTR(q∩)` and
`V = 0 ⟺ v ∈ N_DAM(q∩)`, since a mode vanishes exactly when `q∩` is already
optimal for that model. For polyhedra `N_∩(q∩) = N_FTR(q∩) + N_DAM(q∩)`, so

> both modes are positive iff `v ∈ (N_FTR + N_DAM) \ (N_FTR ∪ N_DAM)`:
> the direction needs generators from both models and lies in neither cone alone.

This holds on every face of `Q^∩`, not only vertices. Vertices are where the
cone is full-dimensional, so the set of such directions has positive measure,
and where the regime map is complete: the vertices of `Q^∩`, each with one
interior direction (`faces`), list every binding pattern the pair admits with
no certificate posited. A realized DAM direction is never generic: `v = Kᵀy*`
lies in the normal cone of the DAM face it exposes, at a facet normal when one
row binds, so at realized directions `V = 0` iff that DAM optimal face meets
`Q^FTR`, and `U = 0` iff some FTR maximizer is DAM-feasible (`prop:vanish`).

- Current design: 88 vertices, 78 with both modes, 10 V-only, 0 U-only, with no
  derate tuned. (The 3-node needed a hand-picked α to show both.)
- The 10 exceptions should be vertices whose direction lies in `N_FTR` alone:
  either a tight set from the FTR only, or DAM rows tight but inside the FTR
  rows' cone (1.2). To check.
- Scale: `~m^⌊(n−1)/2⌋` vertices for `m` facets. Contingencies grow `m`; buses
  grow the exponent. Enumeration is for completeness at small `n`; at RTS the
  sample is realized directions.

Hook: §4, "a small set of special directions".

---

## 2. RTS-GMLC: how often

DAM baseline `D₀` = base at `Cont Rating` + N-1 at `LTE Rating`, one physical
network. Runs end to end; `gap_summary` ~20 s, `block_table` with target ~8 s on
a 26-contingency pair. Merging identical contingency rows in the intersection is
the first speed lever.

| pair | FTR | DAM | expected |
|---|---|---|---|
| (i) derate | `α · D₀` | `D₀` | `U = 0`, `V = (1 − α) h_DAM` exactly (`ex:derate`); sweep `α` |
| (ii) targeted N-2 | `D₀` + selected `(o, e)` | `D₀` | `V` only; select `o` by `Perm OutRate × Duration`, `e` by priced frequency × `\|LODF\|` or by the cone test |
| (iii) outage in DAM base | `D₀` | `(o,)` at Cont + `(o, e)` at LTE | both modes |

If (ii)'s N-2 set contains the `o` realized in (iii), the FTR model anticipated
the outage and `U` should shrink.

| question | instrument | from |
|---|---|---|
| block-type census: `n_blocks` vs `n_priced`, block size, `dim 𝒮`, share of priced sets with a cross-contingency block | `block_table`, `attribution_blocks` | 1.1 |
| of those, share in cone: priced but costless; LODF distribution and mass near `\|L\| = 1` | cone test vs `U`, PTDF | 1.2 |
| share of blocks unidentified; `width / block value` | `block_table` | 1.3 |
| co-occurrence of both modes at realized directions | `gap_summary` over `clear_dam` | 1.4 |
| `U`, `V` by pair (i)–(iii); report card at representative intervals | `gap_summary`, `block_table` | `rem:reporting` |

---

## 3. ERCOT: scale and the funding gap

The CRR and DAM models are built separately and compared on shared nodes, so
differing PTDFs are not a question. What ERCOT adds:

- **Scale.** Merge identical contingency rows; generate post-contingency rows
  from base PTDF + LODF instead of storing them; constraint generation over a
  working set. Per-contingency storage in `NetworkModel` is the seam.
- **The decomposition on market data** (`thm:decomposition`). Auction
  performance, temporal aggregation and `Σ Δ_t` over a contract period, each as
  percentage points of the funding rate (`rem:monitor`); then `U_t`, `V_t` and
  the report card at the intervals that drive the total.
- **The uniform derate priced.** ERCOT's schedule is `ex:derate`, so
  `V_t = (1 − α) h_DAM(v_t)` interval by interval: the hedge value the schedule
  forgoes, measured.
- **Market realism** (`rem:realism`): as-cleared limits under penalty pricing,
  options in the settled portfolio, settlement-point node mapping. Each is a
  data-handling decision to record.
- **The certificate set.** ERCOT publishes shadow prices; where the dispatch is
  degenerate they are one point of `Y_KKT`, and the direction set it induces is
  the object the design section takes.

---

## 4. Ex-ante design

What §6 leans on, and the shape of the problem.

- **Design space.** Coverage (which contingencies the FTR enforces) and levels
  (`b` on shared rows). `c ∈ C_DAM \ C_FTR` or `b_FTR > b_DAM` feeds `U`;
  `c ∈ C_FTR \ C_DAM` or `b_FTR < b_DAM` feeds `V`. A disagreement costs only
  where the loser prices the row and `q∩` leaves it slack.
- **Limit-free structure** (`prop:limit_free`). `h_FTR(b)(v) = min_{μ∈Λ(v)} bᵀμ`
  is concave piecewise-linear in `b`; each certificate is a cut, exact on the
  cell of `b`-space where its priced set persists, and its slack elsewhere is a
  duality gap. A uniform derate never leaves its cell (same priced sets, blocks
  and certificates; `V ∝ h_DAM`); a selective derate crosses cells and breaks
  exactly the block it cuts. Coverage is screened before limits with the cone
  test (1.2).
- **The frontier.** Scenarios `(Q^DAM_ω, v_ω)`. Minimizing `Σ π_ω V_ω(b)` is one
  LP in `(b, q_ω)`, since `h_∩(b)(v) = max{vᵀq : K^FTR q ≤ b, q ∈ Q^DAM}`.
  `U_ω(b) ≤ τ` is exactly `∃ μ ∈ Λ^FTR(v_ω), q ∈ Q^DAM_ω : K^FTR q ≤ b,
  bᵀμ − v_ωᵀq ≤ τ`, bilinear only in `bᵀμ`. Fix `μ` and it is an LP; fix `b`
  and `μ` is a support solve. On the 5-node the regime map is a complete scenario
  set, so the frontier there is exact and compares directly to the uniform
  derate at equal exposure.
- **Adequacy is linear within a cell.** At a vertex `q*(b)` of `Q^FTR(b)`,
  `U > 0` on its cone iff `q* ∉ Q^DAM`; `q*(b)` is linear in `b` while its tight
  set stays a vertex, so adequacy on the whole cone is a finite set of linear
  inequalities in `b`. Over the regime map the adequate set is a union of
  polyhedra, not convex.
- **Open.** The certificate set `Y_KKT(g; ω)` and its worst case (`U` is a
  difference of convex functions of `v`, so not at a vertex in general); the
  N-2 cone check; convergence of the alternating scheme across cell boundaries.
