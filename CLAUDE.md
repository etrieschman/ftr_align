# ftr_align — working notes for Claude

Research codebase computing/analyzing **FTR–DAM structural misalignment** via
support-function geometry, from Erich's papers (PowerUp conference paper →
journal draft, plus two technical memos: attribution blocks, misalignment
attribution). **Follow the journal draft's notation** where sources differ.
Goal: one core that scales toy (3-node oracle) → RTS-GMLC → ERCOT without
rewriting per network.

## Notation (journal draft — authoritative)

| symbol | meaning | code |
|---|---|---|
| `f` | **FTR/SFT** network model (limit vector) | first of a model pair |
| `g` | **DAM** network model | second of a model pair |
| `Δ(f,g;y)` | `h(f;y) − h(g;y)`; `>0` underfunding exposure, `<0` lost hedge value | — |
| `U`, `V` | failure modes; computed **only** in `metrics.gap_summary` | `gap_summary` |
| `A` | node-branch incidence | `PhysicalNetwork.A` |
| `H_c`, `H` | PTDF under `c`; stacked over contingencies | `.ptdf(key)`, `NetworkModel.H` |
| `K = [H; −H]` | stacked constraint matrix | `NetworkModel.K` |
| `y`, `d = Kᵀy` | price certificate; node-space direction | `DamResult.y`, `.direction` |
| `Λ(y)`, `Λ*(b;y)` | dual-feasible set, optimal dual face | `Lambda`, `Lambda_star` |
| `J*(b;y)` | dual-optimal support | `J_star` (one CLARABEL solve) |
| `D(b;y) = ker C(b;y)` | trade space | `trade_space`, `trade_matrix` |
| `W_{J_r}` | block total | `block_totals`; column `value` |
| `U_B` | block's share of a failure mode | column `loss` (**not** `U_B` — see below) |
| `f ∧ g` | intersection model | `meet` |
| `U`, `V` | failure modes | `failure_modes` |
| `U^(S)` | repair value | `repair_value` |

**Naming convention:** capitalized functions (`Lambda`, `Lambda_star`) are
*assembly* — they return cvxpy constraint lists and never solve; lowercase
functions solve. Model pairs are always ordered `(f, g)`, matching `Δ(f,g;y)`;
`MODELS[...]` and `REDUNDANT_MODELS[...]` follow this.

## Locked design decisions (do not relitigate — settled over a long planning pass)

- **One network solve is the primitive**, not a DAM-vs-FTR pair. A `NetworkModel`
  owns its geometry: a `PhysicalNetwork` + a tuple of `Contingency` (each carrying
  the line limits enforced under it). `NetworkModel.build(net, contingencies)`
  assembles `K = [H; −H]` and the stacked limit vector `b`. There is **no separate
  `StackedSystem`** — it was folded into `NetworkModel`.
- **Support is parametrized by a node-space direction `d ∈ Rⁿ`**, not a row-space
  certificate. `SupportProblem(model, direction)`; `h(b;y) = max_{q∈Q(b)} dᵀq`.
  Every downstream object — `Λ`, `Λ*`, `J*`, blocks, floor, ceiling — depends on
  `y` **only** through `d = Kᵀy`, so `d` is what the code carries and the memos'
  `(b;y)` notation describes the same thing with the redundant fibre quotiented
  out. Because `d` lives in node space (shared by every model on the network),
  support **values and the gap need NO alignment** — each model solves on its own
  polytope with the same `d`. `clear_dam` returns the DAM certificate `y*` (over
  its own rows) **and** `direction = Kᵀ y*`.
  This survives Assumption 1 breaking: under `K_f ≠ K_g`, `d = K_gᵀy*` is still a
  node-space vector and both support problems are still well-posed. Passing `y`
  instead would *not* survive — it is meaningless without the model that indexes
  it. **Do not "carry `y` for later"; `d` is the interface that lasts.**
- **`align` puts two models on a common row index**, rebuilding both onto a union
  contingency set (unenforced contingencies added with `+inf` limits). Used for
  row-level cross-model comparison (lining up `μ_f`/`μ_g`, `differences`, joint
  blocks) and — legitimately — as model-level preprocessing for the intersection
  `f ∧ g`, which needs a common index by construction. Never needed before a plain
  support solve. Every comparative quantity aligns first, which is why `embed`
  was deleted: `μ` comes out on the common index natively. Valid only under
  **Assumption 1 (common PTDFs)**; ERCOT's different-PTDF case is flagged,
  separate, not yet handled.
- **`f ∧ g` is an intersection, not an elementwise min.** `Q(f) ∩ Q(g)` is always
  the polytope of `[K_f; K_g] q ⪯ [f; g]`. Under Assumption 1 that stack has
  identical row pairs and collapses *exactly* to `min(f_i, g_i)` after `align` —
  no row growth, no tolerance — which is why the min is the correct
  implementation today. Keep the name and semantics general so the stack fallback
  can fill in for ERCOT without a rewrite.
- `b` and per-row vectors (`μ`) are **co-indexed full-length vectors** over the
  model's rows. Unmonitored rows: `b = +inf`, `μ` pinned to 0.
  `active = np.isfinite(b)`.
- `SupportProblem` is **dual-form** (`min bᵀμ s.t. Kᵀμ + 1s = d, μ ≥ 0`) so the
  multipliers `μ` are variables. `.data` is a typed numpy bundle (`SupportData`:
  `K, b, direction` → `active` property). `.solve()` returns an immutable
  `SupportSolution` value.
- **`μ` stays stacked-nonneg**, never signed. `μ ⪰ 0` is what makes `Λ(y)` a cone
  and the LP a standard-form dual; a signed `μ` cannot express a row degenerate on
  both sides; and upper/lower are distinct columns of `C` that can land in
  different blocks. Collapse to signed net duals only for *reporting*, matching
  the paper's own convention — and that collapse lives with the paper table that
  wants it (`net_dual` in `notebooks/reproduce_conference.py`), not in the
  library.
- **Solver seam = one branch**: `solve(solver=...)` takes `None | dict | custom`.
  `None` or a `dict` runs `solve_support_cvxpy` (default backend) — the dict is
  splatted straight into `cp.Problem.solve` as its options, spelled exactly as
  CVXPY spells them (e.g. `{"solver": "CLARABEL", "verbose": True}`); no wrapper
  type. Anything else is a custom solver called as
  `solver.solve(problem) -> SupportSolution`. (Functions that build their own raw
  LPs — `robust_bounds`' face LPs, `clear_dam` — just splat the dict; `robust_bounds`
  pulls `opts = solver if isinstance(solver, dict) else {}` once.) Erich's future
  ex-ante solver will *orchestrate* CVXPY LP subproblems (bilinear, over a union of
  polyhedra), not replace CVXPY.
- Assembly functions (`Lambda`, `Lambda_star`, `network_constraints` in
  `solve.py`) accept numpy **or** cvxpy args and are reused by `solve`,
  `duality` and `clear_dam` (one definition each — no re-spelling). `clear_dam`
  builds its network block with `network_constraints` on the active rows, so the
  dual of that single row block **is** the certificate `y*` in stacked order — no
  per-contingency scatter.
- `K` is **dense numpy** — PTDF is structurally dense, so sparse storage wastes.
  The scale lever is active-set / column-generation, not sparse storage.

## Degeneracy convention

Toy patterns can have a **non-unique realized `y*`** ([Feng et al. 2012] LMP
non-uniqueness — the thing the robust framework targets). The paper's numbers
correspond to the **analytic-center certificate**, which an interior-point solver
(`CLARABEL`) produces; simplex (`HIGHS`) gives an equally-valid vertex dual with
a different split. The support *value* given `y*` is unique. **Clear with
`solver={"solver": "CLARABEL"}`** to reproduce paper numbers. In `duality.py` the
support threshold (`SUPPORT_TOL`) must exceed the face-construction leak
(`FACE_TOL`); `face_leak()` is the same relationship on the primal side.

## The working set

`ftr_align/__init__.py` re-exports **only** what you reach for to set a problem
up, solve it, and read the answer (~25 names). Everything else stays one import
deeper — `from ftr_align.duality import primal_face_range` — not because it is
private but because a flat namespace of seventy names is not an API you can hold
in your head. A typical session:

```python
f, g = toy.MODELS["mixed"]                # an (FTR, DAM) pair
d = clear_dam(g, scenario).direction      # y*, and d = K^T y*
gap_summary(f, g, d)                      # Delta, U, V, floors   (a dict)
block_table(g, d, meet(f, g))             # per block: W and U_B
constraint_table(f, g, d, mode="V")       # per constraint, underneath
```

## Layout

**Layering.** Each level depends only on those below it, and there is one hard
line: **computation returns numpy/plain values; only `metrics` labels rows and
emits DataFrames.** The computed objects are co-indexed vectors over the rows of
`K` — that indexing is the invariant the whole package leans on, and it survives
only if labels stay out of it. It also keeps the numerical layer testable against
the memos' propositions directly, with no table in between.

```
ftr_align/
  network.py    PTDF (compute_ptdf takes incidence `A` + optional per-element
                `tap`), is_connected (bridge/islanding guard), PhysicalNetwork
                (owns `A`, optional `tap`), Contingency (key + limits; pass one
                `upper` for symmetric), NetworkModel (owns K & b; `.H` is the
                stacked PTDF, the upper half of K), align, meet (f ^ g),
                with_limits, contingency_label/element_label
  solve.py      assembly fns (Lambda / Lambda_star / network_constraints),
                SupportData, SupportProblem, solve_support_cvxpy,
                SupportSolution, DamInstance, DamResult, clear_dam (returns y*
                and direction)
  duality.py    robust_bounds (lo/hi over the dual face; mu restricted to
                primal-binding candidates, single compiled Parameter-objective LP
                reused across rows, forced onto HiGHS internally -- the thin
                value-slab is infeasible for interior-point and must share the
                base solve's engine; bounds are solver-invariant), J_star
                (J*(b;y) from one CLARABEL solve via strict complementarity --
                CLARABEL required, ~50-130x cheaper than the face-LP loop),
                primal_face_range + face_leak, in_span, trade_matrix,
                trade_space (D=ker C), connected_blocks (matroid components via
                QR fundamental circuits), attribution_blocks (row indices per
                block), block_totals (W_{J_r})
  attribution.py  EVERY attributive fn takes a NESTED PAIR `(model, target)`
                -- what `model` loses on adopting `target` -- NOT `(f, g, mode)`.
                U is `(f, f^g)`, V is `(g, f^g)`; the mode IS the first argument,
                so the "attribute on the loser's blocks" rule is structural and
                unstatable wrongly.  `_nested_pair` aligns + enforces
                `Q(target) <= Q(model)` (raises on a crossing pair; a crossing
                pair gives a meaningless number, not a wrong one).  Any nested
                pair works, incl. a partially-repaired model.
                failure_modes (U/V/Delta -- still `(f, g)`, it computes both),
                repaired + repair_value (U^(S)), floor, ceiling, row_shares
                (cor:exact_split as a co-indexed vector -- `.sum()` is the
                failure mode, `[rows].sum()` is a block share), block_shares
                (prop:block_underfunding), primal_invariant +
                block_share_range, differences (level/coverage x U/V per
                prop:kinds)
  polytope.py   the V-representation of Q(b): free_basis / plane_system (a
                basis T with 1^T T = 0; reduced normals are just K T),
                polygon (exact 2-D outline by pairwise intersection -- no
                interior point, no Qhull), faces (vertices + tight rows + an
                exposing direction, general d via Qhull), is_bounded (ONE LP,
                by polar duality -- see below).  Guarded at MAX_NODES.
  viz.py        3-node figures only: frame_axes (limits/origin/names/title --
                call it FIRST, the drawing layers clip to the current limits) /
                draw_region / draw_constraints / draw_optimum / draw_halfplane
                (market bounds in PLOT coords -- generation caps, load >= 0; not
                network feasibility, no PTDF row) / label_axes, composable onto
                one axis.  No direction arrow -- fig_support_vi stays in TikZ.
  metrics.py    row_labels, gap_summary (one flat record per (model pair,
                direction) cell), constraint_table (per-constraint detail),
                support_summary (one-model counterpart of gap_summary),
                block_table (ONE per-block table, `(model, d, target=None)`;
                target optional -- see below).  Paper-shaped tables are NOT here -- Tables II & III
                and their `net_dual` collapse live in
                notebooks/reproduce_conference.py
  cases/toy.py  3-node oracle: fixed data (NETWORK, REDUNDANT_NETWORK, limits,
                bid matrices) + the paper's cases as constants: SCENARIOS (label
                -> DamInstance, via dam_instance(q_dem, max_gen)), MODELS (label
                -> (f, g) pair == (FTR, DAM), built from Contingency lists),
                REDUNDANT_MODELS (double-circuit variant).  No builder fn --
                models are assembled inline with NetworkModel.build.
  cases/texas5.py  5-node of the attribution memo (fig_texas5): parallel WD
                pair (non-singleton Lambda*), WN/SH pair (priced, no circulation
                -> individually identified), SDH triangle (circulation).
                solve_limit_design(patterns) -> the limits making a chosen
                binding pattern bind exactly (maximise the margin at every other
                row; `problem.value > 0` IS the check).
                **Topology only, no bid data by design** -- every proposition
                holds at arbitrary `y ⪰ 0`, and only `prop:support` needs a real
                clearing, which the 3-node oracle already anchors.
  cases/rts_gmlc.py  73-bus loader: SHA-pinned fetch (RTS_GMLC_REF + MANIFEST
                checksums) of bus/branch/gen CSVs + day-ahead load/renewable
                timeseries -> load_network (DC PTDF w/ magnitude taps),
                n1_contingencies (Cont base, LTE post-contingency, bridges
                skipped), dam_instance(interval) (PWL step bids from heat-rate
                segments, interval-synced renewable caps, regional load split to
                buses). Cache gitignored.
notebooks/      run scripts (jupytext `# %%`): explore_toy, explore_texas5,
                explore_rts_gmlc, figures_toy (writes to notebooks/figures/,
                gitignored), and reproduce_conference -- the PowerUp paper's
                fixed target: net_dual + dual_summary, Table II
                (MS_DAM/Delta/eta, plus U/V), Table III, and the toy figures
                inline -- draw_optimum commented out as in the paper, so the
                figures are one per *case* (a scenario changes only the
                direction, which nothing else draws); figures_toy stays the one
                that writes the PNGs, optima and all
tests/          oracle tests: Tables II & III, strong duality, blocks, align;
                test_primitives (meet / primal_face_range / in_span);
                test_attribution (T0 plumbing -- every memo invariant, over all
                4 cases x 3 scenarios x both modes);
                test_rts_gmlc (loader invariants + end-to-end, skips if offline)
```
Library is importable only; analysis run-scripts go in a sibling `notebooks/`
(jupytext `# %%`). Planned: `scenarios.py` (`build_dam_instance` = inverse of
`clear_dam`, a tested roundtrip), `analysis/` (alignment, viz_toy, viz_large).

## Status (2026-08-12): notation aligned to the journal draft, 58 tests pass

- Table II (`MS_DAM`, `Δ`, `η`) and Table III (`μ_f`, `μ_g`) reproduced exactly.
- Robust `μ` bounds + binding/degenerate/slack classification.
- Trade space `D(b;y) = ker C` + matroid-connectivity attribution blocks with
  face-invariant block totals `W_{J_r}`. Validated on the redundant variant
  (parallel `SLa`/`SLb`, reactance 2 each → combined 1, limit 37.5 → combined 75:
  electrically identical to base toy but identical PTDF rows → size-2 block,
  trade `(1,−1)`).
- **Notation aligned to the journal draft** (see the table above). The old code
  had `f`/`g` inverted (`f` was the DAM); every *value* was right and every
  *letter* was wrong. `A`→`K`, `K`→`H`, `A`=incidence; `dual_feasible`→`Lambda`
  (+ new `Lambda_star`, extracted from `robust_bounds`' inline face);
  `support_set`→`J_star`, `support_index`→`J_star_from_bounds`; `discrepancy`
  now returns `level_U/level_V/coverage_U/coverage_V` per `prop:kinds` instead of
  `D_plus`/`D_minus`; model pairs reordered to `(f, g)`.
- **`shapley_repair` removed.** It answered `prop:repair_nonadditive` with an
  order-averaged additive split; the memos answer it with attribution blocks.
  Two answers to one question, and only one is in the paper.
- **Known gap, deliberate:** `marginal_repair` repairs `f` toward `g`, not toward
  `f ∧ g`, so it is *not* the memo's `U^(S)` (whose monotonicity needs the
  one-signed target). `_repair_blocks` also takes `U` blocks from the `g` support,
  where `prop:block_underfunding` uses the blocks of `f`. Both reconciled in
  step 3, not in the rename.
- **Step 2 primitives (`tests/test_primitives.py`, 27 tests):** `meet(f, g)` in
  `network.py`; `primal_face(problem, weights) -> FaceRange` and `in_span` in
  `duality.py`. `meet` guards Assumption 1 explicitly (`np.allclose(K_f, K_g)`)
  and raises `NotImplementedError` naming the stack fallback — that guard is
  where the ERCOT branch goes.

### Two findings from step 2 (both change how later tests must be written)

**1. A DAM-derived direction is never generic.** `d = Kᵀy*` with `y*` supported
on the rows binding at `q^DAM`, so `d` always lies in the normal cone of the face
it exposes. When exactly one row binds, `d` is *parallel* to that row's normal
and the support face is the whole facet — not a vertex. On the 3-node:
`derate`/(a) binds one row (edge face, functionals unidentified),
`dam_outage`/(a) binds two (vertex, everything identified). Consequences:
- Don't assume a "generic direction gives a unique optimum" — it usually doesn't.
- Tests of `prop:primal_invariance` must use a case whose *intersection* face is
  an edge, or the condition holds trivially and the test is vacuous:
  when `span{1} + row(K_{J*(f∧g)})` is all of `Rⁿ` every `w` passes.

**2. `mixed` is tuned to `0.85` to exhibit the T2 headline.**  It has
`coverage_U` rows *and* `level_V` rows (row-level composition is exactly right),
but at the original `0.75` derate `U = 0` in all three scenarios: the derate was
so tight that the DAM's extra `SC` contingency never bit at the intersection
optimum.  That is `prop:composition` item 3 — disagreement composes over rows,
*value* does not.  Sweeping the derate at scenario (a):

| α | 0.75 | 0.85 | 0.90 | 0.95 | 0.98 | 1.00 |
|---|---|---|---|---|---|---|
| `U` | 0 | 1069 | 1604 | 2139 | 2460 | 2674 |
| `V` | 2140 | 1284 | 856 | 428 | 171 | 0 |

`MODELS["mixed"]` and `REDUNDANT_MODELS["mixed"]` now use **α = 0.85**, giving
`U = 1069.5` and `V = 1284.2` at scenario (a) — both strictly positive at once.
Scenarios (b)/(c) still give `V = 0` and `U = 0` respectively, so the witness is
scenario-specific: **(a) is the T2 cell.**

Two consequences, both live in the tests:
- `mixed` is no longer one-signed, so `cor:canonical` does not apply to it and
  `Q(f ∧ g)` is strictly inside *both* regions. It is out of
  `test_meet_region_is_the_tighter_model` and has its own
  `test_mixed_crosses_so_its_meet_is_smaller_than_both`. `block_table` refuses
  the pair `(f, g)` in either order, by design.
- `mixed` no longer has a structurally zero mode, so anything needing one uses
  **`extra_ftr`** (a pure coverage difference feeding `V`, so `U = 0` at every
  scenario).

### The 5-node validates all three attribution structures (step 6)

`notebooks/explore_texas5.py` designs a limit vector making a chosen constraint
pattern bind exactly (maximise the slack at all other rows subject to the chosen
rows being tight), which *is* positing `y` — no bids needed. The resulting
`support_summary` across patterns reproduces the attribution memo's motivating
example exactly:

| pattern | `h` | priced | blocks | max block | `dim ker C` |
|---|---|---|---|---|---|
| `parallel_wd` | 200.0 | 2 | 1 | 2 | 1 |
| `two_blocks` | 1137.5 | 7 | 2 | 4 | 3 |
| `outer_loop` | 575.0 | 4 | 1 | 4 | 1 |
| `no_loop` | 137.5 | 2 | **2** | **1** | **0** |

`no_loop` (the `WN`/`SH` pair) is the one that matters: priced together yet split
into two singleton blocks with an empty trade space, because there is no
circulation joining them — exactly the memo's claim that being in the same
certificate is not a reason to aggregate. `outer_loop` is the contrast: a
circulation binds four rows into one block.

### Primal geometry (step 7)

**Faces and normal-fan cones are dual.** A vertex of `Q(b)` corresponds to a
full-dimensional normal cone, hence to a maximal *realizable active set*, and any
direction interior to that cone exposes it. So `faces(model)` enumerates the
regimes without reference to any `y`, and T5's sweep becomes a lookup. In 2-D the
cyclic vertex order *is* the sweep order. `faces` returns the exposing direction
as `Σ_{i∈rows} k_i` — a strictly positive combination of the cone's generators,
so interior to it (tested: solving at that direction returns exactly that vertex).

**Scale.** Upper Bound Theorem: a `d`-polytope with `m` facets has `~m^⌊d/2⌋`
vertices, with `d = n − 1` after balance and `m = |J(b)|`. 3-node: `d=2, m=6`.
5-node: `d=4, m=18 → ~324`. RTS: `d=72, m≈28,800 → ~10¹⁵⁷`. Contingencies are
survivable; buses are not. Above `MAX_NODES` the *answer* is too big, not the
computation — sample realized directions instead, which is a different and
better-posed question. The guard says this in its error message.

**Plotting needs no correction.** Substituting `q = T u` turns row `i` into
`(Tᵀk_i)ᵀ u ≤ b_i`, so reduced normals are just `K T`; constraint lines are
correct in *any* basis with `1ᵀT = 0`. Only an objective *direction* would need
the covariant `Tᵀd`, and nothing draws one (`fig_support_vi` is TikZ). If a
direction arrow is ever wanted on `(load served, q_S)` axes, put the slack on the
node being eliminated (`C`) and the gradient reads straight off `d` as
`(−d_L, d_S)`.

**Boundedness is one LP, not `2d`.**  The recession cone `{z : Mz ⪯ 0}` is the
polar of `cone(rows of M)`, and a polar is trivial exactly when the cone it came
from is all of `R^d`.  So `Q(b)` is bounded iff the reduced normals *positively
span*: `rank(M) = d` **and** some `λ ⪰ 1` has `Mᵀλ = 0`.  The rank test is not
redundant — a lone zero row satisfies the LP while spanning nothing, and there
is a test pinning exactly that.  This replaced a `±e_k` probe of the recession
cone costing `2d` LPs (29 ms → 11 ms on texas5, `d = 4`); the win is mostly
legibility, since `MAX_NODES` caps `d` at 6 anyway.

**The slack is a labelling convention.** PTDF rows differ between slack
conventions by a multiple of `1ᵀ`, which annihilates balanced injections — so
changing it shifts `d` by a constant vector and leaves `Q(b)`, `h` and `μ`
untouched (tested). Choosing the slack to suit a figure's axes is free.

**Reading the block columns.** Blocks *partition* the priced rows `J*(b;y)`, so
`n_blocks` counts groups, not ambiguity — two rows that cannot trade give **two**
singleton blocks, which is the fully-identified case. Ambiguity is
`dim_trade_space` (`0` = every row separately attributable) or equivalently
`max_block` (`1` = the same thing). `gap_summary` reports `n_priced` alongside so
`n_blocks == n_priced` reads directly as "all singletons". The plain 3-node has
no parallel elements and is all singletons everywhere; the redundant variant and
texas5's `parallel_wd`/`outer_loop` are where it stops being.

**Viz is 3-node only, by decision.** At more nodes two coordinates are a
*projection*, whose boundary edges are not images of individual constraints — a
different object, and one the 5-node work doesn't need.

### Two block attributions -- one table, do not conflate the columns

`block_table(model, d, target=None)` carries both over the *same* partition
(`model`'s blocks). They are different quantities from different theorems with
different multiplicity, and the column names are what keep them apart:

| | support attribution | misalignment attribution |
|---|---|---|
| columns | `value`, `value_frac`, `dim_trade_space` | `loss`, `loss_lo`, `loss_hi`, `identified`, `loss_frac` |
| formula | `W_{J_r} = Σ_{i∈J_r} b_i μ_i` | `U_B = Σ_{i∈B} μ_i[b_i − (Kq)_i]` |
| sums to | `h(model;y)` | `h(model;y) − h(target;y)` |
| needs | **`model` alone** | a **nested pair** |
| theorem | `cor:block_value_invariance` | `prop:block_underfunding` |
| multiplicity | **none** — constant over the whole optimal *dual* face | a **range over the primal injections `q` of the *target* polytope** (`prop:primal_invariance`) |

Omit `target` and you get the support half alone — which is all that is defined
when there is no pair (texas5 probes a posited direction, no DAM, no `f ∧ g`).
Costs 1 solve without a target, 3 with, whatever the block count.

**Column names are mode-agnostic on purpose.** The memos write `W_{J_r}` and `U_B`, but `U_B` names the *U-mode* share specifically, and the same function computes `V_B` whenever you pass `(g, …, f ∧ g)` — a column called `U_B` holding a `V` number is simply wrong. So: `value` is what the block is worth, `loss` is what it costs. Which mode a `loss` belongs to is which model you passed; `labels` is where you record it.

A `value` range column was built once and removed: it is provably and numerically a
point (width `0.0000` vs a face leak of `1.0086` while the `μ_i` in the block
swing by `325.7`), so it is a *test* (`test_toy_blocks`), not a column.
`dim_trade_space` is computed per block as `dim ker C` restricted to it, not
assumed to be `size − 1`. A zero failure mode yields a *typed* null `loss_frac`
so the `U` and `V` frames still stack.

### A finding from step 3: tolerances must scale with the *support values*

Every attribution quantity — `U`, `V`, a repair value, a floor, a block share —
is a **difference of support values** of order `1e4`, so its absolute error is
`~1e-4` however small the difference itself is. Several toy cases have a failure
mode of exactly zero, so a tolerance proportional to the quantity being tested
demands more precision than the inputs carry, and 29 of the first T0 runs failed
on that alone. `tests/test_attribution.py::_tol` scales by `max(|h_f|, |h_g|)`
instead, and this is the right default for anything comparing these objects.

It has to be a *tolerance* rather than exact equality because these bounds are
genuinely **attained**: `cor:canonical` item 1 makes the floor exactly tight for
a uniform derate, and the ceiling closes at `q^∧` by construction, so the
comparisons are routinely between two computations of the same number.

### Results tables (step 4) — deliberately thin

`gap_summary` returns a **dict**, not a frame, so a sweep is
`pl.DataFrame([gap_summary(...) for ...])` and adding a column is adding a key.
`constraint_table` gives per-constraint detail, restricted to rows that either disagree
or carry a share. Both are meant to grow as the analysis asks; don't try to make
them complete up front.

Floors are reported **per mode** (`floor_U`/`floor_V` and their ratios) because
only *level* differences carry a floor at all, so a case routinely has a
meaningful ratio in one mode and a structural zero in the other. And "zero"
for a failure mode means below `1e-6 * max(|h_f|, |h_g|)`, not below `EPS` —
same lesson as the test tolerances.

Already visible in the toy sweep, before T1 is written: `floor_ratio_V = 1.0`
for every uniform derate (`cor:canonical` item 1 — the floor is exactly tight)
and `0.0` for `extra_ftr` (a pure coverage difference — `cor:diagnosable` says
zero floor, all displaced value).

### Removed in step 3, with reasons

- `classify` / `Classification` — binding/degenerate/slack is a one-line read of
  `robust_bounds`' `(lo, hi)`; nothing downstream consumed the labels.
- `J_star_from_bounds` — `J_star` is one CLARABEL solve and exact. The
  `hi`-vector route only existed to reuse a `robust_bounds` call, and with
  `classify` gone there is no reason to run those face LPs just to get a support.
- `support_objective` — it was `b @ mu`. A named wrapper earned nothing.
- `marginal_repair` / `_repair_blocks` — superseded by `repair_value` (repairs
  toward `f ∧ g`, so monotone) and `block_shares` (which takes `U` from the
  blocks of `f` and `V` from those of `g`, per `prop:block_underfunding` —
  `_repair_blocks` had these the other way round).
- `embed` — zero library callers once every comparative quantity aligns its
  models first, which puts `μ` on the common index natively. `align` is the
  whole story.
- `exact_split` — it was `row_shares(...).sum()`. Summing a co-indexed vector is
  the package's normal idiom, not a function.
- `block_shares` — same argument one level up: `row_shares(...)` summed over
  `attribution_blocks(...)`. `metrics.block_table` is the supported way to get
  those numbers, with their ranges attached.
- `support_blocks` / `misalignment_blocks` — briefly two functions for the two
  attributions above. Splitting them fixed the *naming* confusion but cost a
  third `block_table` to join them back. One function with an optional
  `target` does both, and the column names carry the distinction.
- the generic `block_table(model, blocks, values, value_name)` — a view over a
  per-block vector you had already computed; the name now belongs to the real
  table.
- **`failure_modes` and `modes_from_values`** — the redundancy that mattered.
  `failure_modes` had **zero library callers** (16 tests, 2 notebooks) and did
  three support solves then subtracted; `gap_summary` in `metrics.py` does the
  same three solves, the same subtraction, and adds the floors and block shape.
  Two sources for one row, and the test comparing them was circular in substance.
  `gap_summary` is now the **only** place `U`/`V`/`Δ` are computed — it has to
  solve those polytopes anyway for the certificates, so calling out would either
  re-solve them or report a `U` from one certificate beside a `floor_U` from
  another. With `failure_modes` gone, `modes_from_values` had one caller and
  inlined into it: two functions to zero, arithmetic in one place.
- `viz.element_color` → `_element_color`, `rts_gmlc.branch_limits` →
  `_branch_limits`: single-call-site internals, not API.
- **`support_summary`** — kept, but it no longer computes anything. It was a
  second solve-and-partition that had to agree with `block_table` about the same
  certificate (and already disagreed *in principle* about `dim_trade_space`, see
  below); it is now a groupby of that table. `h` is `value.sum()`, exact by
  `cor:block_value_invariance`.
- `_block_shape`'s `dim_trade_space` was `sum(sizes) - len(blocks)`, which assumes
  every block has corank exactly 1. That is **false on texas5** — `two_blocks`
  reported 5 against a true 3, `outer_loop` 3 against a true 1 — so the old
  numbers in the table above were wrong, not just fragile. Now sums the actual
  `dim ker C` per block, which agrees with `dim ker C` taken over all of `J*`
  (as it must: `D` splits as a direct sum over blocks).
- `block_shares` — same argument one level up: it was `row_shares(...)` summed
  over `attribution_blocks(...)`. `metrics.misalignment_blocks` is the supported
  way to get those numbers, with their ranges attached.
- the generic `block_table(model, blocks, values, value_name)` — a view over a
  per-block vector you had already computed. With `support_blocks` and
  `misalignment_blocks` computing the vectors themselves, it earned nothing; the
  name now belongs to the joined per-mode report.
- `dataclasses.replace(model, b=...)` — replaced by `with_limits`, which rebuilds
  the `Contingency` objects too. `replace` leaves them carrying the *old* limits,
  and `align` reads limits from that list, so a repaired model could silently
  re-align to its pre-repair values.

**Vocabulary:** the memos write `f ∧ g` and say "wedge"; the code says **`meet`**
throughout (`meet()`, `q_meet`, `h_meet`). Don't reintroduce `wedge` as an
identifier.

### Per-contingency / asymmetric / emergency-rating limits
Supported: `b` is a free per-row vector and `clear_dam` reads the upper and lower
limit per contingency independently. (`from_limits` was considered and dropped —
`n1_contingencies` builds `Contingency` objects directly, so it was redundant.)

### RTS-GMLC modeling choices (settled)
- **DC PTDF with magnitude transformer taps**: `compute_ptdf` scales susceptance
  `b_eff = 1/(x·tap)` (`Tr Ratio`, 0 ⇒ tap 1). RTS has no phase shifters, so this
  is exact; shunts / line-charging `B` / `BaseKV` are reactive/voltage and do not
  enter a MW-based DCOPF.
- **Limits**: `Cont Rating` pre-contingency (base), `LTE Rating` post-contingency
  (each N-1). The outaged element's own row → `+inf` (it carries no flow).
- **Bids**: PWL step offers, one `M_gen` block-column per heat-rate segment
  (`(HR_incr/1000)·FuelPrice + VOM`); `DamInstance` needed no change.
- **Renewables**: PV/RTPV/Wind/Hydro caps from the same `interval` as load, one
  zero-cost block each. **No UC**: single-period economic dispatch, `PMin` → 0.
- **Islanding**: bridge outages skipped (would island ⇒ singular PTDF), matching
  ISO practice of excluding radial outages from thermal flow constraints.
- Data fully pinned: `RTS_GMLC_REF` commit + `MANIFEST` SHA256s ⇒ reproducible.

### Next — the misalignment-attribution test backlog (T0–T5, toy first)

Steps 1 (notation), 2 (primitives), 3 (`attribution.py`) and 4 (results tables)
are **done**. Remaining, in order:

5. **T1, T2 on the 3-node.** `cor:canonical` items 1/4/5 are exact closed forms —
   pass/fail, no tolerance judgment. **Stop and read the floor-to-total ratio**:
   it decides whether the floor is an instrument or a footnote, and how much
   reporting apparatus is worth carrying to RTS.
6. **T3/T4 on the 5-node** as queries over one shared grid of `(model pair, d)`
   cells, so sub/superadditive witnesses are *found* rather than constructed.
   The case file and notebook are **done** (see the validation below); what
   remains is the grid and the model pairs, i.e. numerical setup.
7. **Search + viz — done** (`polytope.py`, `viz.py`). What remains is using
   them: T5's regime sweep is `faces(model)` plus a groupby, not new code.

Deferred: multi-interval `δ(T)` (Theorem 4 — `dam_instance(interval)` was built
for it); `build_dam_instance` (the inverse of `clear_dam`) — only needed where a
figure must tell a real DAM story, since every proposition except `prop:support`
holds at arbitrary `y ⪰ 0`; storage/batteries; RTS DAM/FTR model pairs; ex-ante
design (bilinear / cutting-plane over a union of polyhedra).

Scale note: dense `K` is fine at 73 buses × ~120 contingencies. The per-row
robust-bound LP loop was the bottleneck and is now fast (candidate restriction to
primal-binding rows + single compiled Parameter-objective LP, ~54x); when only
`J*(b;y)` is needed (attribution/trade space, not the lo/hi ranges), use
`J_star` (one CLARABEL solve, strict complementarity).

### Plotting note (settled)
The 3-node's balanced subspace is 2-D, so `(L, q_S)` with `L = q_S + q_C` is an
exact picture — and a better one than `(q_S, q_C)`, since the axes read
economically. But that basis is a **shear**, so it does not preserve angles: the
arrow to draw for a direction `d` is the *covariant* transform `Tᵀd`, i.e.
`(d_C − d_L, d_S − d_C)`, **not** `d`'s coordinate image. Level sets are then
genuinely perpendicular to the drawn arrow. Sanity check: the arrow is invariant
to `d → d + c·1`, as it must be. Only figures drawing `d` and a supporting
hyperplane are affected — pure region/containment plots are affine and fine.

## Environment

`.venv` from **public PyPI** — the machine's default pip index is a private
Buildkite registry, so always install with `--index-url https://pypi.org/simple`.
Solvers: HiGHS + CLARABEL. Run tests: `.venv/bin/python -m pytest -q`.
