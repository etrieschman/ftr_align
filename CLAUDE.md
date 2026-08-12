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
| `A` | node-branch incidence | `PhysicalNetwork.A` |
| `H_c`, `H` | PTDF under `c`; stacked over contingencies | `.ptdf(key)`, `NetworkModel.H` |
| `K = [H; −H]` | stacked constraint matrix | `NetworkModel.K` |
| `y`, `d = Kᵀy` | price certificate; node-space direction | `DamResult.y`, `.direction` |
| `Λ(y)`, `Λ*(b;y)` | dual-feasible set, optimal dual face | `Lambda`, `Lambda_star` |
| `J*(b;y)` | dual-optimal support | `J_star`, `J_star_from_bounds` |
| `D(b;y) = ker C(b;y)` | trade space | `trade_space`, `trade_matrix` |
| `W_{J_r}` | block total | `attribution_blocks(...)["W"]` |

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
- **`align`/`embed` put two models on a common row index.** `embed(values,
  source, target)` matches rows by `(contingency, element, side)`; `align(*models)`
  rebuilds onto a union contingency set (unenforced contingencies added with
  `+inf` limits). Used for row-level cross-model comparison (lining up `μ_f`/`μ_g`,
  `discrepancy`, joint blocks) and — legitimately — as model-level preprocessing
  for the intersection `f ∧ g`, which needs a common index by construction. Never
  needed before a plain support solve. Valid only under **Assumption 1 (common
  PTDFs)**; ERCOT's different-PTDF case is flagged, separate, not yet handled.
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
  different blocks. Collapse to signed net duals only for *reporting*
  (`net_dual`), matching the paper's own convention.
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
- Assembly functions (`Lambda`, `Lambda_star`, `support_objective`,
  `network_constraints` in `solve.py`) accept numpy **or** cvxpy args and are
  reused by `solve` and `duality` (one definition each — no re-spelling).
- `K` is **dense numpy** — PTDF is structurally dense, so sparse storage wastes.
  The scale lever is active-set / column-generation, not sparse storage.

## Degeneracy convention

Toy patterns can have a **non-unique realized `y*`** ([Feng et al. 2012] LMP
non-uniqueness — the thing the robust framework targets). The paper's numbers
correspond to the **analytic-center certificate**, which an interior-point solver
(`CLARABEL`) produces; simplex (`HIGHS`) gives an equally-valid vertex dual with
a different split. The support *value* given `y*` is unique. **Clear with
`solver={"solver": "CLARABEL"}`** to reproduce paper numbers. In `duality.py`, the classification
tolerance (`CLASS_TOL`) must exceed the face-construction leak (`FACE_TOL`).

## Layout

```
ftr_align/
  network.py    PTDF (compute_ptdf takes incidence `A` + optional per-element
                `tap`), is_connected (bridge/islanding guard), PhysicalNetwork
                (owns `A`, optional `tap`), Contingency (key + limits; pass one
                `upper` for symmetric), NetworkModel (owns K & b; `.H` is the
                stacked PTDF, the upper half of K), align, embed,
                contingency_label/element_label
  solve.py      assembly fns (Lambda/Lambda_star/support_objective/
                network_constraints), SupportData, SupportProblem,
                solve_support_cvxpy, SupportSolution, DamInstance, DamResult,
                clear_dam (returns y* and direction)
  duality.py    robust_bounds (lo/hi over the dual face; mu restricted to
                primal-binding candidates, single compiled Parameter-objective LP
                reused across rows, forced onto HiGHS internally -- the thin
                value-slab is infeasible for interior-point and must share the
                base solve's engine; bounds are solver-invariant), classify,
                J_star_from_bounds (from a robust_bounds hi vector), J_star
                (J*(b;y) from one CLARABEL solve via strict complementarity --
                CLARABEL required, ~50-130x cheaper than the face-LP loop),
                net_dual, trade_matrix, trade_space (D=ker C), connected_blocks
                (matroid components via QR fundamental circuits),
                attribution_blocks (index defaults to J_star), discrepancy
                (level/coverage x U/V per prop:kinds), marginal_repair
  metrics.py    alignment_summary (Table II), dual_summary (Table III)
  cases/toy.py  3-node oracle: fixed data (NETWORK, REDUNDANT_NETWORK, limits,
                bid matrices) + the paper's cases as constants: SCENARIOS (label
                -> DamInstance, via dam_instance(q_dem, max_gen)), MODELS (label
                -> (f, g) pair == (FTR, DAM), built from Contingency lists),
                REDUNDANT_MODELS (double-circuit variant).  No builder fn --
                models are assembled inline with NetworkModel.build.
  cases/rts_gmlc.py  73-bus loader: SHA-pinned fetch (RTS_GMLC_REF + MANIFEST
                checksums) of bus/branch/gen CSVs + day-ahead load/renewable
                timeseries -> load_network (DC PTDF w/ magnitude taps),
                n1_contingencies (Cont base, LTE post-contingency, bridges
                skipped), dam_instance(interval) (PWL step bids from heat-rate
                segments, interval-synced renewable caps, regional load split to
                buses). Cache gitignored.
tests/          oracle tests: Tables II & III, strong duality, blocks, align;
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

Step 1 (notation) is **done**. Remaining, in order:

2. **Three primitives.** `meet(f, g)` (the `f ∧ g` model); a *primal* face
   explorer (max/min over `{q : Kq ⪯ b, dᵀq = h}` — the mirror of
   `robust_bounds`, and likely HiGHS-only for the same thin-slab reason), needed
   to produce two distinct `q^∧` for `prop:primal_invariance`; and an `in_span`
   least-squares residual test, which serves *both* invariance propositions
   (`prop:invariant_subset` via `b_S ⊥ ker C`, and `prop:primal_invariance`).
3. **`attribution.py` — reporting, not primitives.** `U`, `V`, `Δ = U − V`;
   `U^(S)`; the floor `Σ_{i∈S} μ^f_i [f_i − (f∧g)_i]`; the ceiling
   `fᵀμ − dᵀq` (takes `q` as an argument — by `rem:injection_nesting` the choice
   of `q` *is* the bound); `cor:exact_split`; `U_B`. All arithmetic over solved
   objects. T0 assertions written alongside, not after. Retarget
   `marginal_repair` to `f ∧ g` and reconcile `_repair_blocks` with
   `prop:block_underfunding` here.
4. **Results tables** (tidy polars, run-level + row-level); port
   `alignment_summary`/`dual_summary` onto them as views.
5. **T1, T2 on the 3-node.** `cor:canonical` items 1/4/5 are exact closed forms —
   pass/fail, no tolerance judgment. **Stop and read the floor-to-total ratio**:
   it decides whether the floor is an instrument or a footnote, and how much
   reporting apparatus is worth carrying to RTS.
6. **5-node (`toy_degen`), bid-free.** Topology already matches `fig_texas5`
   (parallel `WD1`/`WD2`, the no-circulation `WN`/`SH` pair, the `SDH` triangle)
   but `M_GEN`/`M_DEM` are copy-pasted from the 3-node at the wrong node count.
   Posit `d` directly rather than writing bids. Then T3/T4 as queries over one
   shared grid of `(model pair, d)` cells, so sub/superadditive witnesses are
   *found* rather than constructed.
7. **Search + viz, one tool.** Vertex enumeration of `Q(b)` gives active sets and
   representative directions at any dimension; in 2-D, sorting the vertices
   angularly *is* the `d`-sweep, so T5 and the paper figure come free. Guard at
   `n_nodes > 7` plus a `max_vertices` cap. Above the guard, don't build a
   sampler — the realized `dam_instance(interval)` sweep already samples the
   right distribution for N1–N6.
   Viz: `slice2d`/`polygon`/plot layer, defaulting to **projection** (shadow) for
   `n > 3`, with bid bounds as an opt-in *overlay* (they are not network
   feasibility). Port the styling from the old script, not its geometry.

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
