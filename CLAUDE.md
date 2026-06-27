# ftr_align — working notes for Claude

Research codebase computing/analyzing **FTR–DAM structural misalignment** via
support-function geometry, from Erich's two papers (PowerUp conference paper +
INFORMS pitch memo). **Follow the pitch memo's notation** where the two differ.
Goal: one core that scales toy (3-node oracle) → RTS-GMLC → ERCOT without
rewriting per network.

## Locked design decisions (do not relitigate — settled over a long planning pass)

- **One network solve is the primitive**, not a DAM-vs-FTR pair. A `NetworkModel`
  owns its geometry: a `PhysicalNetwork` + a tuple of `Contingency` (each carrying
  the line limits enforced under it). `NetworkModel.build(net, contingencies)`
  assembles `A = [K; −K]` and the stacked limit vector `b`. There is **no separate
  `StackedSystem`** — it was folded into `NetworkModel`.
- **Support is parametrized by a node-space direction `d ∈ Rⁿ`**, not a row-space
  certificate. `SupportProblem(model, direction)`; `h_Q(d) = max_{q∈Q} dᵀq`.
  Because `d` lives in node space (shared by every model on the network), DAM and
  FTR support **values and the gap need NO alignment** — each solves on its own
  polytope with the same `d`. This was the key correction: alignment is *not*
  required for `Δ`. `clear_dam` returns the DAM certificate `y*` (over its own
  rows) **and** `direction = A_damᵀ y*`.
- **`align`/`embed` are result-conversion tools, used ONLY for row-level
  cross-model comparison** (lining up `μ_f`/`μ_g`, `D±`, joint blocks) — not
  preprocessing before a solve. `embed(values, source, target)` matches rows by
  `(contingency, element, side)`; `align(*models)` rebuilds onto a union
  contingency set (unenforced contingencies added with `+inf` limits). Valid
  only under **Assumption 1 (common PTDFs)**; ERCOT's different-PTDF case is
  flagged, separate, not yet handled.
- `b` and per-row vectors (`μ`) are **co-indexed full-length vectors** over the
  model's rows. Unmonitored rows: `b = +inf`, `μ` pinned to 0.
  `active = np.isfinite(b)`.
- `SupportProblem` is **dual-form** (`min bᵀμ s.t. Aᵀμ + 1s = d, μ ≥ 0`) so the
  multipliers `μ` are variables. `.data` is a typed numpy bundle (`SupportData`:
  `A, b, direction` → `active` property). `.solve()` returns an immutable
  `SupportSolution` value.
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
- Assembly functions (`dual_feasible`, `support_objective`, `network_constraints`
  in `solve.py`) accept numpy **or** cvxpy args and are reused by `solve` and
  `duality` (one definition each — no re-spelling).
- `A` is **dense numpy** — PTDF is structurally dense, so sparse storage wastes.
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
  network.py    PTDF (compute_ptdf takes optional per-element `tap`),
                is_connected (bridge/islanding guard), PhysicalNetwork (optional
                `tap`), Contingency (key + limits; pass one `upper` for
                symmetric), NetworkModel (owns A & b), align, embed,
                contingency_label/element_label
  solve.py      assembly fns (dual_feasible/support_objective/network_constraints),
                SupportData, SupportProblem, solve_support_cvxpy, SupportSolution,
                DamInstance, DamResult, clear_dam (returns y* and direction)
  duality.py    robust_bounds (lo/hi over the dual face; mu restricted to
                primal-binding candidates, single compiled Parameter-objective LP
                reused across rows, forced onto HiGHS internally -- the thin
                value-slab is infeasible for interior-point and must share the
                base solve's engine; bounds are solver-invariant), classify,
                support_index (from a robust_bounds hi vector), support_set
                (I(b;d) from one CLARABEL solve via strict complementarity --
                CLARABEL required, ~50-130x cheaper than the face-LP loop),
                net_dual, trade_matrix, trade_space (D=ker C), connected_blocks
                (matroid components via QR fundamental circuits),
                attribution_blocks (index defaults to support_set), discrepancy
  metrics.py    gap(), ratio(), alignment_summary (Table II), dual_summary (Table III)
  cases/toy.py  3-node oracle: fixed data (NETWORK, REDUNDANT_NETWORK, limits,
                bid matrices) + the paper's cases as constants: SCENARIOS (label
                -> DamInstance, via dam_instance(q_dem, max_gen)), MODELS (label
                -> (dam, ftr) pair, built from Contingency lists), REDUNDANT_MODEL
                (double-circuit variant).  No builder fn -- models are assembled
                inline with NetworkModel.build.
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

## Status (2026-06-26): slices 1–3 + RTS-GMLC loader done, 59 tests pass

- Table II (`MS_DAM`, `Δ`, `η`) and Table III (`μ_f`, `μ_g`) reproduced exactly.
- Robust `μ` bounds + binding/degenerate/slack classification.
- Trade space `D(b;y) = ker C` + matroid-connectivity attribution blocks with
  face-invariant block totals `W_{G_r}`. Validated on the redundant variant
  (`models(..., net=REDUNDANT_NETWORK, limits=REDUNDANT_LIMITS)`):
  (parallel `SLa`/`SLb`, reactance 2 each → combined 1, limit 37.5 → combined 75:
  electrically identical to base toy but identical PTDF rows → size-2 block,
  trade `(1,−1)`).

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

### Next
1. Multi-interval funding-gap `δ(T)` (Theorem 4: single solve on `Σ y*_t`). The
   `dam_instance(interval)` API was built for the sweep.
   Scale note: dense `A` is fine at 73 buses × ~120 contingencies. The per-row
   robust-bound LP loop was the bottleneck and is now fast (candidate restriction
   to primal-binding rows + single compiled Parameter-objective LP, ~54x); when
   only the support `I(b;y)` is needed (attribution/trade space, not the lo/hi
   ranges), use `support_set` (one CLARABEL solve, strict complementarity).
   Also still TODO: storage/batteries
   (single-period `DamInstance` can't model charge/discharge/SOC; excluded in the
   loader) and example DAM/FTR model pairs for RTS.
2. Misalignment attribution (`D±`, block-repair counterfactuals).
3. Ex-ante design (custom bilinear / cutting-plane solver over a union of
   polyhedra; reuses the assembly functions with `g`, `y` promoted to variables).

## Environment

`.venv` from **public PyPI** — the machine's default pip index is a private
Buildkite registry, so always install with `--index-url https://pypi.org/simple`.
Solvers: HiGHS + CLARABEL. Run tests: `.venv/bin/python -m pytest -q`.
