# ftr_align — working notes for Claude

Research codebase computing/analyzing **FTR–DAM structural misalignment** via
support-function geometry, from Erich's two papers (PowerUp conference paper +
INFORMS pitch memo). **Follow the pitch memo's notation** where the two differ.
Goal: one core that scales toy (3-node oracle) → RTS-GMLC → ERCOT without
rewriting per network.

## Locked design decisions (do not relitigate — settled over a long planning pass)

- **One network solve is the primitive**, not a DAM-vs-FTR pair. A `NetworkModel`
  = geometry `A = [K; −K]` + a limit vector `b`. DAM and FTR are two models
  (`f`, `g`). Alignment is a function over two solves: `gap(sol_g, sol_f)`.
- **Models are defined independently**, each with its own `StackedSystem`
  (`NetworkModel.build(net, contingencies, limits)`). To compare two models you
  **must** `align(*models)` them onto a common (union) stacked system — this is
  mathematically required, not convenience: `Δ(g,f;y)` and the shared
  dual-feasible set `Λ(y)` (Corollary 1) only exist when `f`, `g`, and the
  certificate `y` live over one common `A`. `align` matches rows by
  `(contingency, element, side)`; `embed` re-expresses a single vector.
  Valid only under **Assumption 1 (common PTDFs)**. ERCOT's different-PTDF case
  breaks shared `Λ(y)` — flagged, separate, not yet handled.
- After alignment, `b`, `y`, `μ` are **co-indexed full-length vectors** over the
  system rows. Inactive rows (contingencies a model doesn't enforce): `b = +inf`,
  `μ` pinned to 0. `active = np.isfinite(b)`.
- `SupportProblem` is **dual-form** (`min bᵀμ s.t. Aᵀμ + 1s = Aᵀy, μ ≥ 0`) so the
  multipliers `μ` are variables. `.data` is a typed numpy bundle (`SupportData`:
  `A, b, y` → `active`/`direction` properties). `.solve()` returns an immutable
  `SupportSolution` value (so results compose over sets of `y`).
- **Solver seam = one branch**: `solve(solver=...)` passes a CVXPY solver *name*
  (str) straight through (off-the-shelf + commercial). Any non-str object is a
  custom solver, called as `solver.solve(problem) -> SupportSolution`. Erich's
  future ex-ante solver will *orchestrate* CVXPY LP subproblems (bilinear, over a
  union of polyhedra), not replace CVXPY.
- `A` is **dense numpy** — PTDF is structurally dense, so sparse storage wastes.
  The scale lever is active-set / column-generation, not sparse storage.
- Certificate `y` is **stacked-nonnegative** over rows and multiplies `Aᵀ` (not
  `Kᵀ`). Store `y`, not the derived direction `d = Aᵀy`.
- Assembly functions (`dual_feasible`, `support_objective`, `network_constraints`)
  accept numpy **or** cvxpy args, so the same pieces build the ex-post LP and the
  future ex-ante bilinear program.

## Degeneracy convention

Toy patterns can have a **non-unique realized `y*`** ([Feng et al. 2012] LMP
non-uniqueness — the thing the robust framework targets). The paper's numbers
correspond to the **analytic-center certificate**, which an interior-point solver
(`CLARABEL`) produces; simplex (`HIGHS`) gives an equally-valid vertex dual with
a different split. The support *value* given `y*` is unique. **Clear with
CLARABEL** to reproduce paper numbers. In `duality.py`, the classification
tolerance (`CLASS_TOL`) must exceed the face-construction leak (`FACE_TOL`).

## Layout

```
ftr_align/
  network.py    PTDF, contingencies, StackedSystem, NetworkModel,
                build/from_symmetric_limits, align, embed
  solve.py      SupportData, SupportProblem (dual form), SupportSolution,
                DamInstance, DamResult, clear_dam
  duality.py    robust_bounds, classify, net_dual, discrepancy,
                support_index, trade_matrix, trade_space (D=ker C),
                connected_blocks (matroid components), attribution_blocks
  metrics.py    gap(), ratio()
  cases/toy.py  3-node oracle + build_redundant_case (double-circuit variant)
tests/          oracle tests: Tables II & III, strong duality, blocks, align
```
Library is importable only; analysis run-scripts go in a sibling `notebooks/`
(jupytext `# %%`). Planned: `scenarios.py` (`build_dam_instance` = inverse of
`clear_dam`, a tested roundtrip), `analysis/` (alignment, viz_toy, viz_large).

## Status (2026-06-24): slices 1–3 done, 45 tests pass

- Table II (`MS_DAM`, `Δ`, `η`) and Table III (`μ_f`, `μ_g`) reproduced exactly.
- Robust `μ` bounds + binding/degenerate/slack classification.
- Trade space `D(b;y) = ker C` + matroid-connectivity attribution blocks with
  face-invariant block totals `W_{G_r}`. Validated on `build_redundant_case`
  (parallel `SLa`/`SLb`, reactance 2 each → combined 1, limit 37.5 → combined 75:
  electrically identical to base toy but identical PTDF rows → size-2 block,
  trade `(1,−1)`).

### Per-contingency / asymmetric / emergency-rating limits
Already supported: `b` is a free per-row vector and `clear_dam` reads the upper
and lower limit per contingency independently. Only a `from_limits(system,
{contingency: (upper, lower)})` convenience constructor is deferred — it's the
first small addition for RTS-GMLC.

### Next
1. **RTS-GMLC loader** in `cases/rts_gmlc.py` — hand-rolled parse of the
   RTS-GMLC bus/branch/gen CSVs → `PhysicalNetwork` + N-1 `contingencies`, then
   the same `build → align → clear_dam/SupportProblem` flow. Add `from_limits`.
   Scale note: dense `A` is fine at 73 buses × ~120 contingencies, but the
   per-row robust-bound LP loop in `duality.py` should restrict to `I(b;y)`
   (and eventually warm-start) — not a rewrite.
2. Multi-interval funding-gap `δ(T)` (Theorem 4: single solve on `Σ y*_t`).
3. Misalignment attribution (`D±`, block-repair counterfactuals).
4. Ex-ante design (custom bilinear / cutting-plane solver over a union of
   polyhedra; reuses the assembly functions with `g`, `y` promoted to variables).

## Environment

`.venv` from **public PyPI** — the machine's default pip index is a private
Buildkite registry, so always install with `--index-url https://pypi.org/simple`.
Solvers: HiGHS + CLARABEL. Run tests: `.venv/bin/python -m pytest -q`.
