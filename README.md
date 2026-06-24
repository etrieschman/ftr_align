# ftr_align

FTR/DAM structural misalignment via support-function geometry.

Both DAM merchandising surplus and the maximum supportable FTR payout are
support functions of network-feasible injection polytopes over a *shared*
dual-feasible set `Λ(y)`; the signed gap `Δ(g, f; y)` isolates the
network-model component of FTR/DAM misalignment, and the optimal dual face
yields degeneracy-invariant constraint-level attribution.

This is a research codebase designed to run the same analysis across scales —
the 3-node toy model (the reference oracle), RTS-GMLC, and eventually ERCOT —
without rewriting core logic per network.

## Core idea

The primitive is **one network solve**. A *network model* is shared geometry
`A = [K; −K]` plus a limit vector `b`. DAM and FTR are two models (`f`, `g`)
over the same `StackedSystem`, so `b`, the certificate `y`, and the duals `μ`
are co-indexed vectors over the same rows. Alignment is then a function over
two solves: `Δ = h(g; y) − h(f; y)`.

## Layout

```
ftr_align/
  network.py    geometry: PTDF, contingencies, StackedSystem, NetworkModel,
                shared_system / embed (index alignment across contingency sets)
  solve.py      SupportProblem (dual form), SupportSolution, clear_dam
  duality.py    Λ*(b;y): robust μ bounds, binding/degenerate/slack classification,
                signed net duals, DAM/FTR limit discrepancy
  metrics.py    gap(), ratio()
  cases/toy.py  3-node oracle (PowerUp Appendix B)
tests/          oracle tests: Tables II & III, strong duality, Prop. 1
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://pypi.org/simple -e ".[dev]"
.venv/bin/python -m pytest -q
```

## Status

- Support values (`MS_DAM`, `Δ`, `η`) — reproduces **Table II** exactly.
- Dual values (`μ_f`, `μ_g`) — reproduces **Table III** exactly.
- Robust multiplier bounds + binding/degenerate/slack classification.

A note on degeneracy: several toy patterns have a non-unique realized
certificate `y*` (the [Feng et al., 2012] LMP non-uniqueness the framework
targets). The reported numbers correspond to the analytic-center certificate,
which interior-point clearing (CLARABEL) produces; the support *value* given
`y*` is unique.

### Next

Trade space `D(b;y)` and matroid-connectivity attribution blocks; multi-interval
funding-gap decomposition `δ(T)`; RTS-GMLC; ex-ante design.
