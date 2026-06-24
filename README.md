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

The primitive is **one network solve**. A `NetworkModel` owns its geometry: a
network plus `Contingency` objects (each carrying the line ratings enforced
under it), from which it assembles `A = [K; −K]` and the limit vector `b`.

The support function is parametrized by a **node-space direction** `d ∈ Rⁿ`:
`h_Q(d) = max_{q∈Q} dᵀq`. Since `d` is shared by every model on the network, DAM
and FTR support values and the gap `Δ = h(g; d) − h(f; d)` need **no alignment** —
each solves on its own polytope. `clear_dam` returns the DAM certificate `y*` and
the induced `direction = Aᵀy*`. `align`/`embed` are used only to put two models'
per-row duals (`μ_f`, `μ_g`) on a common index for attribution comparison.

## Layout

```
ftr_align/
  network.py    geometry: PTDF, Contingency (key + ratings), NetworkModel
                (owns A & b); align()/embed() for row-level result comparison
  solve.py      SupportProblem (direction-parametrized, dual form),
                SupportSolution, clear_dam (-> certificate y* and direction)
  duality.py    Λ*: robust μ bounds, binding/degenerate/slack classification,
                signed net duals, trade space D=ker C, attribution blocks,
                DAM/FTR limit discrepancy
  metrics.py    gap(), ratio(), alignment_summary (II), dual_summary (III)
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
- Trade space `D(b;y) = ker C` and matroid-connectivity **attribution blocks**
  with face-invariant block totals `W_{G_r}`, validated on a double-circuit
  variant where `Λ*` is genuinely non-singleton.

A note on degeneracy: several toy patterns have a non-unique realized
certificate `y*` (the [Feng et al., 2012] LMP non-uniqueness the framework
targets). The reported numbers correspond to the analytic-center certificate,
which interior-point clearing (CLARABEL) produces; the support *value* given
`y*` is unique.

### Next

Multi-interval funding-gap decomposition `δ(T)`; misalignment attribution
(`D±`, block-repair counterfactuals); RTS-GMLC; ex-ante design.
