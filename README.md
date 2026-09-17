# ftr_align

FTR/DAM structural misalignment via support-function geometry.

Congestion revenue is the support value of the DAM network model in the
interval's congestion direction `v`; the same direction evaluated on the FTR
model is the most the auction could have paid. Their difference `Δ(v)` isolates
the network-model term of the funding gap, and evaluated on the intersection of
the two models it splits into underfunding exposure `U` and lost hedge value `V`.
Both are attributed to constraints at the resolution of attribution blocks, where
the dollar figures are invariant to dual multiplicity.

This is a research codebase designed to run the same analysis across scales —
the 5-node attribution case, RTS-GMLC, and ERCOT — without rewriting core logic
per network. The 3-node toy remains as the code-verification oracle.

## Core idea

A **network model is constraint rows `(K, b)` on a node set**. It is stored one
contingency at a time (PTDF rows `H_c` and limits), with `K = [H; −H]` and `b`
assembled on demand; `NetworkModel.build(net, contingencies)` is one way to build
one. The intersection of two models is their rows stacked, `intersection(ftr, dam)`,
and needs only a common node set — not common elements, shift factors or
reference bus.

The support function is parametrized by a **node-space direction** `v`:
`h_Q(v) = max_{q∈Q} vᵀq`. Because `v` lives in node space, the FTR, DAM and
intersection support values need no alignment — each solves on its own rows.
`clear_dam` returns the DAM certificate `y*` and `direction = Kᵀy*`.

## Layout

```
ftr_align/
  network.py      PhysicalNetwork + Contingency (builder inputs), ContingencyRows,
                  NetworkModel (nodes + rows; H, K, b), intersection, with_limits
  solve.py        SupportProblem (dual form), SupportSolution, clear_dam
  duality.py      robust mu bounds, J*, primal face ranges, shift space
                  S = ker Kbar_J^T, attribution blocks, block totals
  attribution.py  row shares of a failure mode, block identification tests
  polytope.py     vertices / normal-fan regimes of Q
  metrics.py      gap_summary, summary, block_table, constraint_table
  viz.py          3-node figures
  cases/          toy (3-node oracle), texas5 (5-node), rts_gmlc (73-bus loader)
notebooks/        run scripts (jupytext `# %%`)
tests/            oracle and invariant tests
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://pypi.org/simple -e ".[dev]"
.venv/bin/python -m pytest -q
```

## Status

- Support values, `Δ`, `U`, `V` (`gap_summary`); block-level attribution with
  identification ranges (`block_table`); PowerUp Tables II & III reproduced on
  the 3-node.
- 5-node: attribution structure (circulation vs co-pricing, cross-contingency
  blocks, cone criterion, regime map). See `VALIDATION.md`.
- Next: RTS-GMLC model pairs; ERCOT.
