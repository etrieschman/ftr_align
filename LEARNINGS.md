# Learnings

Short notes worth not rediscovering. Newest last.

## Geometry

**Ambient dimension is 4.** Mean-removed PTDF rows are orthogonal to `1`, so all
of them live in a 4-dim subspace of `R⁵` no matter how many rows there are. The
number of independent dependencies is just `#rows − 4`.

**Why contingencies produce so many circulations.** Base alone: 9 rows, rank 4,
so **5** dependencies — exactly the cycle space of the graph, `9 edges − 5 nodes
+ 1 = 5`. KVL is the mechanism: around any cycle `Σ_e x_e k_e = 0`. Adding one
outage puts 8 more rows into *the same* 4-dim space, so dependencies go 5 → 13.
Not new structure — just more vectors in a space that was already full. Hence
2545 of 2749 circuits span base and contingency.

**Circuit size fixes the face dimension:** `face dim = 4 − (|S| − 1) = 5 − |S|`.
Size 5 → vertex, size 2 → 3-face. Small circuits are easier to realize *and* give
the positive-dimensional faces where `prop:primal_invariance` has content.

**A nondegenerate vertex has all-singleton blocks, always.** Its `J*` holds `d`
*independent* rows, so `Σ zᵢ k̄ᵢ = 0` forces `z = 0`. Searching `faces()` is
systematically searching where blocks cannot be. Search circuits instead.

**The outaged element's own row is identically zero** under its own contingency.
Asking it to bind forces `b = 0`, which is infeasible against `b ≥ lo`. This was
100% of the infeasibility in the first contingency sweep (72 = 36 pairs × 2).

## Method

**We set `J* = S`, we don't filter for it.** `solve_limit_design` with pattern
`S` makes exactly `S` bind; a circuit is rank-deficient by one, so the dual face
has positive dimension and every row of `S` is priced.

**A fresh `b` per pattern makes results near-vacuous.** Each cell becomes a
different network engineered to produce its own answer. The interesting object is
**one `b`, several patterns** — which combinations can coexist is a real
structural fact, and it's `solve_limit_design`'s multi-pattern feature.

**Block condition splits into a free half and a linear half:**
`Σ zᵢ k̄ᵢ = 0` is b-free (pure linear algebra, screen with it), and
`Σ zᵢ bᵢ = 0` is *linear in b* (impose it via `trades=`).

## Results

**The floor is a switch, not a gauge.** `floor_ratio` is exactly 1.0 on a level
difference and exactly 0.0 on a coverage difference, never between — across all
12 toy cells. So it carries one bit. The quantity worth reporting at scale is
what *fraction of the gap is coverage vs level*.

**`identified = False` is still unwitnessed anywhere.** Every block reports
`True`, because every intersection optimum has been a vertex where the condition
holds vacuously. Needs a positive-dimensional meet face — see circuit size above.

**`dim_trade_space` was wrong on texas5** under the old `sum(sizes) − n_blocks`
shortcut, which assumes every block has corank 1. True values: `two_blocks` 3
(not 5), `outer_loop` 1 (not 3). Now computed as the real `dim ker C` per block.

**`mixed` is tuned to α = 0.85** so both modes are positive at scenario (a)
(`U = 1069`, `V = 1284`). At 0.75 the derate was tight enough that the extra
contingency never bound and `U = 0`. Binding is the whole game.
