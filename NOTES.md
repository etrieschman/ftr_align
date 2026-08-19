# Where we got to — and what to think about tomorrow

## The headline

**A cross-contingency attribution block, constructed rather than found.**

```
V mode, span_block direction:
  block 0  size 3  {base:WN:upper, base:ND:upper, WD2:WN:upper}   spans = true
```

Two base rows and one post-outage row in a single block. Its attributed value
cannot be split between "the base case" and "the outage" — which is the claim
worth making in the paper. And it was *designed*: pick a circuit, impose its
trade, solve for limits. No search over interesting-looking cases.

Alongside it, `base_block` gives `{base:SD, base:SH, base:DH}` — the SDH
triangle, the known base-only circulation — on **the same network**. That
contrast is the point: one `b`, two blocks, one spanning and one not.

## What made it work (three things that bit)

1. **Blocks belong to the *loser*, not the meet.** `prop:block_underfunding`
   attributes U on `f`'s blocks and V on `g`'s. My first attempt designed the
   circuit on the meet and got all singletons, because `f` (base-only) doesn't
   even contain the contingency rows. A spanning block can only appear in **V**,
   since only `g` has rows from both cases.
2. **The derate breaks the trade.** `Σ zᵢbᵢ = 0` was imposed on the designed
   limits; scaling `g`'s base rows by `1/α` destroys it on exactly the rows meant
   to form the block. Fix: derate only base rows **outside** the circuit support.
3. **`margin = 0` is not infeasibility.** `extra` ties `b[WD2] == b[WD1]`, and the
   pair is parallel with equal reactance, so binding one forces the other and a
   non-pattern row sits at its limit. Selection is now by *realizability* — take
   the first (outage, pattern set) with strictly positive margin — rather than by
   circuit count.

## Tomorrow

**First, the honest gap.** `identified` is still `True` everywhere. A size-3
circuit gives `face dim = 5 − 3 = 2`, so the meet's optimal face *is*
positive-dimensional — yet primal invariance still holds. Either the condition is
harder to violate than expected, or `block_share_range` is collapsing for a
reason worth understanding. That is the most interesting open question, and it is
a real result either way.

**Second, does U ever get a spanning block?** By the argument above it cannot, on
this construction — `f` is base-only. If you want one in U, FTR needs its own
contingency, which changes the FTR = base-only premise. Worth deciding whether
that premise is a modelling choice or a simplification.

**Third, size-2 spanning circuits.** There were only 4 in the whole survey, and
they are the most legible witnesses — two rows, exactly parallel across a
contingency boundary. `face dim = 3` there. Worth trying before the size-3 ones
for the paper figure.

**Fourth, the floor.** Still binary (0 or 1) everywhere. To get a fraction you
need level *and* coverage differences feeding the **same** mode. On this
construction both feed V (derate) and U (coverage) separately, so it will stay
binary until you deliberately build the same-mode case.

## Files

- `LEARNINGS.md` — geometry and method notes worth not rediscovering
- `DESIGN_PLAN.md` — the six-step design procedure this implements
- `PLAN.md` — the broader search/solve strategy per proposition
- `notebooks/explore_texas5.py` — steps 1–5 implemented at the bottom
