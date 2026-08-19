# Designing one network: base + one contingency, several patterns

Goal: **one** limit vector `b` that simultaneously realizes a base-only block, a
cross-contingency block, and a singleton control — so all three are comparable on
the same network rather than on three engineered ones.

FTR = base only. DAM = base + one outage. Derate FTR's base rows by `α` so both
failure modes live (coverage → U, level → V).

---

## Step 1 — choose the contingency

From the circuit table already computed, per outage `c` count:

- small **spanning** circuits (size 2–3) — the new structure, easiest to realize
- small **base-only** circuits — the contrast
- whether the two can be made row-disjoint

Pick the `c` maximizing small spanning circuits. One groupby, no new computation.

## Step 2 — choose the pattern set

Three patterns on one `b`, chosen with **minimal row overlap** (overlapping rows
couple their equality constraints and are the main source of infeasibility):

| name | rows | gives |
|---|---|---|
| `base_block` | a base-only circuit, size 3 | block of 3, base rows only |
| `span_block` | a spanning circuit, size 2–3 | **cross-contingency block** |
| `singletons` | 4 *independent* rows (no circuit) | all-singleton control, a vertex |

Prefer size 2–3: fewer equality constraints, and `face dim = 5 − |S|` means they
leave a positive-dimensional meet face — the regime where `identified` can fail.

## Step 3 — design the limits

```
solve_limit_design(
    tmpl,                                  # base + c; with shared limits this IS the meet
    {"base_block": S_base, "span_block": S_span, "singletons": S_vertex},
    trades=[(S_base, z_base), (S_span, z_span)],   # sum z_i b_i = 0 -> actually blocks
    extra=...,                             # WD ties + injection signs
)
```

`margin > 0` is the whole check. If infeasible, drop `singletons` first, then
fall back to one circuit.

## Step 4 — build the pair

- `g` (DAM) = base + `c`, limits from the design
- `f` (FTR) = base only, base limits scaled by `α < 1`
- confirm `meet(f, g)` reproduces the designed template exactly

## Step 5 — inspect

Per pattern direction `d = Kᵀ·1_S`:

- `summary(meet, d)` — expect `n_blocks`, `max_block = |S|`, `dim_trade_space ≥ 1`
- `block_table(g, d, meet(f, g))` — members of the span block should carry **both**
  a `base:` and a `<c>:` label; that is the headline
- `gap_summary(f, g, d)` — both modes positive
- check `identified` on the two circuit patterns

## Step 6 — recount with `b` in the matrix

Re-run the circuit enumeration on `C = [k̄ᵢ; bᵢ]` with the designed `b`, restricted
to `J*`. Confirms which candidates actually became blocks rather than merely
could have.

---

## Risks, in order

1. **Three patterns on one `b` may be infeasible.** Most likely cause is row
   overlap between the chosen circuits. Mitigate by choosing disjoint supports;
   fall back to two patterns.
2. **Side selection.** A circuit's `z` has mixed signs; each row's side
   (upper/lower) must be chosen consistently with both the binding pattern and
   `Σ zᵢ bᵢ = 0`. Let the LP settle it rather than pre-filtering.
3. **`α` interacts with binding.** Too tight and the contingency rows stop
   binding, exactly as on the toy at 0.75. Sweep a few values, check `U > 0`.
