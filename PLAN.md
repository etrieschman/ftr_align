# Pressure-testing the paper: what to build, in order

Ordered by your priority. Each item has **Search** (brute force that works today)
and **Solve** (construct it directly). Sections are independent — build top-down.

---

## 0. The prerequisite: constructing FTR/DAM pairs

An `(f, g)` pair is fully determined by two things:

| design variable | what it controls |
|---|---|
| contingency sets `C_f`, `C_g` | **coverage** differences |
| limit vectors `b_f`, `b_g` on shared rows | **level** differences |

and the disagreement kinds follow mechanically:

- `c ∈ C_g \ C_f` → coverage difference feeding **U** (FTR blind to a case DAM enforces)
- `c ∈ C_f \ C_g` → coverage difference feeding **V**
- shared `c`, `b_f < b_g` → level difference feeding **V**; `b_f > b_g` → level feeding **U**

So *every* target below is a statement about which rows disagree and which bind.
That is the whole design space. Nothing else matters.

**Search.** Enumerate `C_f, C_g ⊆ candidates` (after the redundancy screen below),
sweep a derate `α` on shared rows, score with `gap_summary`. With ~5 surviving
contingencies that's `4^5 × |α|` cells — trivial.

**Solve.** Pick the disagreement pattern you want, then use `solve_limit_design`
(generalized, §2) to find limits realizing it.

---

## 1. Both failure modes live (`U > 0` and `V > 0`)

**Objective:** maximize `min(U, V)`.

**Search.** Grid over `(C_f, C_g, α)` from §0, take `min(U,V)` from `gap_summary`.
This is how you got `mixed` at α=0.85 on the toy; the same sweep works here.

**Solve.** `U > 0` needs a row where `g` is strictly tighter *and binds at the
intersection optimum*; `V > 0` needs the mirror. So: choose one binding pattern
containing a row only `g` enforces, and one containing a row only `f` enforces,
and solve for limits making **both** patterns exact in one program —
`solve_limit_design` already takes multiple named patterns sharing one `b`.
That is the feature you need, already written.

**Watch:** binding is the whole game. At α=0.75 the toy's extra contingency never
bit and `U = 0` in every scenario. Non-binding ⇒ no witness.

---

## 2. Attribution above constraint level, **spanning contingencies** — DONE

Rows from different contingencies *can* share a block: the cycle space of `A`
governs the base case only, and the object is the stacked system. The block
condition is about the trade matrix `C`, columns `cᵢ = [k̄ᵢ ; bᵢ]`:

```
(a)  Σ_{i∈S} zᵢ k̄ᵢ = 0      ← PTDF only.  b-free.  Screen with it.
(b)  Σ_{i∈S} zᵢ bᵢ = 0      ← implied whenever every row of S is pinned.
```

Built and validated in `notebooks/explore_texas5.py`: circuit enumeration finds
the candidates, `solve_limit_design` realizes them, and the headline is a size-3
block carrying both a `base:` row and a post-outage row. See `LEARNINGS.md` for
why (b) never needs imposing.

---

## 3. Floor is partial (`0 < floor_ratio < 1`) — ANSWERED

Not binary, and it did not need the constructed same-mode case. Across the 60
meet vertices of the texas5 design, `floor_ratio_V` lands strictly inside
`(0.396, 1)` at 56 of them.

The old structural argument was incomplete. `floor_U = Σ μᵢ (bᵢ − (f∧g)ᵢ)` is
supported where the models disagree *and* the row is priced — so the ratio is 1
only when **every priced row** carries the disagreement. Here `V`'s certificate
also prices post-outage rows, where `g` and `f∧g` agree, and those contribute
nothing to the floor while contributing to `V`. A derate uniform over the base
rows is still non-uniform over the *stack*.

So `cor:canonical` item 1 (floor exactly tight) needs uniformity over the whole
stacked system, not over one contingency. That is the sentence for the paper.

---

## 4. Repairs sub/superadditive

`U^(S∪T)` vs `U^(S) + U^(T)` for disjoint `S, T` (`prop:repair_nonadditive`).

**Search.** This one is genuinely cheap and I would just brute force it. Take
`D = ` the disagreeing rows, enumerate disjoint pairs of singletons (or of small
sets), and compute all three repair values. `repair_value(..., base=h_model)`
reuses the unrepaired solve, so each is **one** LP. `|D|²/2` LPs — seconds.
Report `U^(S∪T) − U^(S) − U^(T)`; sign gives you sub vs super, and you want
witnesses of *both* signs.

**Solve.** Partial, and worth knowing: repairs interact only through rows that
bind *together*. If `S` and `T` price into different blocks and no injection
trades between them, the repair is additive. So look for non-additivity where
`S` and `T` land in the **same block** (superadditive candidates: relaxing both
opens a direction neither opens alone) versus **different blocks joined by a
shared binding row** (subadditive candidates: each repair alone already captures
the shared row's slack). Use §2's block computation to pick candidates instead of
enumerating blindly.

---

## 5. Primal invariance has content (`identified = False`) — WITNESSED

20 of 94 probed blocks report `identified = False`, all in **U**, all at the
normal of a **contingency** row. `faces(meet)` vertices give `True` vacuously, as
predicted; the facet normals are where the condition bites, because they expose a
whole facet rather than a vertex.

Mechanism: `f` is base-only, so at `d = kᵢ` for an outage row it cannot price
that row and its block weight `Σ μᵢ kᵢ` falls outside `span{1} + row(K_{J*(f∧g)})`.
`V` never fails, since `g` contains the row.

Cross-validated: the span test and the two-LP width agree on all 94 blocks —
identified widths ≤ `6.7e-4`, unidentified ≥ `48.1`.

What remains is editorial, not computational: decide what a `False` means for the
paper's claim, given the witness is structural (one model blind to a whole
contingency) rather than a knife-edge.

---

## Visualization, now that plots are gone

texas5 is 4-D; `viz` is 3-node only and a projection would be dishonest (its
boundary edges are not images of constraints).

1. **`faces(model)` is the picture as a table** — every vertex with its tight set.
   32 rows for texas5 base. This is the complete regime map.
2. **Facet census** — for each row: never tight (redundant), sometimes, always.
   Three lines over `faces()`, and it doubles as the §0 redundancy screen.
3. **A 2-D slice, if you want an actual plot.** A slice is exact where a
   projection is not. `plane_system` needs one small change: an offset `q₀`, so
   `q = q₀ + Tu` and `c` becomes `b − Kq₀`. ~4 lines, and then `polygon`,
   `draw_region` and `draw_constraints` all work unchanged at 5 nodes.

---

## Build order

1. **Repair non-additivity sweep** (§4) — the only computational item left.
   Brute force, cheap, and §2's blocks pick the candidates.
2. **Write up §3 and §5** — both are answered; what remains is deciding how they
   land in the paper.
3. **Slice plotting** (§viz 3) last.
