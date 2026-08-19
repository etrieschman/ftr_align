# %%
import numpy as np
import polars as pl
import cvxpy as cp
from itertools import combinations, product
from tqdm import tqdm
import matplotlib.pyplot as plt

from ftr_align import (
    SupportProblem,
    NetworkModel,
    Contingency,
    meet,
    with_limits,
)
from ftr_align.metrics import (
    block_table,
    constraint_table,
    row_labels,
    summary,
    gap_summary,
)
from ftr_align.cases import texas5
from ftr_align.cases.texas5 import solve_limit_design
from ftr_align.polytope import faces
from ftr_align.cases.texas5 import WN, WS, WD1, WD2, ND, NH, SD, SH, DH
from ftr_align.cases.texas5 import W, N, S, D, H

from ftr_align.solve import (
    CENTER,
)  # {"solver": "CLARABEL"}: the interior certificate J* needs

pl.Config.set_tbl_rows(50)
pl.Config.set_fmt_table_cell_list_len(-1)
pl.Config.set_fmt_str_lengths(100)
np.set_printoptions(precision=3, suppress=True)

# %%
# -------------------------------------
# INSPECT NETWORK
# -------------------------------------
net = texas5.NETWORK
H_base = net.ptdf()  # not `H`: that is the node index imported above
n = net.n_nodes
ell = net.n_elements

print("~~~~~~ Toy network")
print("nodes   :", net.node_names.tolist())
print("elements:", net.element_names.tolist())
print("slack   :", net.node_names[net.slack_idx])
print("x       :", net.x.tolist())

# Only this model's K is used by the designer -- its limits are what we solve for.
TEMPLATE = NetworkModel.build(net, [Contingency(None, texas5.BASE_LIMITS)])


def extra(model, b, q):
    """Case-specific ties: the parallel pair must stay indistinguishable, an
    element carries one limit across every contingency, and W/N generate while H
    consumes."""
    const = [
        b[WD1] == 100,
        b[WD2] == b[WD1],
        q[W] >= 0,
        q[N] >= 0,
        q[H] <= 0,
        q >= -1_000,
        q <= 1_000,
    ]
    const += [
        b[model.rows_upper(None)] == b[model.rows_upper(c)]
        for c in model.keys
        if c is not None
    ]
    return const


# %%
# -------------------------------------
# THE DESIGN HELPERS
# -------------------------------------
TOL = 1e-9


def realized(problem):
    """Whether a design LP actually pinned its pattern: optimal, with a strictly
    positive margin."""
    return (
        problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        and problem.value is not None
        and problem.value > TOL
    )


def sides(model, rows):
    """Every upper/lower assignment of ``rows`` (given as upper row indices), as
    global row indices into the stacked system.

    All ``2^|rows|`` of them.  Flipping every row at once maps ``q -> -q``, which
    is a symmetry of the limits but not of ``extra``, which pins the sign of
    ``q[W]``, ``q[N]`` and ``q[H]`` -- 22 of 112 mirrored pattern pairs give
    different margins, so no side may be locked.
    """
    half = model.n_rows // 2
    return [
        [r + c for r, c in zip(rows, choice)]
        for choice in product([0, half], repeat=len(rows))
    ]


def direction_of(model, rows):
    """``d = K^T y`` for the certificate putting unit weight on ``rows``.

    The inverse of :func:`solve_limit_design`: positing a binding pattern is
    positing ``y``, which is why this case carries no bid data.  ``d`` lives in
    NODE space, so any model on this network can consume it -- no alignment.
    """
    y = np.zeros(model.n_rows)
    y[list(rows)] = 1.0
    return model.K.T @ y


# %%
# -------------------------------------
# SEARCH DIRECTIONS
# -------------------------------------
# Four named patterns share one `b`, and every row of every pattern can bind on
# either side: 15 rows, so 2^15 joint designs.  Screen each pattern's sides
# alone, then join only the survivors.
PATTERN_ELEMENTS = {
    "parallel_wd": [WD1, WD2],
    "two_blocks": [WN, ND, WD1, WD2, SD, DH, SH],
    "outer_loop": [WN, NH, SH, WS],
    "no_loop": [WN, SH],
}

survivors = {}
for name, elements in PATTERN_ELEMENTS.items():
    kept = []
    for side in sides(TEMPLATE, elements):
        problem, _, _ = solve_limit_design(TEMPLATE, {name: side}, extra=extra)
        if realized(problem):
            kept.append(side)
    survivors[name] = kept
    print(f"{name:12}: {len(kept):4} / {2 ** len(elements):4} sides realizable alone")

# One design = one side assignment per pattern.  `product` takes the choices in
# `names` order, so `zip(names, choice)` pairs each pattern with its own pick.
names = list(survivors)
grid = list(product(*(survivors[name] for name in names)))
print(f"{len(grid)} joint designs to try")

results = []
for choice in tqdm(grid):
    patterns = {name: side for name, side in zip(names, choice)}
    problem, b, _ = solve_limit_design(TEMPLATE, patterns, extra=extra)
    if not realized(problem):
        continue
    results.append(
        {
            "patterns": patterns,
            "limits": b.value.copy(),  # the designer reuses one `b`
            "margin": float(problem.value),
            # a tie-break only: prefer the design binding more upper rows
            "n_upper": sum(r < TEMPLATE.n_rows // 2 for side in choice for r in side),
        }
    )

if not results:
    raise RuntimeError("no strictly feasible pattern set")

# plot margin against n_upper
plt.scatter(
    y=[r["n_upper"] for r in results],
    x=[r["margin"] for r in results],
    s=10,
    alpha=0.5,
)
plt.title(f"Margin by n_upper (N={len(results)})")
plt.ylabel("Number of upper rows binding")
plt.xlabel("Margin")


# %%
# -------------------------------------
# SELECT BEST PATTERN, VERIFY IT BINDS
# -------------------------------------
best = max(results, key=lambda r: (round(r["margin"], 2), r["n_upper"]))
PATTERNS = best["patterns"]
model = with_limits(TEMPLATE, best["limits"])
print("margin:", best["margin"])

display(
    pl.DataFrame(
        {
            "element": net.element_names,
            # b is over rows; limits are symmetric, so the upper half is all of it
            "limit": best["limits"][:ell],
        }
    )
)

# One solve per pattern, reused for the check, the detail and the blocks.
solved = {}
for name, rows in PATTERNS.items():
    problem_p = SupportProblem(model, direction_of(model, rows))
    solved[name] = (problem_p, problem_p.solve(solver=CENTER))

# Did the construction work?  The designed limits should make the requested rows
# bind and nothing else.
display(
    pl.DataFrame(
        [
            {
                "pattern": name,
                "h": sol.value,
                "requested": len(PATTERNS[name]),
                "bound": int(sol.binding.sum()),
                "exact": set(PATTERNS[name])
                == set(np.flatnonzero(sol.binding).tolist()),
                "rows": row_labels(model, np.flatnonzero(sol.binding)),
            }
            for name, (_, sol) in solved.items()
        ]
    ).with_columns(pl.col("h").round(2))
)


# %%
# -------------------------------------
# PER-PATTERN DETAIL
# -------------------------------------
for name, (problem_p, _) in solved.items():
    print(f"\n~~~~~~ Pattern {name}:")
    display(pl.DataFrame(summary(model, problem_p.data.direction)))
    display(constraint_table(model, problem_p.data.direction))
    display(block_table(model, problem_p.data.direction))


# %%
# -------------------------------------
# ENUMERATE CIRCUITS
# -------------------------------------
# Circuits are the candidate blocks
MAX_CIRCUIT = 5  # kbar rows span 4 dimensions, so circuits run to 5


def circuits_for(model, outage, base_up, sizes=(2, 3)):
    """Minimal dependent sets among usable upper rows, with their null vector.

    Minimal means ``rank(kbar_S) == |S| - 1`` with a null vector of FULL support;
    a zero entry means a smaller dependent set is hiding inside.  The SIGNS of
    ``z`` are not a filter -- flipping a row to its lower side flips that entry,
    so every full-support circuit is usable under some assignment, and
    :func:`sides` is what picks one.

    Lower rows are negated upper rows, so only upper rows are searched.  The
    outaged element's own row is identically zero and is dropped.
    """
    rows = [
        int(i)
        for i in list(model.rows_upper(None)) + list(model.rows_upper(outage))
        if np.abs(model.K[int(i)]).max() > 1e-9
    ]
    Kbar = {i: model.K[i] - model.K[i].mean() for i in rows}
    out = []
    for size in sizes:
        for S in combinations(rows, int(size)):
            Kbar_S = np.array([Kbar[i] for i in S])
            if np.linalg.matrix_rank(Kbar_S, tol=1e-8) != size - 1:
                continue
            z = np.linalg.svd(Kbar_S.T, full_matrices=True)[2][-1]
            if np.abs(z).min() < 1e-8:  # not minimal
                continue
            out.append(
                {
                    "outage": int(outage),
                    "outage_name": str(model.network.element_names[outage]),
                    "rows": list(S),
                    "elements": row_labels(model, list(S)),
                    "z": z.tolist(),
                    "size": int(size),
                    "spans": bool(set(S) & base_up) and bool(set(S) - base_up),
                }
            )
    return out


circuits = []
for c in tqdm(range(ell)):
    outage_model = NetworkModel.build(
        net, [Contingency(None, texas5.BASE_LIMITS), Contingency(c, texas5.BASE_LIMITS)]
    )
    base_up = {int(i) for i in outage_model.rows_upper(None)}
    circuits += circuits_for(outage_model, c, base_up, sizes=range(2, MAX_CIRCUIT + 1))

circ = pl.DataFrame(circuits)
print(f"{circ.height} circuits, {circ['spans'].sum()} spanning base and contingency")

print(
    "Count of directional circuits by size and whether they span base and contingency"
)
display(
    pl.DataFrame(
        circ.pivot(index="size", on="spans", values="rows", aggregate_function="len")
    ).sort("size", descending=False)
)

print("Count of directional circuits that span, by outage")
rank = (
    circ.filter(pl.col("spans"), pl.col("outage_name") != "WD2")
    .group_by("outage", "outage_name")
    .len()
    .sort("len", descending=True)
)
display(rank)


# %%
# -------------------------------------
# CHOOSE OUTAGE, PATTERNS, AND DESIGN THE LIMITS
# -------------------------------------
# One `b`, two designed patterns: a spanning circuit (the cross-contingency
# block) and a base-only circuit (the contrast).
TRUNC = 50
designs = []
for outage, outage_name in rank.select("outage", "outage_name").iter_rows():
    print(f"\n~~~~~~ outage {outage_name}")
    outage_model = NetworkModel.build(
        net,
        [
            Contingency(None, texas5.BASE_LIMITS),
            Contingency(outage, texas5.BASE_LIMITS),
        ],
    )
    base_up = {int(i) for i in outage_model.rows_upper(None)}
    found = circuits_for(outage_model, outage, base_up)
    span = sorted([f for f in found if f["spans"]], key=lambda f: f["size"])
    base = sorted(
        [f for f in found if not f["spans"] and set(f["rows"]) <= base_up],
        key=lambda f: f["size"],
    )

    # Screen each pattern alone.  Infeasible alone implies infeasible jointly,
    # and the screen is linear where the join is a product.
    span_ok, base_ok = [], []
    for name, candidates, keep in (
        ("span_block", span[:TRUNC], span_ok),
        ("base_block", base, base_ok),
    ):
        for S in candidates:
            for side in sides(outage_model, S["rows"]):
                problem, _, _ = solve_limit_design(
                    outage_model, {name: side}, extra=extra
                )
                if realized(problem):
                    keep.append((S, side))
    print(f"Patterns surviving (span, base): ({len(span_ok)}, {len(base_ok)})")

    for S_span, rows_span in tqdm(span_ok):
        for S_base, rows_base in base_ok:
            if set(S_base["rows"]) & set(S_span["rows"]):
                continue
            patterns = {"span_block": list(rows_span), "base_block": list(rows_base)}
            problem, b, _ = solve_limit_design(outage_model, patterns, extra=extra)
            if not realized(problem):
                continue
            designs.append(
                {
                    "outage": outage,
                    "model": outage_model,
                    "patterns": patterns,
                    "margin": float(problem.value),
                    "b": b.value.copy(),
                    "span": S_span,
                    "base": S_base,
                }
            )

if not designs:
    raise RuntimeError("no (outage, pattern set) realizable with a positive margin")

display(
    pl.DataFrame(
        [
            {
                "outage": net.element_names[d["outage"]],
                "margin": d["margin"],
                "span_size": d["span"]["size"],
                "base_size": d["base"]["size"],
                "span_pattern": row_labels(d["model"], d["patterns"]["span_block"]),
                "base_pattern": row_labels(d["model"], d["patterns"]["base_block"]),
            }
            for d in designs
        ]
    ).sort(["span_size", "base_size", "margin"], descending=True)
)

# Biggest spanning circuit first -- that is the headline block; margin breaks ties.
design = max(designs, key=lambda d: (d["base"]["size"], d["margin"]))
design_model = design["model"]
print("outage:", net.element_names[design["outage"]], " margin:", design["margin"])
for name, rows in design["patterns"].items():
    print(f"  {name:11}: {row_labels(design_model, rows)}")
display(
    pl.DataFrame(
        {
            "element": net.element_names,
            "b": design["b"][list(design_model.rows_upper(None))].round(2),
        }
    )
)


# %%
# -------------------------------------
# BUILD THE PAIR AND INSPECT AT EACH PATTERN DIRECTION
# -------------------------------------
# g (DAM) = base + the outage, at the designed limits.
# # f (FTR) = base only at the designed limits, uniformly derated by ALPHA.

# U is a pure coverage difference (f is blind to the outage)
# V is a pure level difference (g is looser on every base row).
#
# Inspect ONE DESIGNED PATTERN {span, base} AT A TIME.
ALPHA = 0.85
b_design = design["b"][list(design_model.rows_upper(None))]

f_model = NetworkModel.build(net, [Contingency(None, ALPHA * b_design)])
g_model = NetworkModel.build(
    net, [Contingency(None, b_design), Contingency(design["outage"], b_design)]
)
m_model = meet(f_model, g_model)
# The pattern's row indices are `design_model`'s, so they transfer to g only
assert g_model.keys == design_model.keys


def report(d, title):
    """gap_summary, both modes' blocks, and both constraint tables at one d."""
    print(f"\n~~~~~~ {title}")
    display(pl.DataFrame(gap_summary(f_model, g_model, d, solver=CENTER)))
    for mode, looser in (("U", f_model), ("V", g_model)):
        table = block_table(looser, d, m_model, labels={"mode": mode})
        # The headline: a block carrying BOTH a `base:` label and an outage label
        # cannot be split between the base case and the outage.
        display(
            table.with_columns(
                spans=pl.Series(
                    [
                        any(lbl.startswith("base:") for lbl in members)
                        and any(not lbl.startswith("base:") for lbl in members)
                        for members in table["members"]
                    ]
                )
            ).drop("rows")
        )
    # Both sides.  `coverage` can only show on the U side: the meet inherits g's
    # contingency limits untouched, so g and the meet never differ on coverage.
    #
    # A coverage row carries b = +inf, hence mu = 0 and zero loss BY
    # CONSTRUCTION -- the model does not enforce it at all, so it can never be
    # priced and can never carry a share.  It survives the noise filter on its
    # name, not on its magnitude; filtering by magnitude alone hides every one.
    keep = (
        pl.col("priced")
        | (pl.col("difference") == "coverage")
        | (pl.col("loss").abs() > 1e-6)
    )
    for mode, looser in (("U", f_model), ("V", g_model)):
        print(f"   constraints, {mode} side")
        display(
            constraint_table(
                looser, d, m_model, labels={"mode": mode}, solver=CENTER
            ).filter(keep)
        )


# Three candidate directions.  Each designed pattern on its own, and their union
span_rows = list(design["patterns"]["span_block"])
base_rows = list(design["patterns"]["base_block"])
CANDIDATES = {
    "span_block": span_rows,
    "base_block": base_rows,
    "combined": span_rows + base_rows,
}

overview = []
for name, rows in CANDIDATES.items():
    d = direction_of(g_model, rows)
    gap = gap_summary(f_model, g_model, d, solver=CENTER)
    binding = set(
        np.flatnonzero(SupportProblem(g_model, d).solve(solver=CENTER).binding).tolist()
    )
    zero = 1e-6 * max(1.0, abs(gap["h_g"]))
    overview.append(
        {
            "pattern": name,
            "rows": row_labels(g_model, rows),
            # only the single patterns were designed; the union was not, so its
            # J* is not expected to match
            "realized_on_g": binding == set(rows),
            **{k: gap[k] for k in ("h_f", "h_g", "h_meet", "U", "V", "Delta")},
            "both_modes": abs(gap["U"]) > zero and abs(gap["V"]) > zero,
        }
    )

overview = pl.DataFrame(overview)
display(overview.drop("rows"))
display(overview.select("pattern", "rows"))

for name in overview.filter(pl.col("both_modes"))["pattern"]:
    report(direction_of(g_model, CANDIDATES[name]), name)


# %%
# -------------------------------------
# STEP 6: THE REGIME MAP -- SWEEP THE MEET'S VERTICES
# -------------------------------------
# Faces of Q(b) and cones of its normal fan are dual: a vertex of the meet
# corresponds to a full-dimensional cone of directions, and `faces` returns one
# direction interior to each.  Sweeping those enumerates every regime the pair
# can realize, with no y to posit and no design to run.
regimes = faces(m_model)
print(f"{len(regimes)} vertices of the meet, d = {n - 1}")

sweep = pl.DataFrame(
    [
        gap_summary(
            f_model,
            g_model,
            face.direction,
            solver=CENTER,
            labels={"vertex": i, "n_tight": len(face.rows)},
        )
        for i, face in enumerate(tqdm(regimes))
    ]
)

display(
    sweep.select(
        "vertex",
        "n_tight",
        "h_f",
        "h_g",
        "h_meet",
        "U",
        "V",
        "Delta",
        "floor_ratio_U",
        "floor_ratio_V",
        "n_priced_U",
        "n_blocks_U",
        "n_priced_V",
        "n_blocks_V",
    ).head(10)
)

# Both modes live at once?  U is coverage-only here and V level-only, so the
# question is whether the outage rows bind at the same vertex the base rows do.
zero = 1e-6 * sweep["h_g"].abs().max()
live = (pl.col("U").abs() > zero) & (pl.col("V").abs() > zero)
display(
    sweep.select(
        n=pl.len(),
        both_modes=live.sum(),
        U_only=(pl.col("U").abs() > zero).and_(pl.col("V").abs() <= zero).sum(),
        V_only=(pl.col("U").abs() <= zero).and_(pl.col("V").abs() > zero).sum(),
        neither=(pl.col("U").abs() <= zero).and_(pl.col("V").abs() <= zero).sum(),
    )
)

# Attribution shape across regimes: `n_blocks == n_priced` is fully identified.
for mode in ("U", "V"):
    display(
        sweep.group_by(
            f"n_priced_{mode}",
            f"n_blocks_{mode}",
            f"max_block_{mode}",
            f"dim_trade_space_{mode}",
        )
        .agg(pl.len(), pl.col(mode).mean().alias(f"mean_{mode}"))
        .sort(f"dim_trade_space_{mode}", f"n_priced_{mode}", descending=True)
    )

# Floors, per mode.  A uniform derate makes the floor exactly tight (ratio 1) and
# a coverage difference gives it nothing (ratio 0).  Neither is the general case:
# the derate here is uniform over the BASE rows but the certificate also prices
# outage rows, where the models agree and the floor collects nothing -- so the
# ratio lands strictly inside.  Bucket it rather than grouping raw floats.
display(
    sweep.select("floor_ratio_U", "floor_ratio_V")
    .unpivot(variable_name="mode", value_name="ratio")
    .with_columns(
        bucket=pl.when(pl.col("ratio").is_null())
        .then(pl.lit("no mode"))
        .when(pl.col("ratio").abs() < 1e-9)
        .then(pl.lit("0 (all displaced)"))
        .when((pl.col("ratio") - 1).abs() < 1e-9)
        .then(pl.lit("1 (floor tight)"))
        .otherwise(pl.lit("strictly between"))
    )
    .group_by("mode", "bucket")
    .agg(
        pl.len(),
        pl.col("ratio").min().alias("min"),
        pl.col("ratio").max().alias("max"),
    )
    .sort("mode", "bucket")
)


# %%
# -------------------------------------
# THE SHOWCASE CELL: A VERTEX WHERE BOTH MODES ARE POSITIVE
# -------------------------------------
# The designed pattern directions give U = 0: f is base-only at ALPHA*b, and at
# those directions only base rows bind for it, so h_f = h_meet.  The regime map
# is where both modes are live -- take the vertex with the largest U.  The block
# tables below carry `spans`, so you can read off whether it is also a
# cross-contingency block.
both_modes = sweep.filter(live).sort("U", descending=True)
display(both_modes.select("vertex", "U", "V", "Delta", "floor_ratio_V", "n_blocks_V"))
vertex = int(both_modes["vertex"][0])
report(regimes[vertex].direction, f"vertex {vertex}")


# %%
# -------------------------------------
# STEP 7: WHERE IS THE BLOCK SHARE NOT IDENTIFIED?
# -------------------------------------
# `identified` holds vacuously at a vertex: `span{1} + row(K_{J*(meet)})` is then
# all of R^n and every w lies in it.  It has content only where the MEET's
# optimal face is positive-dimensional, and the cheapest way to force that is to
# point d along a single row's normal -- then that whole facet is optimal.
probe = []
for i in np.flatnonzero(np.isfinite(m_model.b)):
    d = m_model.K[i]
    if np.abs(d).max() < 1e-9:  # the outaged element carries no flow
        continue
    for mode, looser in (("U", f_model), ("V", g_model)):
        for row in block_table(looser, d, m_model, solver=CENTER).iter_rows(named=True):
            probe.append(
                {
                    "facet": row_labels(m_model, [i])[0],
                    "mode": mode,
                    "members": row["members"],
                    "size": row["size"],
                    "loss": row["loss"],
                    "width": row["loss_hi"] - row["loss_lo"],
                    "identified": row["identified"],
                    "dim_trade_space": row["dim_trade_space"],
                }
            )

probe = pl.DataFrame(probe)
display(probe.group_by("mode", "identified").agg(pl.len(), pl.col("width").max()))
display(probe.filter(~pl.col("identified")))

# `identified` is a span test and `width` is a pair of LPs over the same face;
# they answer the same question and must agree.  The threshold has to scale with
# the support values -- an identified block still shows a width of order the
# face-construction leak, which is ~1e-3 at h ~ 1e3, not ~1e-6.
leak = 1e-6 * max(1.0, abs(sweep["h_g"].abs().max()))
display(
    probe.select(
        rows=pl.len(),
        leak=pl.lit(leak),
        disagree=(~pl.col("identified")).and_(pl.col("width").abs() <= leak).sum()
        + pl.col("identified").and_(pl.col("width").abs() > leak).sum(),
        max_width_identified=pl.col("width").filter(pl.col("identified")).max(),
        min_width_unidentified=pl.col("width").filter(~pl.col("identified")).min(),
    )
)


# %%
# -------------------------------------
# LEARNINGS
# -------------------------------------
# 1. The trade constraint is implied, never imposed.  Pinning every row of a
#    circuit S gives `sum z_i b_i = sum z_i (Kq)_i = (sum z_i k_i)^T q = 0`.  So
#    `solve_limit_design` needs no `trades` argument: the only case it would bind
#    is a circuit NOT contained in a pinned pattern, and the designer maximises
#    the margin at every unpinned row, so J* IS the pattern by construction.
#
# 2. The global sign flip is not a symmetry.  Flipping every row of a pattern
#    maps `q -> -q`, which the limits do not see -- but `extra` pins the sign of
#    q[W], q[N] and q[H], so it does.  22 of 112 mirrored pairs differ in margin.
#    Both sides of every row must be enumerated.
#
# 3. Screen each pattern alone before joining.  Infeasible alone implies
#    infeasible jointly, and the screen is linear where the join is a product.
#
# 4. One limit per element across contingencies removes the confound.  With
#    `b[base, e] == b[c, e]` no level difference can hide between two
#    contingencies, so every cross-contingency block found is topology, not
#    ratings.  It costs realism -- real post-contingency ratings are higher --
#    and it costs margin, since pinning a base row now pins its twin.
#
# 5. `dim_trade_space` is the true corank, not `size - 1`.  A block of 9 rows on
#    this network has corank 5, not 8.  What does hold, and is the check worth
#    running, is that the per-block dims sum to `dim ker C` over all of J*.
#
# 6. The combined direction is not a designed pattern.  `d = K^T(1_span+1_base)`
#    exposes a face that is neither, and the spanning block disappears there.
#
# 7. `identified = False` IS witnessable, and the facet normals are where.  20 of
#    94 probed blocks fail primal invariance, every one of them in U and every
#    one at the normal of a CONTINGENCY row.  The mechanism: f is base-only, so
#    at `d = k_i` for an outage row f cannot price that row at all, and its block
#    weight `sum mu_i k_i` falls outside `span{1} + row(K_{J*(meet)})`.  V never
#    fails, because g does contain the row.  The two tests agree perfectly: 0 of
#    94 blocks disagree, with identified widths topping out at 6.7e-4 and
#    unidentified widths starting at 48.1 -- five orders of magnitude apart, so
#    this is real multiplicity and `primal_invariant` and `block_share_range`
#    validate each other.
#
# 8. The floor is a gauge here, not a switch.  `floor_ratio_V` is strictly
#    inside (0.396, 1) at 56 of 60 vertices and exactly 1 at only 4.  The old
#    binary reading assumed the derate covers every priced row; it does not --
#    V's certificate also prices outage rows, where f and g agree and the floor
#    collects nothing.  `floor_ratio_U` is 0.0 at all 58 vertices where U is
#    nonzero, a pure coverage difference as designed.
#
# 9. Both modes live at 58 of the 60 vertices, without tuning ALPHA.  The regime
#    map answers "does this pair exhibit both failure modes" by enumeration,
#    where the toy needed a hand-picked derate.
