"""Plotting the feasible injection polytope, for the 3-node figures.

At three nodes power balance leaves a 2-D injection space, so the picture is
exact.  At more nodes two coordinates are a projection, whose boundary edges are
not images of individual constraints -- a different object, and not one these
functions draw.

One layer per call, onto an axis the caller owns.  :func:`frame_axes` must come
first: the drawing layers clip to the current axis limits.
"""

from __future__ import annotations

import numpy as np

from .network import NetworkModel
from .polytope import free_basis, plane_system, polygon
from .solve import SupportProblem

# Solid for the DAM model, dotted for the FTR overlay, matching the paper's
# existing figures.
DAM_STYLE = dict(color="grey", ls="solid", fill_alpha=0.30)
FTR_STYLE = dict(color="C4", ls="dotted", fill_alpha=0.30)
MEET_STYLE = dict(color="C1", ls="dashed", fill_alpha=0.0)


def basis(model: NetworkModel, drop: int | None = None) -> np.ndarray:
    """Plot basis for a model, eliminating node ``drop`` (default: the slack).

    Any ``T`` with balanced columns works; pass your own for different axes.
    """
    n = model.network.n_nodes
    return free_basis(n, model.network.slack_idx if drop is None else drop)


def to_plot(T: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Node injections -> plot coordinates (the inverse of ``q = T u``)."""
    return np.linalg.lstsq(np.asarray(T), np.asarray(q), rcond=None)[0]


def draw_region(
    ax, model: NetworkModel, T=None, *, label=None, outline=True, bounds=None, **style
):
    """Fill ``Q(b)``, and by default stroke its boundary.

    ``outline=False`` when :func:`draw_constraints` is also drawn -- those lines
    already trace the boundary.  ``bounds`` cuts the fill by extra half-planes in
    plot coordinates; pass the same ones to :func:`draw_halfplane` to draw them.
    """
    T = basis(model) if T is None else T
    style = {**DAM_STYLE, **style}
    verts = polygon(model, T, bounds=bounds)
    if len(verts) == 0:
        return verts
    closed = np.vstack([verts, verts[:1]])
    ax.fill(
        verts[:, 0],
        verts[:, 1],
        color=style["color"],
        alpha=style["fill_alpha"],
        label=label,
        zorder=1,
    )
    if outline:
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color=style["color"],
            ls=style["ls"],
            lw=style.get("lw", 1.4),
            zorder=3,
        )
    return verts


def _element_color(model: NetworkModel, row: int) -> str:
    """Colour for a constraint row, keyed by its element, so a line keeps its
    identity across contingencies, models and figures.  Upper and lower rows of the
    same element share it.
    """
    return f"C{int(row) % model.ell}"


def draw_constraints(
    ax,
    model: NetworkModel,
    T=None,
    *,
    ls=None,
    lw=0.9,
    names=True,
    colors=None,
    styles=("solid", "dashed", "dashdot", "dotted"),
    **kw,
):
    """One line per enforced row, across the current axis limits.

    Colour is the element and line style the contingency; pass ``ls`` to force one
    style for the whole model.  Only upper-limit rows are labelled, so the legend
    gets one entry per line.
    """
    T = basis(model) if T is None else T
    M, c, rows = plane_system(model, T)
    labels = model.labels()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    half, ell = model.n_rows // 2, model.ell

    for a, rhs, row in zip(M, c, rows):
        row = int(row)
        colour = colors[row % ell] if colors is not None else _element_color(model, row)
        style = ls if ls is not None else styles[((row % half) // ell) % len(styles)]
        upper = row < half
        text = (
            f"{labels['contingency'][row]}:{labels['element'][row]}"
            if names and upper
            else None
        )
        _draw_line(ax, a, rhs, xlim, ylim, label=text, color=colour, ls=style, lw=lw)


def _draw_line(ax, a, c, xlim, ylim, *, label=None, color="grey", ls="solid", **kw):
    """The line ``a^T u = c``, clipped to the axes.

    Solved for whichever coordinate has the larger coefficient, so a vertical line
    is not a special case.
    """
    lw = kw.get("lw", 0.9)
    if abs(a[1]) >= abs(a[0]):
        if abs(a[1]) < 1e-12:
            return
        x = np.array(xlim)
        ax.plot(x, (c - a[0] * x) / a[1], color=color, ls=ls, lw=lw, label=label, zorder=2)
    else:
        y = np.array(ylim)
        ax.plot((c - a[1] * y) / a[0], y, color=color, ls=ls, lw=lw, label=label, zorder=2)


def draw_optimum(ax, model: NetworkModel, direction, T=None, *, solver=None, **style):
    """Mark the support maximiser for ``direction`` and draw the supporting
    hyperplane through it.

    The hyperplane is a level set of the objective, so it is exact in any basis.
    """
    T = basis(model) if T is None else T
    style = {**DAM_STYLE, **style}
    sol = SupportProblem(model, direction).solve(solver=solver, want_primal=True)
    u = to_plot(T, sol.q)
    ax.plot(
        *u,
        marker="o",
        ms=6,
        color=style["color"],
        mec="black",
        mew=0.6,
        zorder=5,
        ls="none",
    )
    a = np.asarray(T).T @ np.asarray(direction)  # objective gradient in plot coords
    _draw_line(ax, a, float(a @ u), ax.get_xlim(), ax.get_ylim(),
               color=style["color"], ls=style["ls"], lw=1.0)
    return sol


def draw_halfplane(ax, a, c, *, label=None, color="C2", ls="solid", lw=0.8):
    """A line ``a^T u <= c`` in **plot** coordinates -- market bounds such as a
    generation cap or ``L >= 0``, which restrict the axes rather than any element's
    flow and so are no part of ``Q(b)``.

    On ``(L, q_S)`` axes, ``L >= 0`` is ``a = (-1, 0), c = 0``.
    """
    _draw_line(ax, np.asarray(a, dtype=float), float(c),
               ax.get_xlim(), ax.get_ylim(), label=label, color=color, ls=ls, lw=lw)


def frame_axes(
    ax,
    model: NetworkModel,
    *,
    title: str | None = None,
    xlim=None,
    ylim=None,
    drop: int | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **title_kw,
):
    """Set an axis up to receive the drawing layers: limits, origin lines, axis names,
    optional title.

    Must come first -- :func:`draw_constraints` and :func:`draw_halfplane` clip to
    the current limits.  Extra keywords go to ``set_title``.  ``ylabel=""`` blanks
    the name on the inner column of a shared-y grid.
    """
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.axhline(0, lw=0.4, c="k")
    ax.axvline(0, lw=0.4, c="k")
    label_axes(ax, model, drop=drop, xlabel=xlabel, ylabel=ylabel)
    if title is not None:
        ax.set_title(title, **title_kw)
    return ax


def label_axes(
    ax,
    model: NetworkModel,
    *,
    drop: int | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    """Name the axes.

    Defaults to a plain drop-one-node basis; pass ``xlabel``/``ylabel`` for a basis
    whose axes are combinations rather than single nodes.
    """
    net = model.network
    drop = net.slack_idx if drop is None else drop
    names = net.node_names
    kept = [i for i in range(net.n_nodes) if i != drop % net.n_nodes]
    if xlabel is None:
        xlabel = f"q[{kept[0]}]" if names is None else f"$q_{{{names[kept[0]]}}}$"
    if ylabel is None:
        ylabel = f"q[{kept[1]}]" if names is None else f"$q_{{{names[kept[1]]}}}$"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
