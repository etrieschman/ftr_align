"""The five-node network of the attribution memo (``fig_texas5``).

Nodes W (western renewables), N (northern renewables), S (southern thermal and
storage), H (eastern load), D (a central data centre / microgrid).  Every
peripheral node connects to D, with additional corridors WN, WS, NH, SH.

Three structures the 3-node cannot show:

* the **parallel WD pair** (``WD1``/``WD2``, equal reactance) -- identical PTDF
  rows, so ``mu`` trades freely between them and ``Lambda*(b;y)`` is genuinely
  non-singleton;
* the **WN / SH pair** -- can be priced together while carrying no circulation
  between them, so their attributed values stay individually identified;
* the **SDH triangle** -- a circulation, so when it also preserves the
  limit-weighted value only the joint attribution is identified.

**No bid data, deliberately.**  Every proposition in the memos is stated at an
arbitrary certificate ``y >= 0``; only ``prop:support`` (the market reading of
``h(g;y*)`` as merchandising surplus) needs ``y`` to come from a clearing, and
the 3-node oracle already anchors that.  So work here by positing a direction
``d`` -- or by designing limits so that a chosen constraint pattern binds, as
``notebooks/explore_texas5.py`` does -- rather than inventing offers whose only
job would be to produce a ``y`` we could have written down.
"""

from __future__ import annotations

import numpy as np

from ..network import PhysicalNetwork

NODE_NAMES = np.array(["W", "N", "S", "D", "H"])
W, N, S, D, H = 0, 1, 2, 3, 4
ELEMENT_NAMES = np.array(["WN", "WS", "WD1", "WD2", "ND", "NH", "SD", "SH", "DH"])
WN, WS, WD1, WD2, ND, NH, SD, SH, DH = 0, 1, 2, 3, 4, 5, 6, 7, 8

# node-branch incidence (node x element), reactances, slack at D
A = np.array(
    [
        [1, 1, 1, 1, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, -1, -1, -1, 0, -1, 0, 1],
        [0, 0, 0, 0, 0, -1, 0, -1, -1],
    ],
    dtype=float,
)
X = np.array([2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0])
BASE_LIMITS = np.full(len(ELEMENT_NAMES), 100.0)
NETWORK = PhysicalNetwork(
    A=A, x=X, slack_idx=D, node_names=NODE_NAMES, element_names=ELEMENT_NAMES
)
