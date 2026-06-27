"""RTS-GMLC loader tests.  These need the pinned CSVs; if the cache is empty and
the machine is offline the suite skips, keeping the toy oracle tests hermetic."""

from __future__ import annotations

import numpy as np
import pytest

from ftr_align.cases import rts_gmlc as rts
from ftr_align.network import NetworkModel
from ftr_align.solve import SupportProblem, clear_dam


@pytest.fixture(scope="module")
def net():
    try:
        return rts.load_network()
    except OSError as exc:  # URLError etc. with an empty cache
        pytest.skip(f"RTS-GMLC data unavailable (offline, empty cache): {exc}")


@pytest.fixture(scope="module")
def conts(net):
    return rts.n1_contingencies(net, verbose=False)


# --- network geometry -------------------------------------------------------
def test_network_invariants(net):
    assert net.n_nodes == 73 and net.n_elements == 120
    assert set(np.unique(net.inc).tolist()) <= {-1.0, 0.0, 1.0}
    # each branch column has exactly one +1 and one -1 (two distinct endpoints)
    assert np.all((net.inc == 1.0).sum(axis=0) == 1)
    assert np.all((net.inc == -1.0).sum(axis=0) == 1)
    assert np.all(net.x > 0.0)
    assert net.node_names.shape == (73,) and net.element_names.shape == (120,)


def test_single_ref_bus(net):
    bus, _, _ = rts.bus_table()
    assert (bus["Bus Type"].to_numpy() == "Ref").sum() == 1
    assert bus["Bus Type"].to_numpy()[net.slack_idx] == "Ref"


def test_bus_index_roundtrip(net):
    _, ids, id_to_row = rts.bus_table()
    assert all(ids[id_to_row[int(b)]] == b for b in ids)
    assert list(ids) == sorted(ids)  # canonical order is sorted Bus ID


def test_taps_default_off_transformers(net):
    branch = rts._read_csv(rts._BRANCH)
    tr = branch["Tr Ratio"].to_numpy().astype(float)
    assert np.allclose(net.tap[tr == 0.0], 1.0)  # non-transformers -> ratio 1
    assert np.all(net.tap[tr > 0.0] == tr[tr > 0.0])
    assert (net.tap != 1.0).any()  # RTS does have transformers


def test_ptdf_slack_column_zero(net):
    # injection at the slack is the reference -> zero sensitivity
    assert np.allclose(net.ptdf(None)[:, net.slack_idx], 0.0)


# --- connectivity / contingencies ------------------------------------------
def test_bridges_skipped(net, conts):
    from ftr_align.network import is_connected

    bridges = [i for i in range(net.n_elements) if not is_connected(net.inc, i)]
    assert bridges, "RTS-GMLC has radial branches"
    keys = {c.key for c in conts}
    assert all(b not in keys for b in bridges)
    # one base case (None) + one per non-bridge branch
    assert len(conts) == 1 + (net.n_elements - len(bridges))


def test_ptdf_disconnected_raises(net):
    from ftr_align.network import is_connected

    bridge = next(i for i in range(net.n_elements) if not is_connected(net.inc, i))
    with pytest.raises(ValueError, match="disconnected"):
        net.ptdf(bridge)


def test_outage_row_unmonitored(net, conts):
    cont = next(c for c in conts if c.key is not None)
    assert np.isinf(cont.upper[cont.key])  # outaged element carries no flow
    assert np.isfinite(cont.upper[: cont.key]).all()  # others monitored at LTE


# --- DAM instance + end-to-end ---------------------------------------------
def test_interval_lookup(net):
    assert rts.interval_index(1, 1, 1) == 0


def test_dam_instance_shapes(net):
    inst = rts.dam_instance(0, net)
    assert inst.M_gen.shape[0] == net.n_nodes
    assert inst.M_gen.shape[1] == inst.max_gen.shape[0] == inst.p_gen.shape[0]
    assert (inst.min_gen == 0.0).all()  # PMin relaxed
    assert (inst.p_gen >= 0.0).all() and (inst.p_gen == 0.0).any()  # renewables free
    assert inst.q_dem.sum() > 0.0 and inst.max_gen.sum() > inst.q_dem.sum()


def test_end_to_end(net, conts):
    model = NetworkModel.build(net, conts)
    inst = rts.dam_instance(0, net)
    r = clear_dam(model, inst, solver={"solver": "CLARABEL"})
    assert r.status == "optimal"
    sp = SupportProblem(model, r.direction).solve(solver={"solver": "CLARABEL"})
    assert np.isfinite(sp.value)
    # support value at the DAM's own congestion direction is the congestion rent
    assert sp.value == pytest.approx(r.merch_surp, rel=1e-6)
