"""Importable worker wrapper for the FIRST-HIT teleport selector diagnostic.

The vendor source is unchanged. Each worker installs the selector itself, so
the diagnostic also works when multiprocessing starts workers with ``spawn``.
No training or walk generation runs when this module is imported.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/DREAMwalk_upstream"))
from DREAMwalk import generate_embeddings as ge

SELECTOR_NAME = "teleport_selector_diagnostic.teleport_operation_fixed"
_SELECTOR_CALLS = 0


def teleport_operation_fixed(cur, G_sim):
    """Upstream selector, with only first-hit return replacing inner break.

Keep the shuffle, set iteration, parallel-edge accumulation and NumPy draw.
The counter records actual use without consuming random numbers.
"""
    global _SELECTOR_CALLS
    _SELECTOR_CALLS += 1
    cur_nbrs = sorted(G_sim.neighbors(cur))
    ge.random.shuffle(cur_nbrs)
    selected_nbrs = []
    distance_sum = 0
    for nbr in cur_nbrs:
        nbr_links = G_sim[cur][nbr]
        for i in nbr_links:
            nbr_link_weight = nbr_links[i]["weight"]
            distance_sum += nbr_link_weight
            selected_nbrs.append(nbr)
    if distance_sum == 0:
        return False
    rand = ge.np.random.rand() * distance_sum
    threshold = 0
    for nbr in set(selected_nbrs):
        nbr_links = G_sim[cur][nbr]
        for i in nbr_links:
            nbr_link_weight = nbr_links[i]["weight"]
            threshold += nbr_link_weight
            if threshold >= rand:
                next_node = nbr
                return next_node  # DIAGNOSTIC CHANGE: upstream breaks only the inner loop.
    return next_node  # Preserve upstream failure if no threshold is reached.


def install_fixed_selector():
    ge._teleport_operation = teleport_operation_fixed


def worker_walks(*args):
    """Run the original worker with the fixed selector; return use evidence."""
    install_fixed_selector()
    before = _SELECTOR_CALLS
    walks = ge._parmap_walks(*args)
    audit = dict(task_id=int(args[0]), pid=os.getpid(), parent_pid=os.getppid(),
                 selector=SELECTOR_NAME,
                 selector_is_fixed=ge._teleport_operation is teleport_operation_fixed,
                 selector_calls=_SELECTOR_CALLS - before,
                 selector_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                 rng_state_claim="Worker Python/NumPy RNG states are not asserted")
    return walks, audit
