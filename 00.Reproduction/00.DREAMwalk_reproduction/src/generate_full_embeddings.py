"""Explicitly review-gated wrapper around the original DREAMwalk embedding code."""

import argparse
import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/DREAMwalk_upstream"
sys.dont_write_bytecode = True
sys.path.insert(0, str(UPSTREAM))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", action="store_true", help="Acknowledge the user review gate has cleared")
    parser.add_argument("--hierarchy", required=True, type=Path, help="Reviewed combined drug ATC and disease MeSH child,parent CSV")
    parser.add_argument("--walk-length", required=True, type=int, choices=(4, 10), help="Resolve main-paper versus supplement discrepancy")
    parser.add_argument("--tp-factor", required=True, type=float, help="Resolve unspecified Figure 2a teleport factor")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.reviewed:
        parser.error("Full MSI embedding generation is held at the user's review gate; --reviewed is required after approval")
    if not args.hierarchy.is_file():
        parser.error("Reviewed ATC+MeSH hierarchy CSV is missing")
    if not 0 <= args.tp_factor <= 1:
        parser.error("--tp-factor must be between 0 and 1")
    graph = ROOT / "data/processed/msi_graph.tsv"
    nodetypes = ROOT / "data/processed/msi_nodetypes.tsv"
    if not graph.is_file() or not nodetypes.is_file():
        parser.error("Run src/prepare_msi.py before embedding generation")
    work = ROOT / "logs/full_embedding"
    work.mkdir(parents=True, exist_ok=True)
    code_link = work / "DREAMwalk"
    if not code_link.exists():
        code_link.symlink_to(UPSTREAM / "DREAMwalk", target_is_directory=True)
    os.chdir(work)
    os.environ["CPATH"] = "/Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk/usr/include/c++/v1"
    from DREAMwalk.generate_similarity_net import save_sim_graph
    from DREAMwalk.generate_embeddings import save_embedding_files
    from DREAMwalk.utils import read_graph
    digest = hashlib.sha256(args.hierarchy.read_bytes()).hexdigest()[:12]
    similarity = ROOT / f"data/processed/msi_similarity_{digest}.tsv"
    if not similarity.exists():
        save_sim_graph(str(graph), str(args.hierarchy), str(similarity), cutoff=0.4)
    sim = read_graph(str(similarity))
    types = {}
    with nodetypes.open() as f:
        next(f)
        for line in f:
            node, kind = line.rstrip("\n").split("\t")
            types[node] = kind
    sim_types = {types.get(n) for n in sim.nodes()}
    if not {"drug", "disease"} <= sim_types or not set(sim.nodes()) <= set(types):
        raise RuntimeError("Full semantic graph must contain valid drug and disease nodes")
    output = ROOT / f"data/embeddings/msi_seed_{args.seed}.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_embedding_files(netf=str(graph), sim_netf=str(similarity), outputf=str(output),
                         nodetypef=str(nodetypes), tp_factor=args.tp_factor,
                         seed=args.seed, num_walks=100, walk_length=args.walk_length,
                         dimension=128, window_size=4, workers=args.workers,
                         em_max_iter=5, p=1, q=1)


if __name__ == "__main__":
    main()
