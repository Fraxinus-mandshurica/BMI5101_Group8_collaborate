"""Package the frozen local MSI inputs and upstream commit for Colab.

This script only copies files and hashes them. It never launches DREAMwalk.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
UPSTREAM = PROJECT / "vendor/DREAMwalk_upstream"
COMMIT = "2f93723c3fbfa899d0f23c02e3459de8bfac0581"
TOP = "DREAMwalk_MSI_seed0_bundle"
ZIP = OUT / "DREAMwalk_MSI_seed0_bundle.zip"
LOCAL_MANIFEST = OUT / "manifest_seed0.json"
RELEVANT = ("DREAMwalk/generate_embeddings.py", "DREAMwalk/HeterogeneousSG.py",
            "DREAMwalk/HeterogeneousSG.cpp", "DREAMwalk/utils.py", "environment.yaml")
INPUTS = (
    ("data/processed/msi_graph.tsv", "input/msi_graph.tsv", "molecular_graph"),
    ("data/processed/msi_similarity_graph_final.txt", "input/msi_similarity_graph_final.txt", "semantic_graph"),
    ("data/processed/msi_nodetypes.tsv", "input/msi_nodetypes.tsv", "node_types"),
    ("data/processed/msi_positive_pairs.tsv", "evaluation/msi_positive_pairs.tsv", "positive_pairs"),
    ("results/msi_repeat0_negative_pairs.csv", "evaluation/msi_repeat0_negative_pairs.csv", "negative_pairs"),
    ("results/msi_repeat0_run_config.json", "provenance/msi_repeat0_run_config.json", "provenance"),
    ("results/msi_semantic_similarity_summary.csv", "provenance/msi_semantic_similarity_summary.csv", "provenance"),
    ("colab/colab_seed0_runner.py", "scripts/colab_seed0_runner.py", "runner"),
)
PROVENANCE_HASH_ONLY = (
    "data/processed/msi_semantic_hierarchy_final.csv",
    "results/msi_atc_mapping_audit.csv",
    "results/msi_cui_mesh_umls_mapping.csv",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(UPSTREAM), *args], text=True).strip()


def main():
    assert git("rev-parse", "HEAD") == COMMIT
    assert not git("diff", "--name-only") and not git("diff", "--cached", "--name-only"), \
        "Tracked upstream source differs from frozen commit"
    original_status = git("status", "--porcelain=v1")
    config = json.loads((PROJECT / "results/msi_repeat0_run_config.json").read_text())
    assert sha256(PROJECT / INPUTS[0][0]) == config["graph"]["graph_sha256"]
    assert sha256(PROJECT / INPUTS[1][0]) == config["dreamwalk"]["similarity_sha256"]
    assert sha256(PROJECT / INPUTS[4][0]) == config["evaluation"]["negative_pairs_sha256"]
    tracked = subprocess.check_output(["git", "-C", str(UPSTREAM), "ls-files", "-z"]).split(b"\0")
    tracked = [Path(x.decode()) for x in tracked if x]
    manifest = {"bundle_name":TOP,"upstream_commit":COMMIT,
                "original_upstream_git_status_porcelain":original_status,
                "bundled_upstream_git_status_expected":"clean",
                "git_metadata_bundled":True,
                "note":"The original checkout has only an untracked __pycache__; tracked source is unchanged. The bundle contains tracked files and Git metadata, excluding that cache.",
                "files":[],
                "provenance_not_bundled":[{"original_local_path":str(PROJECT/name),
                    "size_bytes":(PROJECT/name).stat().st_size,"sha256":sha256(PROJECT/name),
                    "reason":"Final reviewed semantic network is already bundled; mapping reconstruction is not needed at runtime."}
                    for name in PROVENANCE_HASH_ONLY],
                "upstream_relevant_sha256":{name:sha256(UPSTREAM/name) for name in RELEVANT}}
    with tempfile.TemporaryDirectory(prefix="dreamwalk_colab_bundle_") as tmp:
        staging = Path(tmp) / TOP
        for rel in tracked:
            source = UPSTREAM / rel
            dest = staging / "vendor/DREAMwalk_upstream" / rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
            manifest["files"].append({"original_local_path":str(source),
                "bundled_relative_path":str(dest.relative_to(staging)),"size_bytes":source.stat().st_size,
                "sha256":sha256(source),"role":"upstream_tracked_file"})
        shutil.copytree(UPSTREAM / ".git",staging / "vendor/DREAMwalk_upstream/.git")
        for source_rel,bundle_rel,role in INPUTS:
            source = PROJECT / source_rel
            assert source.is_file(),source
            dest = staging / bundle_rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
            manifest["files"].append({"original_local_path":str(source),
                "bundled_relative_path":bundle_rel,"size_bytes":source.stat().st_size,
                "sha256":sha256(source),"role":role})
        (staging/"results").mkdir()
        (staging/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        bundled_status=subprocess.check_output(["git","status","--porcelain=v1"],
            cwd=str(staging/"vendor/DREAMwalk_upstream"),text=True).strip()
        assert not bundled_status,"Bundled upstream checkout not clean: {}".format(bundled_status)
        with zipfile.ZipFile(ZIP,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path in sorted(staging.rglob("*")):
                archive.write(path,arcname=str(path.relative_to(staging.parent)))
    LOCAL_MANIFEST.write_text(json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(ZIP) as archive:
        bad=archive.testzip()
        assert bad is None,"ZIP CRC error in {}".format(bad)
        members=set(archive.namelist())
        for item in manifest["files"]:
            assert TOP+"/"+item["bundled_relative_path"] in members
    print(json.dumps({"zip":str(ZIP),"zip_bytes":ZIP.stat().st_size,
                      "zip_sha256":sha256(ZIP),"tracked_upstream_files":len(tracked),
                      "manifest_files":len(manifest["files"]),
                      "original_upstream_status":original_status,
                      "bundled_upstream_status":bundled_status},indent=2))


if __name__=="__main__":main()
