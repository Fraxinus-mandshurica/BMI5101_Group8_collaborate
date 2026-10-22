"""Launch the unchanged DTi2vec runner after a directory-only relocation.

Historical manifests retain their original paths and fingerprints. Reuse is
allowed only when settings, packages, input bytes and source bytes still match.
No graph, embedding, classifier or evaluation behaviour is changed here.
"""
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
import run_dti2vec as runner

original_context = runner.context


def source_location(path):
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part in {"00.DREAMwalk_reproduction", "02.DTi2vec_reproduction"}:
            return Path(*parts[index:]).as_posix()
    return str(path)


def relocated_context(name):
    manifest = runner.ROOT / "results" / name / "run_config.json"
    if not manifest.is_file():
        return original_context(name)
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("RUN_NAME accepts letters, digits, underscore or hyphen")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Use the notebook-managed Python 3.8 environment")
    saved = json.loads(manifest.read_text())
    paths = [Path(runner.__file__), runner.UPSTREAM / "DREAMwalk/generate_similarity_net.py",
             runner.UPSTREAM / "DREAMwalk/utils.py"]
    current_sources = {source_location(p): runner.sha256(p) for p in paths}
    saved_sources = {source_location(p): sha for p, sha in saved["sources"].items()}
    packages = {p: runner.version(p) for p in runner.PINS}
    inputs = {k: runner.sha256(p) for k, p in runner.INPUTS.items()}
    if (packages != runner.PINS or saved["packages"] != packages
            or saved["settings"] != runner.SETTINGS or saved["inputs"] != inputs
            or current_sources != saved_sources):
        raise RuntimeError("Inputs/code/settings changed; use a new RUN_NAME. Existing outputs are preserved.")
    identity = {key: saved[key] for key in ("settings", "packages", "inputs", "sources")}
    expected = runner.hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if expected != saved["fingerprint"]:
        raise RuntimeError("Historical run manifest fingerprint is inconsistent")
    data, results = runner.ROOT / "data" / name, runner.ROOT / "results" / name
    data.mkdir(parents=True, exist_ok=True)
    print("Reusing unchanged DTi2vec run after directory relocation; historical provenance retained", flush=True)
    return data, results, saved["fingerprint"]


if __name__ == "__main__":
    runner.context = relocated_context
    runner.main()
