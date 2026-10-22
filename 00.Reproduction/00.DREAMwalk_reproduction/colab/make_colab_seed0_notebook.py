"""Write the user-run Colab notebook without executing any notebook cell."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "DREAMwalk_MSI_seed0_Colab.ipynb"


def md(source):
    return {"cell_type":"markdown","metadata":{},"source":source.splitlines(True)}


def code(source):
    return {"cell_type":"code","execution_count":None,"metadata":{},
            "outputs":[],"source":source.splitlines(True)}


cells = [
md("""# DREAMwalk MSI semantic seed 0 — full Colab/Linux run

**Prepared, not executed.** Select a standard Linux CPU runtime, then choose **Runtime → Run all** starting at Section 1. No GPU is required. This notebook verifies the frozen inputs and author's package pins, then runs exactly one full 2,995,900-walk semantic model. There is no smoke or reduced run.

Frozen choice: cutoff 0.4, teleport factor 0.3, 100 walks per node, walk length 10, 128 dimensions, window 4, `p=q=1`, up to 5 EM iterations, 4 workers, seed 0. Teleport factor 0.3 is a best-supported Figure 2a reconstruction choice, not an explicitly reported Figure 2a setting. The exact authors' released environment file is used. The notebook never sets `PYTHONHASHSEED` externally or changes multiprocessing start method.

Outputs are checkpointed to `MyDrive/DREAMwalk_MSI_seed0_Colab/results/`. A Colab runtime disconnection can still interrupt the full run; the persistent log shows the last completed stage.
"""),
md("## 1 — Mount Drive and prepare a local `/content` workspace"),
code(r"""from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import hashlib, json, os, platform, shutil, subprocess, sys, zipfile

DRIVE_DIR = Path('/content/drive/MyDrive/DREAMwalk_MSI_seed0_Colab')
BUNDLE_ZIP = DRIVE_DIR / 'DREAMwalk_MSI_seed0_bundle.zip'
DRIVE_RESULTS = DRIVE_DIR / 'results'
WORKSPACE = Path('/content/DREAMwalk_MSI_seed0_run')
ROOT = WORKSPACE / 'DREAMwalk_MSI_seed0_bundle'
assert BUNDLE_ZIP.is_file(), f'Missing run bundle: {BUNDLE_ZIP}'
assert not WORKSPACE.exists(), ('Local workspace already exists. Restart the Colab runtime '
                                'before a clean Run all; this notebook will not overwrite an earlier run.')
DRIVE_RESULTS.mkdir(parents=True, exist_ok=True)
WORKSPACE.mkdir()
LOCAL_ZIP = WORKSPACE / BUNDLE_ZIP.name
shutil.copy2(BUNDLE_ZIP, LOCAL_ZIP)
with zipfile.ZipFile(LOCAL_ZIP) as archive:
    assert archive.testzip() is None, 'Bundle ZIP CRC verification failed'
    for member in archive.namelist():
        path = Path(member)
        assert not path.is_absolute() and '..' not in path.parts, f'Unsafe ZIP path: {member}'
    archive.extractall(WORKSPACE)
assert (ROOT / 'manifest.json').is_file()
(ROOT / 'results').mkdir(exist_ok=True)
# Older bundle ZIPs omit empty directories; Git requires .git/refs to recognize the checkout.
(ROOT / 'vendor/DREAMwalk_upstream/.git/refs').mkdir(exist_ok=True)
RUNNER = ROOT / 'scripts/colab_seed0_runner.py'
print('Local workspace:', ROOT)
print('Drive results:', DRIVE_RESULTS)
print('Notebook cells have not started DREAMwalk yet.')
"""),
md("## 2 — Record the Colab/Linux environment"),
code(r"""import datetime
safe_colab_metadata = {key: os.environ[key] for key in ('COLAB_GPU', 'COLAB_RELEASE_TAG') if key in os.environ}
environment_lines = [f'UTC time: {datetime.datetime.now(datetime.timezone.utc).isoformat()}',
                     f'Colab metadata: {json.dumps(safe_colab_metadata, sort_keys=True)}']
commands = [
    ['date', '-u'], ['uname', '-a'], ['cat', '/etc/os-release'],
    ['uname', '-m'], ['nproc'], ['free', '-h'], ['df', '-h', '/content'],
    ['g++', '--version'], [sys.executable, '--version'],
]
for command in commands:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=True)
    environment_lines += ['$ ' + ' '.join(command), result.stdout.rstrip()]
cpuinfo = Path('/proc/cpuinfo').read_text()
model = next((line for line in cpuinfo.splitlines() if line.lower().startswith('model name')), 'unavailable')
environment_lines.append(model)
ENVIRONMENT_TXT = ROOT / 'results/colab_environment.txt'
ENVIRONMENT_TXT.write_text('\n'.join(environment_lines) + '\n')
shutil.copy2(ENVIRONMENT_TXT, DRIVE_RESULTS / ENVIRONMENT_TXT.name)
print(ENVIRONMENT_TXT.read_text())
"""),
md("## 3 — Create Python 3.8 from the frozen upstream `environment.yaml`"),
code(r"""assert platform.system() == 'Linux', 'This notebook requires a Linux Colab runtime'
assert platform.machine() == 'x86_64', ('Expected Colab linux-64/x86_64; stopping instead of '
                                       'silently using a different package platform')
MICROMAMBA_TAR = WORKSPACE / 'micromamba.tar.bz2'
MICROMAMBA = WORKSPACE / 'bin/micromamba'
subprocess.run(['curl', '-fLsS', 'https://micro.mamba.pm/api/micromamba/linux-64/latest',
                '-o', str(MICROMAMBA_TAR)], check=True)
subprocess.run(['tar', '-xjf', str(MICROMAMBA_TAR), '-C', str(WORKSPACE),
                'bin/micromamba'], check=True)
assert MICROMAMBA.is_file()
ENV_PREFIX = WORKSPACE / 'dreamwalk-env'
MAMBA_ENV = os.environ.copy()
MAMBA_ENV['MAMBA_ROOT_PREFIX'] = str(WORKSPACE / 'micromamba-root')
subprocess.run([str(MICROMAMBA), 'create', '-y', '-p', str(ENV_PREFIX), '-f',
                str(ROOT / 'vendor/DREAMwalk_upstream/environment.yaml')],
               env=MAMBA_ENV, check=True)
PYTHON = ENV_PREFIX / 'bin/python'
assert PYTHON.is_file(), 'Pinned Python environment was not created'
version_check = r'''import sys
from importlib import metadata
expected = {'networkx':'2.8.8','pandas':'1.4.4','scikit-learn':'1.1.3',
            'scipy':'1.9.1','numpy':'1.23.3','tqdm':'4.64.1','parmap':'1.6.0',
            'xgboost':'1.7.4'}
assert sys.version_info[:2] == (3,8), sys.version
for name, pin in expected.items():
    actual = metadata.version(name)
    assert actual == pin, f'{name}: expected {pin}, got {actual}'
    print(name, actual)
print('python', sys.version)
'''
subprocess.run([str(PYTHON), '-c', version_check], check=True)
print('Pinned environment:', ENV_PREFIX)
"""),
md("## 4 — Verify the frozen upstream checkout"),
code(r"""UPSTREAM = ROOT / 'vendor/DREAMwalk_upstream'
commit = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
status = subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain=v1'], text=True).strip()
assert commit == '2f93723c3fbfa899d0f23c02e3459de8bfac0581'
assert not status, f'Bundled upstream checkout is not clean: {status}'
manifest = json.loads((ROOT / 'manifest.json').read_text())
for relpath, expected_hash in manifest['upstream_relevant_sha256'].items():
    actual_hash = hashlib.sha256((UPSTREAM / relpath).read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f'Upstream hash mismatch: {relpath}'
    print(relpath, actual_hash)
print('Upstream commit:', commit)
print('Upstream git status: clean')
"""),
md("## 5 — Verify manifest, graph, labels, semantic network, and pinned packages"),
code(r"""subprocess.run([str(PYTHON), str(RUNNER), 'verify'], cwd=str(ROOT), check=True)
for name in ('colab_seed0_run_config.json', 'colab_input_manifest.json'):
    shutil.copy2(ROOT / 'results' / name, DRIVE_RESULTS / name)
verified_config = json.loads((ROOT / 'results/colab_seed0_run_config.json').read_text())
assert verified_config['inputs']['molecular_edge_rows'] == 478728
assert verified_config['inputs']['known_treatment_edges_in_graph'] == 0
assert verified_config['inputs']['semantic_edge_counts'] == {'drug':27530, 'disease':25573}
print('All canonical hashes and biological input invariants passed.')
print('Full model has not started.')
"""),
md("## 6 — Record multiprocessing start method without overriding it"),
code(r"""subprocess.run([str(PYTHON), '-c',
                'import multiprocessing as mp; print("multiprocessing start method:", mp.get_start_method())'],
               check=True)
print('Notebook external PYTHONHASHSEED assignment: none')
"""),
md("## 7 — FULL EMBEDDING: 2,995,900 molecular/semantic walks\n\n**This is the first cell that starts the full DREAMwalk computation.** It calls the unmodified upstream `save_embedding_files` with the frozen seed-0 parameters. Output is streamed to both `/content` and a persistent Drive log. The cell will not create a second embedding or overwrite an existing output."),
code(r"""EMBEDDING = ROOT / 'results/msi_seed0_colab_linux.pkl'
assert not EMBEDDING.exists(), 'Embedding already exists; refusing to overwrite'
LOCAL_LOG = ROOT / 'results/colab_full_run.log'
DRIVE_LOG = DRIVE_RESULTS / 'colab_full_run.log'
assert not DRIVE_LOG.exists(), ('Drive full-run log already exists. Use a fresh results folder '
                                'for a new run; this notebook prepares exactly one full model.')
with subprocess.Popen([str(PYTHON), '-u', str(RUNNER), 'embed'], cwd=str(ROOT),
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, bufsize=1) as process:
    with LOCAL_LOG.open('w', buffering=1) as local_log, DRIVE_LOG.open('w', buffering=1) as drive_log:
        for line in process.stdout:
            print(line, end='', flush=True)
            local_log.write(line)
            drive_log.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f'Full DREAMwalk process failed with exit code {returncode}; inspect {DRIVE_LOG}')
assert EMBEDDING.is_file(), 'Full run returned without an embedding file'
shutil.copy2(EMBEDDING, DRIVE_RESULTS / EMBEDDING.name)
shutil.copy2(ROOT / 'results/colab_embedding_runtime.json',
             DRIVE_RESULTS / 'colab_embedding_runtime.json')
print('Full embedding finished and checkpointed to Drive:', DRIVE_RESULTS / EMBEDDING.name)
"""),
md("## 8 — Check full embedding integrity before supervised evaluation"),
code(r"""subprocess.run([str(PYTHON), str(RUNNER), 'integrity'], cwd=str(ROOT), check=True)
shutil.copy2(ROOT / 'results/colab_embedding_integrity.json',
             DRIVE_RESULTS / 'colab_embedding_integrity.json')
print('Embedding integrity passed; evaluation may proceed.')
"""),
md("## 9 — Evaluate on the original saved positives/negatives and deterministic 10 folds"),
code(r"""subprocess.run([str(PYTHON), '-u', str(RUNNER), 'evaluate'], cwd=str(ROOT), check=True)
for name in ('colab_seed0_fold_results.csv', 'colab_seed0_summary.json',
             'colab_seed0_run_config.json'):
    shutil.copy2(ROOT / 'results' / name, DRIVE_RESULTS / name)
print('Ten-fold DDA evaluation finished and checkpointed.')
"""),
md("## 10 — Compare Colab, local seed 0, and paper Figure 2a"),
code(r"""summary = json.loads((ROOT / 'results/colab_seed0_summary.json').read_text())
print(f"{'Metric':<10} {'Local':>12} {'Colab':>12} {'Paper':>12} {'Colab-local':>14} {'Colab-paper':>14}")
for metric in ('accuracy', 'auroc', 'aupr'):
    row = summary['comparison'][metric]
    print(f"{metric:<10} {row['local_semantic_seed0_10fold']:>12.6f} "
          f"{row['colab_semantic_seed0_10fold']:>12.6f} {row['paper_figure2a_mean']:>12.6f} "
          f"{row['colab_minus_local']:>14.6f} {row['colab_minus_paper']:>14.6f}")
    print(' ' * 10 + f"Colab 10-fold SD: {summary['metrics'][metric]['sd']:.6f}")
print('\nCaveats:')
for caveat in summary['reconstruction_caveats']:
    print('-', caveat)
print('Fold IDs were deterministically regenerated from the local 10-fold implementation.')
"""),
md("## 11 — Verify final Drive persistence and embedding SHA-256"),
code(r"""def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

required = ['msi_seed0_colab_linux.pkl', 'colab_seed0_fold_results.csv',
            'colab_seed0_summary.json', 'colab_seed0_run_config.json',
            'colab_environment.txt', 'colab_input_manifest.json',
            'colab_embedding_integrity.json', 'colab_full_run.log',
            'colab_embedding_runtime.json']
for name in required:
    local = ROOT / 'results' / name
    remote = DRIVE_RESULTS / name
    assert local.is_file(), f'Missing local output: {local}'
    if not remote.is_file():
        shutil.copy2(local, remote)
    assert file_sha256(local) == file_sha256(remote), f'Drive copy mismatch: {name}'
    print(name, remote.stat().st_size, 'bytes; SHA256 verified')
embedding_hash = file_sha256(EMBEDDING)
(DRIVE_RESULTS / 'colab_embedding_sha256.txt').write_text(embedding_hash + '  ' + EMBEDDING.name + '\n')
print('Final embedding SHA256:', embedding_hash)
print('All required outputs are present in', DRIVE_RESULTS)
"""),
]

for index, cell in enumerate(cells, 1):
    cell["id"] = "dreamwalk-seed0-{:02d}".format(index)

notebook = {"cells":cells,"metadata":{"colab":{"name":OUT.name,"provenance":[]},
           "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
           "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
OUT.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
print(OUT)
