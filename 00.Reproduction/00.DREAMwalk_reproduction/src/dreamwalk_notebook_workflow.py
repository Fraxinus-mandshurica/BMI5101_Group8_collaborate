"""Small I/O helpers for the narrative DREAMwalk notebook.

Experimental settings and stage ordering are visible in notebook cells.
No model imports or computation occur on import.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import signal
import subprocess


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary_metrics(path):
    return {row["metric"]: float(row["mean"]) for row in read_csv(path)}


def runner_constants(path, *names):
    """Read static parameters without importing scientific/model modules."""
    def literal(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
            if node.args or any(item.arg is None for item in node.keywords):
                raise ValueError("Only literal keyword dictionaries are supported")
            return {item.arg: literal(item.value) for item in node.keywords}
        return ast.literal_eval(node)

    found = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = literal(node.value)
    missing = set(names) - found.keys()
    if missing:
        raise ValueError("Runner constants not found: " + ", ".join(sorted(missing)))
    return found


def check_parameters(actual, expected):
    """Check that a fixed runner implements the notebook's stated contrast."""
    mismatch = {key: (value, actual.get(key)) for key, value in expected.items()
                if actual.get(key) != value}
    if mismatch:
        raise ValueError("Notebook section and runner parameters differ: " + repr(mismatch) +
                         ". Update the section, runner and run name together for a new parameter contrast.")


def table_complete(table, marker, config):
    """Verify a saved comparison without rewriting its CSV."""
    table, marker, config = Path(table), Path(marker), Path(config)
    if not marker.is_file():
        return False
    meta, run = read_json(marker), read_json(config)
    if (not table.is_file() or meta["fingerprint"] != run["fingerprint"]
            or hashlib.sha256(table.read_bytes()).hexdigest() != meta["sha256"]):
        raise RuntimeError("Saved comparison checkpoint differs: " + str(table))
    return True


def run_phase(root, python, runner, phase, run_name):
    """Launch one visible notebook step with shared logging/cancellation."""
    root = Path(root).resolve()
    python = Path(python).expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError("DREAMwalk interpreter not found: " + str(python))
    if not isinstance(run_name, str) or not run_name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in run_name):
        raise ValueError("Run name accepts only letters, digits, underscores and hyphens.")
    script = root / "src" / runner
    if not script.is_file():
        raise FileNotFoundError(script)
    logs = root / "logs" / run_name
    logs.mkdir(parents=True, exist_ok=True)
    command = [str(python), "-u", str(script), phase, "--run-name", run_name]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    print("Running: ", runner, phase, "|", run_name, flush=True)
    with (logs / (phase + ".log")).open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + repr(command) + "\n")
        with subprocess.Popen(command, cwd=root, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, start_new_session=True) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)


def show_rows(rows, columns=None):
    from IPython.display import HTML, display
    if not rows:
        print("No saved results.")
        return
    columns = columns or list(rows[0])

    def format_value(value):
        if value is None or value == "":
            return "—"
        if isinstance(value, bool):
            return str(value)
        try:
            return f"{float(value):.6f}" if "." in str(value) else str(value)
        except (TypeError, ValueError):
            return str(value)

    header = "".join("<th>" + html.escape(column) + "</th>" for column in columns)
    body = "".join("<tr>" + "".join(
        "<td>" + html.escape(format_value(row.get(column))) + "</td>"
        for column in columns) + "</tr>" for row in rows)
    display(HTML("<table><thead><tr>" + header +
                 "</tr></thead><tbody>" + body + "</tbody></table>"))


def show_csv(path):
    print(Path(path).name)
    show_rows(read_csv(path))
