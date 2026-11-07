"""Restore or verify the isolated Windows environment without PowerShell scripts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ..common.files import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true", help="Install the locked environment; default only verifies")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise SystemExit("This lock profile is for Windows x86-64")
    manager = shutil.which("uv")
    if not manager:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/getting-started/installation/")
    version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    cache = ROOT / "artifacts/toolchain/uv-cache"
    installs = ROOT / "artifacts/toolchain/python"
    base = installs / f"cpython-{version}-windows-x86_64-none/python.exe"
    venv = ROOT / ".venv"
    python = venv / "Scripts/python.exe"
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "42"
    environment["UV_CACHE_DIR"] = str(cache)
    environment["UV_PYTHON_INSTALL_DIR"] = str(installs)
    environment["TIKTOKEN_CACHE_DIR"] = str(ROOT / "artifacts/tokenizer-cache")

    def run(*command):
        subprocess.run([str(part) for part in command], cwd=ROOT, env=environment, check=True)

    if args.restore:
        if not base.is_file():
            run(manager, "python", "install", version, "--install-dir", installs, "--cache-dir", cache, "--no-bin", "--no-registry", "--no-progress")
        if not python.is_file():
            run(manager, "venv", venv, "--python", base, "--cache-dir", cache, "--no-python-downloads", "--no-progress")
    if not python.is_file():
        raise SystemExit("The project .venv is missing; run with --restore to create it")
    observed = subprocess.check_output([str(python), "--version"], text=True).strip()
    if observed != f"Python {version}":
        raise SystemExit(f"Existing .venv uses {observed}; expected Python {version}. No environment was deleted.")
    if args.restore:
        run(manager, "pip", "sync", ROOT / "environment/requirements-win-py311.lock.txt", "--python", python, "--torch-backend", "cpu", "--require-hashes", "--strict", "--only-binary", ":all:", "--cache-dir", cache, "--no-progress")
    run(manager, "pip", "check", "--python", python, "--cache-dir", cache, "--no-progress")
    run(python, "-m", "hgrag.evaluation.environment")


if __name__ == "__main__":
    main()
