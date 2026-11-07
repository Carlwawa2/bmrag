"""One command-line entrypoint for daily research work."""
from __future__ import annotations

import argparse
import importlib
import json
import sys


def main():
    parser = argparse.ArgumentParser(prog="python -m hgrag", description="Controlled HyperGraphRAG research workspace")
    commands = {
        "smoke": "hgrag.evaluation.smoke",
        "pilot": "hgrag.evaluation.pilot",
        "review": "hgrag.evaluation.review",
        "replay": "hgrag.evaluation.replay",
        "environment": "hgrag.tools.environment",
        "download-model": "hgrag.tools.models",
        "snapshot": "hgrag.tools.snapshot",
    }
    parser.add_argument("command", choices=["verify", "prepare-tokenizer", *commands])
    args, remaining = parser.parse_known_args()
    if args.command == "verify":
        options = argparse.ArgumentParser(prog="python -m hgrag verify")
        options.parse_args(remaining)
        from .configuration import load
        config = load(verify_artifacts=True)
        report = {"status": "passed", "llm": config["models"]["generation"], "embedding": config["models"]["embedding"]["name"], "configuration": "configs/baseline.json"}
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "prepare-tokenizer":
        if remaining:
            parser.error("prepare-tokenizer takes no arguments")
        from .models.tokenizer import load_tokenizer
        print(f"Prepared {load_tokenizer(download=True).name}")
    else:
        sys.argv = [f"python -m hgrag {args.command}", *remaining]
        importlib.import_module(commands[args.command]).main()


if __name__ == "__main__":
    main()
