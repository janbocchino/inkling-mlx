"""Command-line tools that keep analysis separate from optional model loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("summarize", help="Recompute results without loading a model")
    analyze.add_argument("results", type=Path)
    analyze.add_argument("--output", type=Path)
    verify = commands.add_parser("verify-model", help="Check saved tensor signatures without MLX")
    verify.add_argument("model", type=Path)
    frozen = commands.add_parser("verify-frozen", help="Check archived experiment code")
    frozen.add_argument("repository", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    try:
        if args.command == "summarize":
            from .results import summarize
            result = summarize(args.results)
        elif args.command == "verify-model":
            from .verify import verify_model
            result = verify_model(args.model.expanduser())
        else:
            from .verify import verify_frozen
            result = verify_frozen(args.repository)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.command == "summarize" and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
