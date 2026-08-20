"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_flores, load_jsonl, prepare_flores
from .pipeline import run_analysis
from .validation import validate_results


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xlmr-token-overlap",
        description="Reproducible XLM-R token-overlap matrices across 24 languages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-flores", help="Download pinned FLORES/XLM-R inputs")
    prepare.add_argument("--source-dir", type=_path, default=_path("data"))

    run_flores = subparsers.add_parser("run-flores", help="Analyze aligned FLORES dev + devtest")
    run_flores.add_argument("--flores-root", type=_path, required=True)
    run_flores.add_argument("--tokenizer-json", type=_path, required=True)
    run_flores.add_argument("--output-dir", type=_path, default=_path("results/flores"))
    run_flores.add_argument(
        "--splits",
        nargs="+",
        choices=("dev", "devtest"),
        default=("dev", "devtest"),
    )

    all_flores = subparsers.add_parser(
        "all-flores", help="Prepare, run, and validate the pinned FLORES pass"
    )
    all_flores.add_argument("--source-dir", type=_path, default=_path("data"))
    all_flores.add_argument("--output-dir", type=_path, default=_path("results/flores"))

    run_jsonl = subparsers.add_parser(
        "run-jsonl", help="Analyze Pass-2/3 interchange rows by independent condition"
    )
    run_jsonl.add_argument("--input", type=_path, required=True)
    run_jsonl.add_argument("--tokenizer-json", type=_path, required=True)
    run_jsonl.add_argument("--output-dir", type=_path, required=True)
    run_jsonl.add_argument(
        "--allow-partial-languages",
        action="store_true",
        help="Permit conditions with fewer than the frozen 24 languages (diagnostics only)",
    )

    validate = subparsers.add_parser("validate", help="Re-check all matrix invariants")
    validate.add_argument("results_dir", type=_path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-flores":
        paths = prepare_flores(args.source_dir)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    elif args.command == "run-flores":
        records, provenance = load_flores(args.flores_root, args.splits)
        run_analysis(records, args.tokenizer_json, args.output_dir, provenance)
    elif args.command == "all-flores":
        paths = prepare_flores(args.source_dir)
        records, provenance = load_flores(paths["flores_root"])
        run_analysis(records, paths["tokenizer_json"], args.output_dir, provenance)
    elif args.command == "run-jsonl":
        records, provenance = load_jsonl(args.input)
        run_analysis(
            records,
            args.tokenizer_json,
            args.output_dir,
            provenance,
            require_all_languages=not args.allow_partial_languages,
        )
    elif args.command == "validate":
        report = validate_results(args.results_dir)
        print(json.dumps(report, indent=2))
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)

