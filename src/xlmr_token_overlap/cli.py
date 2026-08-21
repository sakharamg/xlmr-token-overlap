"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_flores, load_jsonl, prepare_flores, prepare_tokenizer
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

    all_mteb = subparsers.add_parser(
        "all-mteb",
        help="Prepare, run, compare, and validate the pinned MTEB Multilingual v2 pass",
    )
    all_mteb.add_argument("--source-dir", type=_path, default=_path("data"))
    all_mteb.add_argument("--output-dir", type=_path, default=_path("results/mteb"))
    all_mteb.add_argument("--flores-dir", type=_path, default=_path("results/flores"))
    all_mteb.add_argument("--family-token-budget", type=int, default=20_000)
    all_mteb.add_argument("--overall-token-budget", type=int, default=30_000)
    all_mteb.add_argument("--seed", type=int, default=1729)

    all_pass3 = subparsers.add_parser(
        "all-pass3",
        help="Run the full local translated-STS and Belebele Pass-3 suite",
    )
    all_pass3.add_argument("--datasets-root", type=_path, required=True)
    all_pass3.add_argument("--source-dir", type=_path, default=_path("data"))
    all_pass3.add_argument("--tokenizer-json", type=_path)
    all_pass3.add_argument("--output-dir", type=_path, default=_path("results/pass3"))
    all_pass3.add_argument("--flores-dir", type=_path, default=_path("results/flores"))

    all_sqe = subparsers.add_parser(
        "all-sqe",
        help="Run full-text SQE overlap independently by domain and query variant",
    )
    all_sqe.add_argument("--datasets-root", type=_path, required=True)
    all_sqe.add_argument("--source-dir", type=_path, default=_path("data"))
    all_sqe.add_argument("--tokenizer-json", type=_path)
    all_sqe.add_argument("--output-dir", type=_path, default=_path("results/sqe"))
    all_sqe.add_argument("--flores-dir", type=_path, default=_path("results/flores"))
    all_sqe.add_argument(
        "--domains",
        nargs="+",
        choices=(
            "calendar",
            "call_recording",
            "notes",
            "reminder",
            "settings",
            "voice_recording",
        ),
        help="Optional domain subset; default analyzes every SQE domain.",
    )
    all_sqe.add_argument(
        "--allow-coverage-drift",
        action="store_true",
        help="Accept locale/variant coverage different from the audited snapshot.",
    )
    all_sqe.add_argument(
        "--strict-ground-truth",
        action="store_true",
        help=(
            "Fail on empty gt_ids or references missing from data.id; "
            "default records warnings and analyzes every non-empty query."
        ),
    )

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
    elif args.command == "all-mteb":
        if args.family_token_budget <= 0 or args.overall_token_budget <= 0:
            raise ValueError("MTEB token budgets must be positive")
        from .mteb_data import run_mteb

        tokenizer_json = prepare_tokenizer(args.source_dir)
        result = run_mteb(
            tokenizer_json=tokenizer_json,
            cache_dir=args.source_dir / "mteb-cache",
            output_dir=args.output_dir,
            flores_dir=args.flores_dir,
            family_token_budget=args.family_token_budget,
            overall_token_budget=args.overall_token_budget,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2))
    elif args.command in {"all-pass3", "all-sqe"}:
        from .local_data import run_pass3, run_sqe

        tokenizer_json = args.tokenizer_json or prepare_tokenizer(args.source_dir)
        if args.command == "all-pass3":
            result = run_pass3(
                datasets_root=args.datasets_root,
                tokenizer_json=tokenizer_json,
                output_dir=args.output_dir,
                flores_dir=args.flores_dir,
            )
        else:
            result = run_sqe(
                datasets_root=args.datasets_root,
                tokenizer_json=tokenizer_json,
                output_dir=args.output_dir,
                flores_dir=args.flores_dir,
                domains=args.domains,
                allow_coverage_drift=args.allow_coverage_drift,
                strict_ground_truth=args.strict_ground_truth,
            )
        print(json.dumps(result, indent=2))
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
