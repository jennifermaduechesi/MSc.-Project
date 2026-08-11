"""Command-line entry points.

    python -m lagos_flood demo            # synthetic end-to-end run
    python -m lagos_flood labels-sweep    # label-threshold sensitivity table
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .config import DEFAULT, Config, LabelConfig, ValidationConfig
from .labels import threshold_sensitivity
from .pipeline import compare_models, prepare_dataset, run_experiment


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the full pipeline on synthetic data."""
    from .data.synthetic import make_synthetic_panel

    print("Generating synthetic Lagos panel (NOT real data — plumbing check only)\n")
    dynamic, static = make_synthetic_panel(seed=args.seed)

    config = Config(
        label=LabelConfig(strategy=args.label_strategy, per_lga=args.per_lga),
        validation=ValidationConfig(
            test_start=args.test_start, n_folds=args.folds, lead_time_days=args.lead_time
        ),
    )

    panel, features = prepare_dataset(dynamic, static, config)
    print(f"Panel: {len(panel):,} rows × {len(features)} features")
    print(f"Labelling: {config.label.strategy} (per-LGA={config.label.per_lga})")
    print(f"Lead time: {config.validation.lead_time_days} day(s) | embargo: {config.embargo_days} days\n")

    if args.compare:
        table, _ = compare_models(panel, features, config=config)
        print(table.round(4).to_string(index=False))
    else:
        result = run_experiment(panel, features, args.model, config)
        print(result.report())

        if args.shap:
            try:
                from .explain import compare_to_literature, compute_shap

                print("\n=== SHAP ===")
                shap_result = compute_shap(
                    result.fitted_model, result.test_frame[features], max_samples=2000
                )
                print(compare_to_literature(shap_result))
            except ImportError as exc:
                print(f"\nSHAP skipped: {exc}")

    print("\nReminder: these numbers come from synthetic data and mean nothing.")
    return 0


def cmd_labels_sweep(args: argparse.Namespace) -> int:
    """Show how the class balance responds to the labelling choice."""
    from .data.synthetic import make_synthetic_panel

    dynamic, _ = make_synthetic_panel(seed=args.seed)
    candidates = {
        "absolute_default": LabelConfig(strategy="absolute", absolute_cuts=(0.01, 0.05, 0.15)),
        "absolute_strict": LabelConfig(strategy="absolute", absolute_cuts=(0.02, 0.10, 0.25)),
        "absolute_loose": LabelConfig(strategy="absolute", absolute_cuts=(0.005, 0.02, 0.08)),
        "pct_80_95_99": LabelConfig(strategy="percentile", percentile_cuts=(0.80, 0.95, 0.99)),
        "pct_70_90_98": LabelConfig(strategy="percentile", percentile_cuts=(0.70, 0.90, 0.98)),
        "pct_per_lga": LabelConfig(strategy="percentile", percentile_cuts=(0.80, 0.95, 0.99), per_lga=True),
    }
    table = threshold_sensitivity(dynamic, candidates)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(table.to_string())
    print(
        "\nReport this table in the methodology chapter: it separates what the class "
        "distribution says about Lagos from what it says about where you put the cuts."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lagos_flood", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="log pipeline progress")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the pipeline end to end on synthetic data")
    demo.add_argument("--model", default="random_forest")
    demo.add_argument("--compare", action="store_true", help="compare RF, XGBoost and LightGBM")
    demo.add_argument("--shap", action="store_true", help="compute SHAP attributions")
    demo.add_argument("--label-strategy", default="absolute", choices=["absolute", "percentile"])
    demo.add_argument("--per-lga", action="store_true", help="fit thresholds per LGA")
    demo.add_argument("--test-start", default=DEFAULT.validation.test_start)
    demo.add_argument("--folds", type=int, default=DEFAULT.validation.n_folds)
    demo.add_argument("--lead-time", type=int, default=DEFAULT.validation.lead_time_days)
    demo.add_argument("--seed", type=int, default=42)
    demo.set_defaults(func=cmd_demo)

    sweep = sub.add_parser("labels-sweep", help="label-threshold sensitivity table")
    sweep.add_argument("--seed", type=int, default=42)
    sweep.set_defaults(func=cmd_labels_sweep)

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
