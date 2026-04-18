"""CLI: validate metric gates on a model version and set a Model Registry alias (e.g. champion)."""

from __future__ import annotations

import argparse
import os
import sys

from mlflow import MlflowClient

from src.mlops.registry_constants import CHAMPION_ALIAS, DEFAULT_REGISTERED_MODEL_NAME


def metric_gates_pass(
    metrics: dict[str, float],
    min_auprc: float | None,
    min_f1: float | None,
) -> tuple[bool, list[str]]:
    """Return (ok, error_messages) for optional minimum ``auprc`` and ``f1`` (training run metrics)."""
    errors: list[str] = []
    if min_auprc is not None:
        val = metrics.get("auprc")
        if val is None:
            errors.append("Run is missing metric 'auprc' (required for --min-auprc).")
        elif val < min_auprc:
            errors.append(f"auprc={val} < --min-auprc={min_auprc}")
    if min_f1 is not None:
        val = metrics.get("f1")
        if val is None:
            errors.append("Run is missing metric 'f1' (required for --min-f1).")
        elif val < min_f1:
            errors.append(f"f1={val} < --min-f1={min_f1}")
    return (len(errors) == 0, errors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a model version by assigning a Model Registry alias after optional gates.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (or set MLFLOW_TRACKING_URI).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_REGISTERED_MODEL_NAME,
        help="Registered model name.",
    )
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="Model version number to assign the alias to.",
    )
    parser.add_argument(
        "--alias",
        default=CHAMPION_ALIAS,
        help="Registry alias to set (default: champion).",
    )
    parser.add_argument(
        "--min-auprc",
        type=float,
        default=None,
        help="Require training run metric auprc >= this value.",
    )
    parser.add_argument(
        "--min-f1",
        type=float,
        default=None,
        help="Require training run metric f1 >= this value.",
    )
    args = parser.parse_args()

    if not args.tracking_uri:
        print("error: set --tracking-uri or MLFLOW_TRACKING_URI", file=sys.stderr)
        sys.exit(2)

    client = MlflowClient(args.tracking_uri)
    mv = client.get_model_version(args.model_name, str(args.version))
    run = client.get_run(mv.run_id)
    metrics = dict(run.data.metrics)

    ok, errs = metric_gates_pass(metrics, args.min_auprc, args.min_f1)
    if not ok:
        for msg in errs:
            print(f"gate failed: {msg}", file=sys.stderr)
        sys.exit(1)

    client.set_registered_model_alias(args.model_name, args.alias, str(mv.version))
    print(
        f"Set alias {args.alias!r} -> {args.model_name} version {mv.version} (run {mv.run_id})"
    )


if __name__ == "__main__":
    main()
