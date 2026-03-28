"""
Main evaluation entry-point for the GNN Molecular Graph Classification Challenge.
==================================================================================

This script is the **single scoring gateway** called by CI.  It:

1. Validates the submission  (via ``validate_submission.validate``)
2. Loads ground-truth labels (from file, GitHub Secret, or private repo)
3. Computes primary + secondary metrics  (via ``metrics``)
4. Prints machine-readable ``KEY:VALUE`` lines for CI to capture
5. Optionally writes a JSON results file

Usage::

    python -m competition.evaluate \\
        submissions/inbox/alice/run_01/predictions.csv \\
        --labels data/test_labels.csv \\
        --output-json results.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Sibling imports (works both as ``python -m competition.evaluate``
# and when the repo root is on sys.path).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from metrics import compute_all_metrics, efficiency_score, macro_f1  # noqa: E402
from validate_submission import validate  # noqa: E402

try:
    import yaml  # type: ignore[import-untyped]

    _YAML = True
except ImportError:
    _YAML = False


# ------------------------------------------------------------------
# Label loading helpers
# ------------------------------------------------------------------

def _labels_from_secret() -> Optional[pd.DataFrame]:
    """Decode ``TEST_LABELS_CSV`` GitHub Secret (base64-encoded CSV)."""
    raw = os.environ.get("TEST_LABELS_CSV")
    if not raw:
        return None
    try:
        csv_bytes = base64.b64decode(raw)
        import io

        return pd.read_csv(io.BytesIO(csv_bytes))
    except Exception:
        return None


def _labels_from_file(path: str | Path) -> Optional[pd.DataFrame]:
    """Load labels from a local CSV file."""
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    return None


def _resolve_labels(explicit_path: Optional[str]) -> pd.DataFrame:
    """Try, in order: explicit path → GitHub Secret → conventional path."""
    if explicit_path:
        df = _labels_from_file(explicit_path)
        if df is not None:
            return df

    # GitHub Secret
    df = _labels_from_secret()
    if df is not None:
        return df

    # Convention: data/test_labels.csv (populated by CI from private repo)
    repo_root = _HERE.parent
    for candidate in [
        repo_root / "data" / "test_labels.csv",
        repo_root / "private_data" / "test_labels.csv",
    ]:
        df = _labels_from_file(candidate)
        if df is not None:
            return df

    print("ERROR: Could not locate ground-truth labels.")
    print("  Provide --labels <path> or set the TEST_LABELS_CSV secret.")
    sys.exit(1)


# ------------------------------------------------------------------
# Submission normalisation
# ------------------------------------------------------------------

def _normalise_submission(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy 'target' column to 'y_pred' if needed."""
    if "y_pred" not in df.columns and "target" in df.columns:
        df = df.rename(columns={"target": "y_pred"})
    return df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def evaluate(
    submission_path: str | Path,
    labels_df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score a single submission and return a metrics dict."""
    sub_df = _normalise_submission(pd.read_csv(submission_path))

    # Align on id
    merged = labels_df.merge(sub_df, on="id", suffixes=("_true", "_pred"))
    if "target" in merged.columns:
        y_true = merged["target"].values
    else:
        y_true = merged["y_pred_true"].values if "y_pred_true" in merged.columns else merged["target_true"].values
    y_pred = merged["y_pred"].values if "y_pred" in merged.columns else merged["y_pred_pred"].values

    results = compute_all_metrics(y_true.tolist(), y_pred.tolist())

    # Efficiency (from metadata)
    if metadata:
        t = metadata.get("inference_time_ms") or metadata.get(
            "efficiency_metrics", {}
        ).get("inference_time_ms")
        p = metadata.get("total_params") or metadata.get(
            "efficiency_metrics", {}
        ).get("total_params")
        if t and p:
            results["efficiency"] = efficiency_score(results["macro_f1"], t, int(p))
            results["inference_time_ms"] = t
            results["total_params"] = int(p)

    return results


def _load_metadata(submission_dir: Path) -> Optional[Dict[str, Any]]:
    """Load metadata.json next to the predictions file."""
    for name in ("metadata.json", "metadata.yaml", "metadata.yml"):
        p = submission_dir / name
        if p.exists():
            with open(p) as fh:
                if name.endswith(".json"):
                    return json.load(fh)  # type: ignore[no-any-return]
                elif _YAML:
                    return yaml.safe_load(fh)  # type: ignore[no-any-return]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GNN Challenge — Evaluate a submission"
    )
    parser.add_argument(
        "submission", type=str, help="Path to predictions.csv"
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Path to ground-truth CSV (id,target). "
             "Falls back to TEST_LABELS_CSV secret → data/test_labels.csv.",
    )
    parser.add_argument(
        "--test-ids", type=str, default=None,
        help="Path to public test-ID list for validation "
             "(default: data/public/test.csv).",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Write full results to a JSON file.",
    )
    parser.add_argument(
        "--pairs-csv", type=str, default=None,
        help="MMP-OOD pairs CSV for cliff-accuracy evaluation.",
    )

    args = parser.parse_args()
    submission_path = Path(args.submission)

    if not submission_path.exists():
        print(f"ERROR: Submission not found: {submission_path}")
        sys.exit(1)

    # --- Validate ------------------------------------------------
    repo_root = _HERE.parent
    test_ids = args.test_ids or str(repo_root / "data" / "public" / "test.csv")
    # Fallback to old location
    if not Path(test_ids).exists():
        test_ids = str(repo_root / "data" / "test.csv")

    ok, msgs = validate(str(submission_path), test_ids)
    for m in msgs:
        print(m)
    if not ok:
        sys.exit(1)

    # --- Load labels ---------------------------------------------
    labels_df = _resolve_labels(args.labels)

    # --- Load metadata -------------------------------------------
    metadata = _load_metadata(submission_path.parent)

    # --- Evaluate ------------------------------------------------
    print("=" * 60)
    print("GNN Molecular Graph Classification Challenge — Scoring")
    print("=" * 60)

    results = evaluate(submission_path, labels_df, metadata)

    print(f"\n🎯  Macro F1 Score: {results['macro_f1']:.4f}")
    print(f"    Accuracy:       {results['accuracy']:.4f}")
    print(f"    Precision:      {results['precision_macro']:.4f}")
    print(f"    Recall:         {results['recall_macro']:.4f}")

    if "efficiency" in results:
        print(f"\n⚡  Efficiency:      {results['efficiency']:.4f}")
        print(f"    Params:         {results.get('total_params', '?'):,}")
        print(f"    Time (ms):      {results.get('inference_time_ms', '?')}")

    # --- MMP-OOD cliff accuracy ----------------------------------
    if args.pairs_csv:
        try:
            sys.path.insert(0, str(repo_root))
            from evaluation.mmp_ood import (
                compute_cliff_accuracy_hard,
                load_pairs_csv,
            )

            pairs = load_pairs_csv(args.pairs_csv)
            sub_df = _normalise_submission(pd.read_csv(submission_path))
            merged = labels_df.merge(sub_df, on="id", suffixes=("_true", "_pred"))
            pred_col = "y_pred" if "y_pred" in merged.columns else "y_pred_pred"
            pred_dict = dict(
                zip(merged["id"].values, merged[pred_col].values)
            )
            cliff_acc, per_pair = compute_cliff_accuracy_hard(pairs, pred_dict)
            results["cliff_accuracy"] = cliff_acc
            results["cliff_pairs_evaluated"] = len(per_pair)
            print(f"\n🧬  Cliff Accuracy: {cliff_acc:.4f}  ({sum(per_pair)}/{len(per_pair)} pairs)")
        except Exception as exc:
            print(f"\n⚠️  MMP-OOD evaluation skipped: {exc}")

    # --- Machine-readable output for CI --------------------------
    print("\n" + "=" * 60)
    print(f"SCORE:{results['macro_f1']:.6f}")
    if "efficiency" in results:
        print(f"EFFICIENCY:{results['efficiency']:.6f}")
        print(f"PARAMS:{results.get('total_params', '')}")
        print(f"TIME_MS:{results.get('inference_time_ms', '')}")
    if "cliff_accuracy" in results:
        print(f"CLIFF_ACC:{results['cliff_accuracy']:.6f}")

    # --- JSON output ---------------------------------------------
    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    main()
