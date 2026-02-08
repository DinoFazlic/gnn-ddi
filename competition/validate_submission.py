"""
Submission validator for the GNN Molecular Graph Classification Challenge.
==========================================================================

Checks that a ``predictions.csv`` file conforms to the expected schema
defined in ``config.yaml``.

Usage (standalone)::

    python -m competition.validate_submission \\
        submissions/inbox/alice/run_01/predictions.csv \\
        data/public/test.csv

Exit code 0 = valid, 1 = invalid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:
    import yaml  # type: ignore[import-untyped]

    _YAML = True
except ImportError:
    _YAML = False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_config() -> dict:
    """Load competition/config.yaml (best-effort)."""
    cfg_path = Path(__file__).parent / "config.yaml"
    if cfg_path.exists() and _YAML:
        with open(cfg_path) as fh:
            return yaml.safe_load(fh)  # type: ignore[no-any-return]
    return {}


def _pred_column(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Return (column_name, warning_or_None) for the prediction column."""
    if "y_pred" in df.columns:
        return "y_pred", None
    if "target" in df.columns:
        return "target", (
            "WARNING: Using legacy column name 'target'. "
            "Please rename to 'y_pred' for future submissions."
        )
    return None, None


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def validate(
    submission_path: str | Path,
    test_ids_path: str | Path,
) -> Tuple[bool, List[str]]:
    """
    Validate a submission CSV.

    Parameters
    ----------
    submission_path : path to predictions.csv
    test_ids_path   : path to the public test-ID list (``data/public/test.csv``)

    Returns
    -------
    (is_valid, messages)
        ``is_valid`` is True when the submission can be scored.
        ``messages`` may contain warnings even when valid.
    """
    messages: List[str] = []

    # --- read CSV ------------------------------------------------
    try:
        df = pd.read_csv(submission_path)
    except Exception as exc:
        return False, [f"Cannot read CSV: {exc}"]

    # --- required columns ----------------------------------------
    if "id" not in df.columns:
        messages.append("ERROR: Missing required column 'id'.")

    pred_col, pred_warn = _pred_column(df)
    if pred_col is None:
        messages.append(
            "ERROR: Missing prediction column — expected 'y_pred' (or legacy 'target')."
        )
    elif pred_warn:
        messages.append(pred_warn)

    if any(m.startswith("ERROR") for m in messages):
        return False, messages

    assert pred_col is not None  # mypy guard

    # --- load expected IDs ---------------------------------------
    test_ids_df = pd.read_csv(test_ids_path, header=None)
    expected_ids = set(test_ids_df[0].tolist())

    if len(df) != len(expected_ids):
        messages.append(
            f"ERROR: Row count mismatch — got {len(df)}, expected {len(expected_ids)}."
        )

    sub_ids = set(df["id"].tolist())
    missing = expected_ids - sub_ids
    extra = sub_ids - expected_ids
    if missing:
        messages.append(
            f"ERROR: Missing {len(missing)} molecule IDs: "
            f"{sorted(list(missing))[:5]}…"
        )
    if extra:
        messages.append(
            f"ERROR: {len(extra)} unexpected IDs: {sorted(list(extra))[:5]}…"
        )

    # --- value check ---------------------------------------------
    invalid = df[~df[pred_col].isin([0, 1])]
    if len(invalid) > 0:
        messages.append(
            f"ERROR: {len(invalid)} invalid prediction values (must be 0 or 1)."
        )

    is_valid = not any(m.startswith("ERROR") for m in messages)
    return is_valid, messages


def check_duplicate_team(
    team_name: str,
    leaderboard_csv: str | Path | None = None,
) -> Tuple[bool, List[str]]:
    """
    Verify that *team_name* has NOT already submitted.

    Returns ``(is_allowed, messages)``.
    ``is_allowed`` is False when the team already appears on the leaderboard.
    """
    messages: List[str] = []
    if leaderboard_csv is None:
        leaderboard_csv = Path(__file__).parent.parent / "leaderboard" / "leaderboard.csv"
    lb = Path(leaderboard_csv)
    if not lb.exists():
        return True, []

    try:
        lb_df = pd.read_csv(lb)
    except Exception:
        return True, ["WARNING: Could not read leaderboard CSV for duplicate check."]

    if "team" not in lb_df.columns:
        return True, []

    existing = lb_df[lb_df["team"].str.lower() == team_name.lower()]
    if not existing.empty:
        messages.append(
            f"ERROR: Team '{team_name}' already has a submission on the leaderboard. "
            "Each team is limited to one submission."
        )
        return False, messages

    return True, []


def validate_metadata(metadata_path: str | Path) -> Tuple[bool, List[str]]:
    """Optional: validate the companion metadata.json."""
    messages: List[str] = []
    p = Path(metadata_path)
    if not p.exists():
        return True, ["INFO: No metadata.json found (optional)."]
    try:
        with open(p) as fh:
            meta = json.load(fh)
    except Exception as exc:
        return False, [f"ERROR: Cannot parse metadata.json: {exc}"]

    for key in ("team_name", "model_name"):
        if key not in meta:
            messages.append(f"WARNING: metadata.json missing recommended key '{key}'.")

    is_valid = not any(m.startswith("ERROR") for m in messages)
    return is_valid, messages


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python -m competition.validate_submission "
            "<predictions.csv> <test_ids.csv>"
        )
        sys.exit(2)
    ok, msgs = validate(sys.argv[1], sys.argv[2])
    for m in msgs:
        print(m)
    sys.exit(0 if ok else 1)
