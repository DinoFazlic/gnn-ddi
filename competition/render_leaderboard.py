"""
Render the leaderboard from the authoritative ``leaderboard/leaderboard.csv``
into ``leaderboard/leaderboard.md`` (Markdown) and optionally update the
interactive ``docs/leaderboard.js`` data blob.

Usage::

    python -m competition.render_leaderboard          # default paths
    python -m competition.render_leaderboard --csv leaderboard/leaderboard.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CSV = _REPO_ROOT / "leaderboard" / "leaderboard.csv"
_DEFAULT_MD = _REPO_ROOT / "leaderboard" / "leaderboard.md"
_DOCS_JS = _REPO_ROOT / "docs" / "leaderboard.js"

_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}


# ------------------------------------------------------------------
# CSV I/O
# ------------------------------------------------------------------

def load_leaderboard_csv(path: Path) -> List[Dict[str, Any]]:
    """Read leaderboard.csv → list of dicts.  Empty string → None."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            entry: Dict[str, Any] = {}
            for k, v in row.items():
                if v == "" or v is None:
                    entry[k] = None
                else:
                    # attempt numeric conversion
                    try:
                        entry[k] = int(v)
                    except ValueError:
                        try:
                            entry[k] = float(v)
                        except ValueError:
                            entry[k] = v
            rows.append(entry)
    return rows


def save_leaderboard_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows back to leaderboard.csv."""
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("rank,team,macro_f1,efficiency,params,time_ms,cliff_accuracy,submission_type,submitted_at\n")
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in r.items()})


def _sort_and_rank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort by macro_f1 descending and assign rank."""
    rows.sort(key=lambda r: float(r.get("macro_f1") or 0), reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


# ------------------------------------------------------------------
# Upsert helper (used by CI after scoring)
# ------------------------------------------------------------------

def upsert_entry(
    rows: List[Dict[str, Any]],
    team: str,
    macro_f1: float,
    efficiency: float | None = None,
    params: int | None = None,
    time_ms: float | None = None,
    cliff_accuracy: float | None = None,
    submission_type: str | None = None,
) -> List[Dict[str, Any]]:
    """Insert a team row.  Raises if the team already exists (one submission limit)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    existing = next((r for r in rows if str(r.get("team", "")).lower() == team.lower()), None)
    if existing:
        raise ValueError(
            f"Team '{team}' already has a submission on the leaderboard. "
            "Each team is limited to one submission."
        )
    rows.append({
        "rank": 0,
        "team": team,
        "macro_f1": round(macro_f1, 6),
        "efficiency": round(efficiency, 6) if efficiency else None,
        "params": params,
        "time_ms": round(time_ms, 2) if time_ms else None,
        "cliff_accuracy": round(cliff_accuracy, 4) if cliff_accuracy else None,
        "submission_type": submission_type,
        "submitted_at": now,
    })
    return _sort_and_rank(rows)


# ------------------------------------------------------------------
# Markdown renderer
# ------------------------------------------------------------------

def _fmt_params(p: Any) -> str:
    if p is None:
        return "-"
    p = int(p)
    if p >= 1_000_000:
        return f"{p / 1_000_000:.1f}M"
    if p >= 1_000:
        return f"{p / 1_000:.1f}K"
    return str(p)


def render_markdown(rows: List[Dict[str, Any]]) -> str:
    """Produce the full ``leaderboard.md`` content."""
    lines: List[str] = [
        "# 🏆 Leaderboard\n",
        "",
        "Competition: **GNN Molecular Graph Classification Challenge**\n",
        "",
        "Primary Metric: **Macro F1 Score** (higher is better)\n",
        "",
        r"Efficiency: $\text{Eff} = \frac{F_1^2}{\log_{10}(t_{ms}) \times \log_{10}(p)}$" + "\n",
        "",
        "---\n",
        "",
        "| Rank | Team | Macro-F1 | Efficiency | Params | Time (ms) | Cliff Acc | Type | Submitted |",
        "|------|------|----------|------------|--------|-----------|-----------|-----------|-----------|",
    ]

    for r in rows:
        rank = r.get("rank", "")
        medal = _MEDALS.get(rank, "")
        rank_str = f"{medal} {rank}" if medal else str(rank)
        team = r.get("team", "?")
        f1 = f"{float(r['macro_f1']):.4f}" if r.get("macro_f1") is not None else "-"
        eff = f"{float(r['efficiency']):.4f}" if r.get("efficiency") is not None else "-"
        par = _fmt_params(r.get("params"))
        tms = f"{float(r['time_ms']):.1f}" if r.get("time_ms") is not None else "-"
        ca = f"{float(r['cliff_accuracy']):.4f}" if r.get("cliff_accuracy") is not None else "-"
        stype = r.get("submission_type") or "-"
        dt = r.get("submitted_at", "")
        lines.append(f"| {rank_str} | {team} | {f1} | {eff} | {par} | {tms} | {ca} | {stype} | {dt} |")

    lines += [
        "",
        "---\n",
        "",
        "### Legend\n",
        "",
        "- **Macro-F1**: Primary ranking metric",
        "- **Efficiency**: Higher is better — rewards accuracy + speed",
        "- **Params**: Trainable parameters",
        "- **Time (ms)**: Average inference time per batch",
        "- **Cliff Acc**: MMP-OOD pairwise activity-cliff accuracy",
        "- **Type**: human, llm, human+llm, or baseline",
        "",
        "*Baseline entries are provided by organisers.*\n",
        "",
        f"*Auto-generated by `competition/render_leaderboard.py` — "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*\n",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Optional: inject data into docs/leaderboard.js
# ------------------------------------------------------------------

_JS_MARKER_START = "// --- LEADERBOARD_DATA_START ---"
_JS_MARKER_END = "// --- LEADERBOARD_DATA_END ---"


def update_docs_js(rows: List[Dict[str, Any]], js_path: Path = _DOCS_JS) -> None:
    """Replace the data blob between markers in ``docs/leaderboard.js``."""
    if not js_path.exists():
        return
    text = js_path.read_text()
    start = text.find(_JS_MARKER_START)
    end = text.find(_JS_MARKER_END)
    if start == -1 or end == -1:
        return
    blob = json.dumps(rows, indent=2)
    new_text = (
        text[: start + len(_JS_MARKER_START)]
        + "\nconst LEADERBOARD_DATA = "
        + blob
        + ";\n"
        + text[end:]
    )
    js_path.write_text(new_text)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Render leaderboard.md from leaderboard.csv")
    ap.add_argument("--csv", type=str, default=str(_DEFAULT_CSV))
    ap.add_argument("--md", type=str, default=str(_DEFAULT_MD))
    ap.add_argument("--update-js", action="store_true", help="Also update docs/leaderboard.js")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    md_path = Path(args.md)

    rows = load_leaderboard_csv(csv_path)
    rows = _sort_and_rank(rows)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(rows))
    print(f"✅  Wrote {md_path}  ({len(rows)} entries)")

    if args.update_js:
        update_docs_js(rows)
        print(f"✅  Updated {_DOCS_JS}")


if __name__ == "__main__":
    main()
