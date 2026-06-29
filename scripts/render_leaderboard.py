#!/usr/bin/env python3
"""Render the GeoText-1652 leaderboard from structured JSON data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "leaderboard" / "data.json"
OUTPUT_PATH = ROOT / "LEADERBOARD.md"

REQUIRED_ENTRY_FIELDS = {
    "id",
    "method",
    "submitter",
    "training_data",
    "backbone",
    "metrics",
}

REQUIRED_METRICS = ("txt_r1", "txt_r5", "txt_r10", "img_r1", "img_r5", "img_r10")


def fail(message: str) -> None:
    raise SystemExit(f"leaderboard: {message}")


def load_data(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        fail(f"missing data file: {path}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON in {path}: {exc}")

    if not isinstance(data, dict):
        fail("top-level data must be a JSON object")
    return data


def as_float(value: Any, field: str, entry_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{entry_id}: metric {field} must be a number")
    value = float(value)
    if value < 0.0 or value > 100.0:
        fail(f"{entry_id}: metric {field} must be between 0 and 100")
    return value


def enrich_entry(entry: dict[str, Any]) -> dict[str, Any]:
    entry_id = str(entry.get("id", "<missing-id>"))
    missing = sorted(REQUIRED_ENTRY_FIELDS - set(entry))
    if missing:
        fail(f"{entry_id}: missing required fields: {', '.join(missing)}")

    metrics = entry["metrics"]
    if not isinstance(metrics, dict):
        fail(f"{entry_id}: metrics must be an object")

    for metric in REQUIRED_METRICS:
        if metric not in metrics:
            fail(f"{entry_id}: missing required metric {metric}")
        metrics[metric] = as_float(metrics[metric], metric, entry_id)

    txt_mean = sum(metrics[name] for name in ("txt_r1", "txt_r5", "txt_r10")) / 3.0
    img_mean = sum(metrics[name] for name in ("img_r1", "img_r5", "img_r10")) / 3.0
    r_mean = (txt_mean + img_mean) / 2.0

    for key, value in {
        "txt_r_mean": txt_mean,
        "img_r_mean": img_mean,
        "r_mean": r_mean,
    }.items():
        if key in metrics:
            supplied = as_float(metrics[key], key, entry_id)
            if abs(supplied - value) > 0.05:
                fail(f"{entry_id}: supplied {key}={supplied:.3f} does not match computed {value:.3f}")
        metrics[key] = value

    return entry


def validate(data: dict[str, Any]) -> None:
    if data.get("primary_metric") != "r_mean":
        fail("primary_metric must be r_mean")

    leaderboards = data.get("leaderboards")
    if not isinstance(leaderboards, list) or not leaderboards:
        fail("leaderboards must be a non-empty list")

    seen_splits: set[str] = set()
    seen_entries: set[str] = set()
    for board in leaderboards:
        if not isinstance(board, dict):
            fail("each leaderboard must be an object")
        split = board.get("split")
        name = board.get("name")
        entries = board.get("entries")
        if not split or not isinstance(split, str):
            fail("each leaderboard requires a string split")
        if split in seen_splits:
            fail(f"duplicate leaderboard split: {split}")
        seen_splits.add(split)
        if not name or not isinstance(name, str):
            fail(f"{split}: requires a string name")
        if not isinstance(entries, list):
            fail(f"{split}: entries must be a list")

        for entry in entries:
            if not isinstance(entry, dict):
                fail(f"{split}: each entry must be an object")
            entry_id = entry.get("id")
            if not entry_id or not isinstance(entry_id, str):
                fail(f"{split}: each entry requires a string id")
            scoped_id = f"{split}:{entry_id}"
            if scoped_id in seen_entries:
                fail(f"duplicate entry id in {split}: {entry_id}")
            seen_entries.add(scoped_id)
            enrich_entry(entry)


def escape_cell(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if not text:
        return "-"
    return text.replace("|", "\\|").replace("\n", " ")


def link_or_text(label: str, url: Any) -> str:
    if not isinstance(url, str) or not url.strip():
        return "-"
    url = url.strip()
    return f"[{label}]({url})"


def fmt_metric(value: float) -> str:
    return f"{value + 1e-9:.1f}"


def entry_sort_key(entry: dict[str, Any]) -> tuple[float, float, float, str]:
    metrics = entry["metrics"]
    return (
        metrics["r_mean"],
        metrics["txt_r1"],
        metrics["img_r1"],
        entry["method"].lower(),
    )


def render_board(board: dict[str, Any]) -> list[str]:
    entries = sorted(board["entries"], key=entry_sort_key, reverse=True)
    lines = [
        f"## {board['name']}",
        "",
        str(board.get("description", "")).strip(),
        "",
    ]

    if not entries:
        lines.extend([
            "No public submissions yet. Submit a PR by adding an entry to `leaderboard/data.json`.",
            "",
        ])
        return lines

    header = [
        "Rank",
        "Method",
        "Submitter",
        "Training Data",
        "Backbone",
        "Text R@1",
        "Text R@5",
        "Text R@10",
        "Image R@1",
        "Image R@5",
        "Image R@10",
        "Mean",
        "Links",
        "Verified",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for rank, entry in enumerate(entries, start=1):
        metrics = entry["metrics"]
        links = " ".join(
            part
            for part in (
                link_or_text("paper", entry.get("paper")),
                link_or_text("code", entry.get("code")),
                link_or_text("ckpt", entry.get("checkpoint")),
            )
            if part != "-"
        ) or "-"
        row = [
            str(rank),
            escape_cell(entry["method"]),
            escape_cell(entry["submitter"]),
            escape_cell(entry["training_data"]),
            escape_cell(entry["backbone"]),
            fmt_metric(metrics["txt_r1"]),
            fmt_metric(metrics["txt_r5"]),
            fmt_metric(metrics["txt_r10"]),
            fmt_metric(metrics["img_r1"]),
            fmt_metric(metrics["img_r5"]),
            fmt_metric(metrics["img_r10"]),
            f"**{fmt_metric(metrics['r_mean'])}**",
            links,
            "yes" if entry.get("verified") else "pending",
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return lines


def render(data: dict[str, Any]) -> str:
    lines = [
        "# GeoText-1652 Leaderboard",
        "",
        "<!-- This file is generated by scripts/render_leaderboard.py. Edit leaderboard/data.json instead. -->",
        "",
        "The primary ranking metric is **Mean**, computed as the average of Text R@1/R@5/R@10 and Image R@1/R@5/R@10 means. All scores are percentages.",
        "",
        "Use the official evaluation command from the README and report the metrics printed by `Method/re_bbox.py`.",
        "",
    ]

    for board in data["leaderboards"]:
        lines.extend(render_board(board))

    lines.extend(
        [
            "## Submit Results",
            "",
            "Open a pull request that edits `leaderboard/data.json`, then run:",
            "",
            "```bash",
            "python3 scripts/render_leaderboard.py",
            "```",
            "",
            "Required fields for each entry:",
            "",
            "- `id`: stable lowercase identifier, unique within the split",
            "- `method`: model or system name",
            "- `submitter`: person, team, or organization",
            "- `training_data`: training data used, including any external data",
            "- `backbone`: visual/text backbone or main model family",
            "- `metrics`: `txt_r1`, `txt_r5`, `txt_r10`, `img_r1`, `img_r5`, `img_r10`",
            "",
            "Please include paper/code/checkpoint links when public. Mark `verified` as `false` unless the maintainers have reproduced or audited the result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if LEADERBOARD.md is out of date")
    args = parser.parse_args()

    data = load_data(DATA_PATH)
    validate(data)
    rendered = render(data)

    if args.check:
        current = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
        if current != rendered:
            print("LEADERBOARD.md is out of date. Run: python3 scripts/render_leaderboard.py", file=sys.stderr)
            return 1
        return 0

    OUTPUT_PATH.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
