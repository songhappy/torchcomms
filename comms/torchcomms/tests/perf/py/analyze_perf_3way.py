#!/usr/bin/env python3
"""Compare three XPU perf logs: pure torchcomms, pure c10d, and c10d+torchcomms (distwrap).

For each collective and SendMsgSize(B), produces a table with columns:
    SendMsgSize(B)  AvgXPU_comms(us)  AvgXPU_c10d(us)  AvgXPU_c10d_comms(us)

Usage:
    python3 analyze_perf_3way.py --perf-dir /path/to/perf_results
    python3 analyze_perf_3way.py \
        --comms-log  perf_results/collective_perf_..._xpu_comms.log \
        --c10d-log   perf_results/collective_perf_..._xpu_c10d.log \
        --distwrap-log perf_results/collective_perf_..._xpu_distwrap.log
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path


SECTION_RE = re.compile(r"^===\s+Synchronous\s+(.+?)\s+Performance\s+===$")
ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-9]*\.?[0-9]+)")

CSV_HEADER = [
    "SendMsgSize(B)",
    "AvgXPU_comms(us)",
    "AvgXPU_c10d(us)",
    "AvgXPU_c10d_comms(us)",
    "PctDiff_c10d_vs_comms(%)",
    "PctDiff_distwrap_vs_comms(%)",
    "PctDiff_distwrap_vs_c10d(%)",
]

SUMMARY_CSV_HEADER = [
    "Collective",
    "RowsUsed",
    "Avg_PctDiff_c10d_vs_comms(%)",
    "Avg_PctDiff_distwrap_vs_comms(%)",
    "Avg_PctDiff_distwrap_vs_c10d(%)",
]


def parse_log(path: Path) -> dict[str, dict[int, float]]:
    """Parse a perf log file into {collective: {msg_size: avg_us}}."""
    by_collective: dict[str, dict[int, float]] = {}
    current_collective: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            m_section = SECTION_RE.match(line.strip())
            if m_section:
                current_collective = m_section.group(1).strip()
                by_collective.setdefault(current_collective, {})
                continue

            if not current_collective:
                continue

            m_row = ROW_RE.match(line)
            if not m_row:
                continue

            msg_size = int(m_row.group(1))
            avg_us = float(m_row.group(4))
            by_collective[current_collective][msg_size] = avg_us

    return by_collective


def find_latest_log(
    perf_dir: Path,
    must_include: list[str],
    any_include: list[str] | None = None,
    must_exclude: list[str] | None = None,
) -> Path | None:
    """Find the most recent log matching the given tokens."""
    candidates: list[Path] = []

    for p in perf_dir.glob("*.log"):
        if not p.is_file() or p.stat().st_size == 0:
            continue
        name = p.name.lower()
        if any(token.lower() not in name for token in must_include):
            continue
        if any_include and not any(token.lower() in name for token in any_include):
            continue
        if must_exclude and any(token.lower() in name for token in must_exclude):
            continue
        candidates.append(p)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def pct_diff(numerator: float, denominator: float) -> float:
    """Compute (numerator / denominator - 1) * 100."""
    if denominator == 0:
        return float("inf")
    return ((numerator / denominator) - 1.0) * 100.0


def fmt(val: float) -> str:
    if math.isnan(val) or math.isinf(val):
        return str(val)
    return f"{val:.2f}"


def avg_finite(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def write_3way_csv(
    output_csv: Path,
    xpu_comms: dict[str, dict[int, float]],
    xpu_c10d: dict[str, dict[int, float]],
    xpu_distwrap: dict[str, dict[int, float]],
) -> bool:
    """Write the per-collective, per-size comparison CSV."""
    wrote_any = False
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Union of all collectives present in any log
    all_collectives = sorted(set(xpu_comms) | set(xpu_c10d) | set(xpu_distwrap))

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for coll in all_collectives:
            comms_map = xpu_comms.get(coll, {})
            c10d_map = xpu_c10d.get(coll, {})
            distwrap_map = xpu_distwrap.get(coll, {})

            # Union of all sizes across the three logs for this collective
            all_sizes = sorted(set(comms_map) | set(c10d_map) | set(distwrap_map))
            if not all_sizes:
                continue

            wrote_any = True
            writer.writerow([coll])
            writer.writerow(CSV_HEADER)

            for size in all_sizes:
                v_comms = comms_map.get(size, float("nan"))
                v_c10d = c10d_map.get(size, float("nan"))
                v_distwrap = distwrap_map.get(size, float("nan"))

                # PctDiff: positive means slower than comms (baseline)
                pct_c10d_vs_comms = (
                    pct_diff(v_c10d, v_comms)
                    if math.isfinite(v_c10d) and math.isfinite(v_comms)
                    else float("nan")
                )
                pct_distwrap_vs_comms = (
                    pct_diff(v_distwrap, v_comms)
                    if math.isfinite(v_distwrap) and math.isfinite(v_comms)
                    else float("nan")
                )
                pct_distwrap_vs_c10d = (
                    pct_diff(v_distwrap, v_c10d)
                    if math.isfinite(v_distwrap) and math.isfinite(v_c10d)
                    else float("nan")
                )

                writer.writerow([
                    size,
                    fmt(v_comms),
                    fmt(v_c10d),
                    fmt(v_distwrap),
                    fmt(pct_c10d_vs_comms),
                    fmt(pct_distwrap_vs_comms),
                    fmt(pct_distwrap_vs_c10d),
                ])

            writer.writerow([])

    return wrote_any


def write_3way_summary_csv(
    summary_csv: Path,
    xpu_comms: dict[str, dict[int, float]],
    xpu_c10d: dict[str, dict[int, float]],
    xpu_distwrap: dict[str, dict[int, float]],
) -> bool:
    """Write summary CSV averaging the last-4 sizes per collective."""
    wrote_any = False
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    all_collectives = sorted(set(xpu_comms) | set(xpu_c10d) | set(xpu_distwrap))

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_CSV_HEADER)

        for coll in all_collectives:
            comms_map = xpu_comms.get(coll, {})
            c10d_map = xpu_c10d.get(coll, {})
            distwrap_map = xpu_distwrap.get(coll, {})

            all_sizes = sorted(set(comms_map) | set(c10d_map) | set(distwrap_map))
            if not all_sizes:
                continue

            wrote_any = True
            last4 = all_sizes[-4:]

            c10d_vs_comms_vals: list[float] = []
            distwrap_vs_comms_vals: list[float] = []
            distwrap_vs_c10d_vals: list[float] = []

            for size in last4:
                v_comms = comms_map.get(size, float("nan"))
                v_c10d = c10d_map.get(size, float("nan"))
                v_distwrap = distwrap_map.get(size, float("nan"))

                if math.isfinite(v_c10d) and math.isfinite(v_comms):
                    c10d_vs_comms_vals.append(pct_diff(v_c10d, v_comms))
                if math.isfinite(v_distwrap) and math.isfinite(v_comms):
                    distwrap_vs_comms_vals.append(pct_diff(v_distwrap, v_comms))
                if math.isfinite(v_distwrap) and math.isfinite(v_c10d):
                    distwrap_vs_c10d_vals.append(pct_diff(v_distwrap, v_c10d))

            writer.writerow([
                coll,
                len(last4),
                fmt(avg_finite(c10d_vs_comms_vals)),
                fmt(avg_finite(distwrap_vs_comms_vals)),
                fmt(avg_finite(distwrap_vs_c10d_vals)),
            ])

    return wrote_any


def default_perf_dir() -> Path:
    script = Path(__file__).resolve()
    repo_perf = script.parents[5] / "perf_results"
    if repo_perf.exists():
        return repo_perf
    return script.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 3 XPU perf logs: pure torchcomms, pure c10d, "
            "and c10d+torchcomms (distwrap)."
        ),
    )
    parser.add_argument(
        "--perf-dir",
        type=Path,
        default=default_perf_dir(),
        help="Directory containing perf logs.",
    )
    parser.add_argument(
        "--comms-log",
        type=Path,
        default=None,
        help="Path to XPU comms log. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--c10d-log",
        type=Path,
        default=None,
        help="Path to XPU c10d log. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--distwrap-log", "--c10d-comms-log",
        dest="distwrap_log",
        type=Path,
        default=None,
        help="Path to XPU c10d_comms (distwrap) log. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to output CSV. Default: <perf-dir>/analyze_perf_3way.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Path to summary CSV. Default: <perf-dir>/analyze_perf_3way_summary.csv",
    )
    args = parser.parse_args()

    # Locate logs — names are like collective_perf_*_xpu_{comms,c10d,c10d_comms}.log
    comms_log = args.comms_log or find_latest_log(
        args.perf_dir, must_include=["xpu", "comms"], must_exclude=["c10d_comms"],
    )
    c10d_log = args.c10d_log or find_latest_log(
        args.perf_dir, must_include=["xpu", "c10d"], must_exclude=["c10d_comms"],
    )
    distwrap_log = args.distwrap_log or find_latest_log(
        args.perf_dir, must_include=["xpu", "c10d_comms"],
    )

    print(f"XPU comms      log: {comms_log or '(not found)'}")
    print(f"XPU c10d       log: {c10d_log or '(not found)'}")
    print(f"XPU c10d_comms log: {distwrap_log or '(not found)'}")

    missing = []
    if not comms_log:
        missing.append("xpu-comms")
    if not c10d_log:
        missing.append("xpu-c10d")
    if not distwrap_log:
        missing.append("xpu-c10d_comms")

    if missing:
        print(f"Error: missing logs: {', '.join(missing)}", file=sys.stderr)
        print(
            "Provide them via --comms-log / --c10d-log / --distwrap-log "
            "or place them in --perf-dir.",
            file=sys.stderr,
        )
        return 1

    xpu_comms = parse_log(comms_log)
    xpu_c10d = parse_log(c10d_log)
    xpu_distwrap = parse_log(distwrap_log)

    output_csv = args.output_csv or (args.perf_dir / "analyze_perf_3way.csv")
    summary_csv = args.summary_csv or (args.perf_dir / "analyze_perf_3way_summary.csv")

    wrote = write_3way_csv(output_csv, xpu_comms, xpu_c10d, xpu_distwrap)
    wrote_summary = write_3way_summary_csv(summary_csv, xpu_comms, xpu_c10d, xpu_distwrap)

    if wrote:
        print(f"3-way CSV written to: {output_csv}")
    else:
        print("No common data found across the three logs.")

    if wrote_summary:
        print(f"3-way summary CSV written to: {summary_csv}")

    # Also print to stdout for quick inspection
    if wrote and output_csv.exists():
        print()
        print(output_csv.read_text())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
