#!/usr/bin/env python3
"""Compare TorchComms collective perf logs between CUDA and XPU.

Computes row-wise:
    pct_diff = (t_cuda / t_xpu - 1) * 100
using Avg(us), matched by (collective, SendMsgSize(B)).

CSV output format:
- One section per collective
- First line is the collective name
- Followed by a table header and rows
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


SECTION_RE = re.compile(r"^===\s+Synchronous\s+(.+?)\s+Performance\s+===$")
ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-9]*\.?[0-9]+)")
CSV_HEADER = ["SendMsgSize(B)", "AvgCUDA(us)", "AvgXPU(us)", "PctDiff(%)"]


def parse_log(path: Path) -> dict[str, dict[int, float]]:
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


def find_latest_log(perf_dir: Path, keyword: str) -> Path:
    candidates = [
        p
        for p in perf_dir.glob("*.log")
        if keyword.lower() in p.name.lower() and p.is_file() and p.stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No non-empty log found in {perf_dir} with '{keyword}' in filename."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def pct_diff_str(t_cuda: float, t_xpu: float) -> str:
    if t_xpu == 0:
        return "inf"
    return f"{((t_cuda / t_xpu) - 1.0) * 100.0:.2f}"


def write_grouped_csv(
    output_csv: Path,
    cuda_data: dict[str, dict[int, float]],
    xpu_data: dict[str, dict[int, float]],
) -> bool:
    """Write grouped CSV and return True if any overlapping rows were written."""
    wrote_any = False
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for coll in sorted(set(cuda_data) & set(xpu_data)):
            common_sizes = sorted(set(cuda_data[coll]) & set(xpu_data[coll]))
            if not common_sizes:
                continue

            wrote_any = True
            writer.writerow([coll])
            writer.writerow(CSV_HEADER)
            for size in common_sizes:
                t_cuda = cuda_data[coll][size]
                t_xpu = xpu_data[coll][size]
                writer.writerow([
                    size,
                    f"{t_cuda:.2f}",
                    f"{t_xpu:.2f}",
                    pct_diff_str(t_cuda, t_xpu),
                ])
            writer.writerow([])

    return wrote_any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare CUDA vs XPU collective perf logs using Avg(us)."
    )
    parser.add_argument(
        "--perf-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing perf log files (default: script directory).",
    )
    parser.add_argument(
        "--cuda-log",
        type=Path,
        default=None,
        help="Path to CUDA log. If omitted, latest non-empty '*cuda*.log' in --perf-dir.",
    )
    parser.add_argument(
        "--xpu-log",
        type=Path,
        default=None,
        help="Path to XPU log. If omitted, latest non-empty '*xpu*.log' in --perf-dir.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to CSV report file. Default: <perf-dir>/compare_cuda_xpu_report.csv",
    )
    args = parser.parse_args()

    try:
        cuda_log = args.cuda_log or find_latest_log(args.perf_dir, "cuda")
        xpu_log = args.xpu_log or find_latest_log(args.perf_dir, "xpu")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not cuda_log.exists() or cuda_log.stat().st_size == 0:
        print(f"ERROR: CUDA log missing or empty: {cuda_log}", file=sys.stderr)
        return 2
    if not xpu_log.exists() or xpu_log.stat().st_size == 0:
        print(f"ERROR: XPU log missing or empty: {xpu_log}", file=sys.stderr)
        return 2

    output_csv = args.output_csv or (args.perf_dir / "compare_cuda_xpu_report.csv")
    cuda_data = parse_log(cuda_log)
    xpu_data = parse_log(xpu_log)

    print(f"CUDA log: {cuda_log}")
    print(f"XPU  log: {xpu_log}")
    print("Formula: (t_cuda / t_xpu - 1) * 100%")

    wrote_any = write_grouped_csv(output_csv, cuda_data, xpu_data)
    if not wrote_any:
        print("No overlapping collectives between CUDA and XPU logs.")
        print(f"CSV report written to: {output_csv}")
        return 1

    print(f"CSV report written to: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
