#!/usr/bin/env python3
"""Build a comprehensive perf comparison report from TorchComms logs.

For each collective + SendMsgSize(B), this script compares:
- CUDA vs XPU using comms
- CUDA vs XPU using c10d
- XPU comms vs XPU c10d

Output is a grouped CSV (one section per collective).

Useage: python3 torchcomms/comms/torchcomms/tests/perf/py/analyze_perf_report.py --perf-dir /home/guoqiong/git/torchcomms/perf_results_binding
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path


SECTION_RE = re.compile(r"^===\s+Synchronous\s+(.+?)\s+Performance\s+===$")
ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-9]*\.?[0-9]+)")
TS_RE = re.compile(r"(\d{8}_\d{6})")
CSV_HEADER = [
    "SendMsgSize(B)",
    "AvgXPU_comms(us)",
    "AvgCUDA_comms(us)",
    "AvgXPU_c10d(us)",
    "AvgCUDA_c10d(us)",
    "PctDiff_comms_xpu_vs_cuda(%)",
    "PctDiff_c10d_xpu_vs_cuda(%)",
    "PctDiff_xpu_comms_vs_c10d(%)",
    "PctDiff_cuda_comms_vs_c10d(%)",
    "gapChange(%)",
]
SUMMARY_CSV_HEADER = [
    "Collective",
    "RowsUsed",
    "Avg_gapChange(%)",
    "AbsGapChangeGT20",
    "Avg_PctDiff_cuda_comms_vs_c10d(%)",
    "AbsCudaGT20",
    "Avg_PctDiff_xpu_comms_vs_c10d(%)",
    "AbsXpuGT20",
    "GapDirection",
    "CudaDirection",
    "XpuDirection",
]
VS_LAST_CSV_HEADER = [
    "Collective",
    "AvgXPU_comms_CURRENT_DATE(us)",
    "AvgXPU_comms_LAST_DATE(us)",
    "Avg_PctDiff_xpu_comms_vs_last(%)",
    "AvgXPU_c10d_CURRENT_DATE(us)",
    "AvgXPU_c10d_LAST_DATE(us)",
    "Avg_PctDiff_xpu_c10d_vs_last(%)",
]


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


def find_latest_log(perf_dir: Path, must_include: list[str], any_include: list[str] | None = None) -> Path:
    candidates: list[Path] = []

    for p in perf_dir.glob("*.log"):
        if not p.is_file() or p.stat().st_size == 0:
            continue
        name = p.name.lower()
        if any(token.lower() not in name for token in must_include):
            continue
        if any_include and not any(token.lower() in name for token in any_include):
            continue
        candidates.append(p)

    if not candidates:
        wanted = ", ".join(must_include + (any_include or []))
        raise FileNotFoundError(
            f"No non-empty log found in {perf_dir} matching tokens: {wanted}"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_logs(perf_dir: Path, must_include: list[str], any_include: list[str] | None = None) -> list[Path]:
    candidates: list[Path] = []

    for p in perf_dir.glob("*.log"):
        if not p.is_file() or p.stat().st_size == 0:
            continue
        name = p.name.lower()
        if any(token.lower() not in name for token in must_include):
            continue
        if any_include and not any(token.lower() in name for token in any_include):
            continue
        candidates.append(p)

    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def log_datetime(path: Path) -> datetime:
    m = TS_RE.search(path.name)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    return datetime.fromtimestamp(path.stat().st_mtime)


def pct_diff_val(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("inf")
    return ((numerator / denominator) - 1.0) * 100.0


def pct_diff_str(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return "inf"
    return f"{pct_diff_val(numerator, denominator):.2f}"


def fmt_num(val: float) -> str:
    if math.isnan(val):
        return "nan"
    if val == float("inf"):
        return "inf"
    if val == float("-inf"):
        return "-inf"
    return f"{val:.2f}"


def avg_finite(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def direction(val: float) -> str:
    if math.isnan(val):
        return "nan"
    if val > 0:
        return "speedup"
    if val < 0:
        return "slowdown"
    return "flat"


def avg_last4_by_collective(logs: list[Path]) -> dict[str, float]:
    sums: dict[str, float] = {}
    cnts: dict[str, int] = {}

    for log in logs:
        data = parse_log(log)
        for coll, size_map in data.items():
            if not size_map:
                continue
            last4_sizes = sorted(size_map.keys())[-4:]
            mean_last4 = avg_finite([size_map[sz] for sz in last4_sizes])
            if math.isnan(mean_last4):
                continue
            sums[coll] = sums.get(coll, 0.0) + mean_last4
            cnts[coll] = cnts.get(coll, 0) + 1

    out: dict[str, float] = {}
    for coll, s in sums.items():
        out[coll] = s / cnts[coll]
    return out


def pctdiff_map(current_avg: dict[str, float], last_avg: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for coll in sorted(set(current_avg) & set(last_avg)):
        # Keep the same direction style as:
        # PctDiff_comms_xpu_vs_cuda = (AvgCUDA_comms / AvgXPU_comms - 1) * 100
        # so here: (AvgXPU_last / AvgXPU_current - 1) * 100
        out[coll] = pct_diff_val(last_avg[coll], current_avg[coll])
    return out


def two_latest_dates(logs: list[Path]) -> tuple[str, str, list[Path], list[Path]] | None:
    by_date: dict[str, list[Path]] = {}
    for p in logs:
        d = log_datetime(p).strftime("%Y%m%d")
        by_date.setdefault(d, []).append(p)
    dates = sorted(by_date.keys())
    if len(dates) < 2:
        return None
    last_date = dates[-2]
    curr_date = dates[-1]
    return curr_date, last_date, by_date[curr_date], by_date[last_date]


def compare_against_last(perf_dir: Path, output_csv: Path) -> bool:
    xpu_comms_logs = find_logs(perf_dir, must_include=["xpu", "comms"])
    xpu_c10d_logs = find_logs(perf_dir, must_include=["xpu"], any_include=["c10d", "c10"])

    comms_info = two_latest_dates(xpu_comms_logs)
    c10d_info = two_latest_dates(xpu_c10d_logs)
    if not comms_info and not c10d_info:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(VS_LAST_CSV_HEADER)
        return False

    comms_curr_avg: dict[str, float] = {}
    comms_last_avg: dict[str, float] = {}
    c10d_curr_avg: dict[str, float] = {}
    c10d_last_avg: dict[str, float] = {}
    comms_pct: dict[str, float] = {}
    c10d_pct: dict[str, float] = {}
    curr_comms_date = last_comms_date = ""
    curr_c10d_date = last_c10d_date = ""

    if comms_info:
        curr_comms_date, last_comms_date, curr_logs, last_logs = comms_info
        comms_curr_avg = avg_last4_by_collective(curr_logs)
        comms_last_avg = avg_last4_by_collective(last_logs)
        comms_pct = pctdiff_map(comms_curr_avg, comms_last_avg)

    if c10d_info:
        curr_c10d_date, last_c10d_date, curr_logs, last_logs = c10d_info
        c10d_curr_avg = avg_last4_by_collective(curr_logs)
        c10d_last_avg = avg_last4_by_collective(last_logs)
        c10d_pct = pctdiff_map(c10d_curr_avg, c10d_last_avg)

    header = [
        "Collective",
        f"AvgXPU_comms_{curr_comms_date or 'CURRENT'}(us)",
        f"AvgXPU_comms_{last_comms_date or 'LAST'}(us)",
        "Avg_PctDiff_xpu_comms_vs_last(%)",
        f"AvgXPU_c10d_{curr_c10d_date or 'CURRENT'}(us)",
        f"AvgXPU_c10d_{last_c10d_date or 'LAST'}(us)",
        "Avg_PctDiff_xpu_c10d_vs_last(%)",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        collectives = sorted(
            set(comms_curr_avg) | set(comms_last_avg) | set(comms_pct)
            | set(c10d_curr_avg) | set(c10d_last_avg) | set(c10d_pct)
        )
        for coll in collectives:
            writer.writerow(
                [
                    coll,
                    fmt_num(comms_curr_avg.get(coll, float("nan"))),
                    fmt_num(comms_last_avg.get(coll, float("nan"))),
                    fmt_num(comms_pct.get(coll, float("nan"))),
                    fmt_num(c10d_curr_avg.get(coll, float("nan"))),
                    fmt_num(c10d_last_avg.get(coll, float("nan"))),
                    fmt_num(c10d_pct.get(coll, float("nan"))),
                ]
            )

    return bool(comms_pct or c10d_pct)


def default_perf_dir() -> Path:
    script = Path(__file__).resolve()
    repo_perf = script.parents[5] / "perf_results"
    if repo_perf.exists():
        return repo_perf
    return script.parent


def write_comprehensive_csv(
    output_csv: Path,
    xpu_comms: dict[str, dict[int, float]],
    cuda_comms: dict[str, dict[int, float]],
    xpu_c10d: dict[str, dict[int, float]],
    cuda_c10d: dict[str, dict[int, float]],
) -> bool:
    wrote_any = False
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    common_collectives = sorted(
        set(xpu_comms) & set(cuda_comms) & set(xpu_c10d) & set(cuda_c10d)
    )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for coll in common_collectives:
            common_sizes = sorted(
                set(xpu_comms[coll])
                & set(cuda_comms[coll])
                & set(xpu_c10d[coll])
                & set(cuda_c10d[coll])
            )
            if not common_sizes:
                continue

            wrote_any = True
            writer.writerow([coll])
            writer.writerow(CSV_HEADER)

            for size in common_sizes:
                avg_xpu_comms = xpu_comms[coll][size]
                avg_cuda_comms = cuda_comms[coll][size]
                avg_xpu_c10d = xpu_c10d[coll][size]
                avg_cuda_c10d = cuda_c10d[coll][size]
                pct_comms_xpu_vs_cuda = pct_diff_val(avg_cuda_comms, avg_xpu_comms)
                pct_c10d_xpu_vs_cuda = pct_diff_val(avg_cuda_c10d, avg_xpu_c10d)
                gap_change = pct_comms_xpu_vs_cuda - pct_c10d_xpu_vs_cuda

                writer.writerow(
                    [
                        size,
                        f"{avg_xpu_comms:.2f}",
                        f"{avg_cuda_comms:.2f}",
                        f"{avg_xpu_c10d:.2f}",
                        f"{avg_cuda_c10d:.2f}",
                        fmt_num(pct_comms_xpu_vs_cuda),
                        fmt_num(pct_c10d_xpu_vs_cuda),
                        pct_diff_str(avg_xpu_c10d, avg_xpu_comms),
                        pct_diff_str(avg_cuda_c10d, avg_cuda_comms),
                        fmt_num(gap_change),
                    ]
                )

            writer.writerow([])

    return wrote_any


def write_summary_csv(
    summary_csv: Path,
    xpu_comms: dict[str, dict[int, float]],
    cuda_comms: dict[str, dict[int, float]],
    xpu_c10d: dict[str, dict[int, float]],
    cuda_c10d: dict[str, dict[int, float]],
) -> bool:
    wrote_any = False
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    common_collectives = sorted(
        set(xpu_comms) & set(cuda_comms) & set(xpu_c10d) & set(cuda_c10d)
    )

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_CSV_HEADER)

        for coll in common_collectives:
            common_sizes = sorted(
                set(xpu_comms[coll])
                & set(cuda_comms[coll])
                & set(xpu_c10d[coll])
                & set(cuda_c10d[coll])
            )
            if not common_sizes:
                continue

            wrote_any = True
            last4_sizes = common_sizes[-4:]
            gap_vals: list[float] = []
            cuda_vals: list[float] = []
            xpu_vals: list[float] = []

            for size in last4_sizes:
                avg_xpu_comms = xpu_comms[coll][size]
                avg_cuda_comms = cuda_comms[coll][size]
                avg_xpu_c10d = xpu_c10d[coll][size]
                avg_cuda_c10d = cuda_c10d[coll][size]

                pct_comms_xpu_vs_cuda = pct_diff_val(avg_cuda_comms, avg_xpu_comms)
                pct_c10d_xpu_vs_cuda = pct_diff_val(avg_cuda_c10d, avg_xpu_c10d)
                gap_vals.append(pct_comms_xpu_vs_cuda - pct_c10d_xpu_vs_cuda)
                cuda_vals.append(pct_diff_val(avg_cuda_c10d, avg_cuda_comms))
                xpu_vals.append(pct_diff_val(avg_xpu_c10d, avg_xpu_comms))

            avg_gap = avg_finite(gap_vals)
            avg_cuda = avg_finite(cuda_vals)
            avg_xpu = avg_finite(xpu_vals)

            writer.writerow(
                [
                    coll,
                    len(last4_sizes),
                    fmt_num(avg_gap),
                    "Yes" if (math.isfinite(avg_gap) and abs(avg_gap) > 20) else "No",
                    fmt_num(avg_cuda),
                    "Yes" if (math.isfinite(avg_cuda) and abs(avg_cuda) > 20) else "No",
                    fmt_num(avg_xpu),
                    "Yes" if (math.isfinite(avg_xpu) and abs(avg_xpu) > 20) else "No",
                    direction(avg_gap),
                    direction(avg_cuda),
                    direction(avg_xpu),
                ]
            )

    return wrote_any


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create comprehensive CUDA/XPU and comms/c10d perf comparison CSV."
    )
    parser.add_argument(
        "--perf-dir",
        type=Path,
        default=default_perf_dir(),
        help="Directory containing perf logs (default: torchcomms/perf_results).",
    )
    parser.add_argument(
        "--xpu-comms-log",
        type=Path,
        default=None,
        help="Path to XPU comms log. If omitted, uses latest '*xpu*comms*.log'.",
    )
    parser.add_argument(
        "--cuda-comms-log",
        type=Path,
        default=None,
        help="Path to CUDA comms log. If omitted, uses latest '*cuda*comms*.log'.",
    )
    parser.add_argument(
        "--xpu-c10d-log",
        type=Path,
        default=None,
        help="Path to XPU c10d log. If omitted, uses latest '*xpu*(c10d|c10)*.log'.",
    )
    parser.add_argument(
        "--cuda-c10d-log",
        type=Path,
        default=None,
        help="Path to CUDA c10d log. If omitted, uses latest '*cuda*(c10d|c10)*.log'.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to output CSV. Default: <perf-dir>/analyze_perf_report.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Path to summary CSV. Default: <perf-dir>/analyze_perf_report_summary.csv",
    )
    parser.add_argument(
        "--vs-last-csv",
        type=Path,
        default=None,
        help="Path to xpu today-vs-last-date CSV. Default: <perf-dir>/analyze_perf_report_vs_last.csv",
    )
    args = parser.parse_args()

    try:
        xpu_comms_log = args.xpu_comms_log or find_latest_log(
            args.perf_dir, must_include=["xpu", "comms"]
        )
        cuda_comms_log = args.cuda_comms_log or find_latest_log(
            args.perf_dir, must_include=["cuda", "comms"]
        )
        xpu_c10d_log = args.xpu_c10d_log or find_latest_log(
            args.perf_dir, must_include=["xpu"], any_include=["c10d", "c10"]
        )
        cuda_c10d_log = args.cuda_c10d_log or find_latest_log(
            args.perf_dir, must_include=["cuda"], any_include=["c10d", "c10"]
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    for p in [xpu_comms_log, cuda_comms_log, xpu_c10d_log, cuda_c10d_log]:
        if not p.exists() or p.stat().st_size == 0:
            print(f"ERROR: missing or empty log: {p}", file=sys.stderr)
            return 2

    xpu_comms = parse_log(xpu_comms_log)
    cuda_comms = parse_log(cuda_comms_log)
    xpu_c10d = parse_log(xpu_c10d_log)
    cuda_c10d = parse_log(cuda_c10d_log)

    output_csv = args.output_csv or (args.perf_dir / "analyze_perf_report.csv")
    summary_csv = args.summary_csv or (args.perf_dir / "analyze_perf_report_summary.csv")
    vs_last_csv = args.vs_last_csv or (args.perf_dir / "analyze_perf_report_vs_last.csv")

    print(f"XPU  comms log: {xpu_comms_log}")
    print(f"CUDA comms log: {cuda_comms_log}")
    print(f"XPU  c10d  log: {xpu_c10d_log}")
    print(f"CUDA c10d  log: {cuda_c10d_log}")

    wrote_any = write_comprehensive_csv(
        output_csv,
        xpu_comms,
        cuda_comms,
        xpu_c10d,
        cuda_c10d,
    )
    wrote_any_summary = write_summary_csv(
        summary_csv,
        xpu_comms,
        cuda_comms,
        xpu_c10d,
        cuda_c10d,
    )
    wrote_any_vs_last = compare_against_last(args.perf_dir, vs_last_csv)

    if not wrote_any:
        print("No overlapping collective/message-size entries across all four logs.")
        print(f"CSV report written to: {output_csv}")
        print(f"Summary CSV written to: {summary_csv}")
        print(f"Vs-last CSV written to: {vs_last_csv}")
        return 1

    print(f"CSV report written to: {output_csv}")
    if wrote_any_summary:
        print(f"Summary CSV written to: {summary_csv}")
    else:
        print(f"Summary CSV written to: {summary_csv} (no overlapping rows)")
    if wrote_any_vs_last:
        print(f"Vs-last CSV written to: {vs_last_csv}")
    else:
        print(f"Vs-last CSV written to: {vs_last_csv} (not enough xpu date history)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
