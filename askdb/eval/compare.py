"""对比两次评测：把"改进了多少"变成一张表。

用法：
    python -m askdb.eval.compare runs/baseline.json runs/v2.json
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

METRIC_ROWS = [
    ("exact_match", "执行准确率", "pct"),
    ("exact_match_valid", "EX（剔除调用失败）", "pct"),
    ("correct", "答对题数", "int"),
    ("avg_steps", "平均步数", "float"),
    ("avg_tokens", "平均 token", "float"),
    ("p50_latency_ms", "P50 延迟(ms)", "float"),
    ("p95_latency_ms", "P95 延迟(ms)", "float"),
]


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _display_width(text: str) -> int:
    """中文字符占两列，用 len() 对齐会歪。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, *, right: bool = False) -> str:
    gap = max(0, width - _display_width(text))
    return (" " * gap + text) if right else (text + " " * gap)


def _fmt(value, kind: str) -> str:
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "int":
        return f"{value}"
    return f"{value:g}"


def _delta(before, after, kind: str) -> str:
    diff = after - before
    if kind == "pct":
        return f"{diff * 100:+.1f}pt"
    if kind == "int":
        return f"{diff:+d}"
    return f"{diff:+g}"


def compare(before: dict, after: dict) -> dict:
    """返回与展示无关的差异数据，便于测试与后续画图。"""
    rows = []
    for key, label, kind in METRIC_ROWS:
        b = before["metrics"].get(key, 0)
        a = after["metrics"].get(key, 0)
        rows.append(
            {
                "key": key,
                "label": label,
                "before": b,
                "after": a,
                "delta": a - b,
                "delta_text": _delta(b, a, kind),
                "kind": kind,
            }
        )

    before_fail = before["metrics"].get("failures", {})
    after_fail = after["metrics"].get("failures", {})
    failures = [
        {"label": name, "before": before_fail.get(name, 0), "after": after_fail.get(name, 0)}
        for name in sorted(set(before_fail) | set(after_fail))
    ]
    return {
        "before_n": before["metrics"]["n"],
        "after_n": after["metrics"]["n"],
        "rows": rows,
        "failures": failures,
    }


def format_comparison(diff: dict) -> str:
    lines = []
    if diff["before_n"] != diff["after_n"]:
        lines.append(f"注意：两次题量不同（{diff['before_n']} vs {diff['after_n']}），百分比仅供参考")
    lines.append(_pad("指标", 20) + _pad("基线", 10, right=True) + _pad("现在", 10, right=True) + _pad("变化", 12, right=True))
    lines.append("-" * 52)
    for row in diff["rows"]:
        lines.append(
            _pad(row["label"], 20)
            + _pad(_fmt(row["before"], row["kind"]), 10, right=True)
            + _pad(_fmt(row["after"], row["kind"]), 10, right=True)
            + _pad(row["delta_text"], 12, right=True)
        )
    if diff["failures"]:
        lines.append("")
        lines.append(_pad("失败类型", 20) + _pad("基线", 10, right=True) + _pad("现在", 10, right=True))
        lines.append("-" * 40)
        for row in diff["failures"]:
            lines.append(_pad(row["label"], 20) + _pad(str(row["before"]), 10, right=True) + _pad(str(row["after"]), 10, right=True))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m askdb.eval.compare", description="对比两次评测报告")
    parser.add_argument("before", help="基线报告 JSON")
    parser.add_argument("after", help="新报告 JSON")
    args = parser.parse_args(argv)

    print(format_comparison(compare(load_report(args.before), load_report(args.after))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
