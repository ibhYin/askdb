"""把多份评测报告拼成一张消融表（Markdown）。

    python scripts/ablation_table.py runs/A.json runs/B.json runs/C.json
    python scripts/ablation_table.py --labels 基线,+检索,+检索与v2 runs/A.json runs/B.json

为什么单独做：报告多了以后，从哪个 JSON 里抄哪个数字很容易抄错，
而简历和 README 上的数字不允许抄错。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def describe(config: dict) -> str:
    parts = [f"prompt={config.get('prompt_version', '?')}"]
    if config.get("schema_retrieval"):
        parts.append(f"检索k={config.get('schema_top_k')}")
    else:
        parts.append("全量schema")
    if config.get("value_samples"):
        parts.append(f"取值样例={config['value_samples']}")
    return " · ".join(parts)


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render(paths: list[str], labels: list[str] | None = None) -> str:
    reports = [load(p) for p in paths]
    if labels is None:
        labels = [describe(r.get("config", {})) for r in reports]

    header = (
        "| 配置 | 题量 | EX | 答对 | 平均步数 | 平均 token | P50 | P95 | 失败 |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    rows = [header]
    base_ex = reports[0]["metrics"]["exact_match"]
    for label, report in zip(labels, reports):
        m = report["metrics"]
        delta = (m["exact_match"] - base_ex) * 100
        ex = f"{m['exact_match']:.1%}"
        if report is not reports[0]:
            ex += f" ({delta:+.1f}pt)"
        failures = "，".join(f"{k} {v}" for k, v in m.get("failures", {}).items()) or "—"
        rows.append(
            f"| {label} | {m['n']} | {ex} | {m['correct']} | {m['avg_steps']} | "
            f"{m['avg_tokens']:.0f} | {m['p50_latency_ms'] / 1000:.1f}s | "
            f"{m['p95_latency_ms'] / 1000:.1f}s | {failures} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成消融表")
    parser.add_argument("reports", nargs="+", help="按顺序给出报告路径，第一个作为基线")
    parser.add_argument("--labels", default=None, help="逗号分隔的配置名，默认从报告里读")
    args = parser.parse_args(argv)

    labels = [s.strip() for s in args.labels.split(",")] if args.labels else None
    if labels is not None and len(labels) != len(args.reports):
        raise SystemExit("--labels 数量必须与报告数量一致")

    print(render(args.reports, labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
