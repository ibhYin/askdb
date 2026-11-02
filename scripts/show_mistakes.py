"""逐条看错题：把预测 SQL 和 gold SQL 摆在一起，人工找规律。

自动化指标告诉你"错了多少"，只有看样例才知道"为什么错"。
这个脚本就是为了让后一件事足够省力。

    python scripts/show_mistakes.py --run runs/xxx.json --limit 10 --db card_games
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def latest_run(directory: Path) -> Path:
    runs = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise SystemExit(f"{directory} 下没有评测报告，先跑 make bird-eval")
    return runs[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="查看错题明细")
    parser.add_argument("--run", default=None, help="报告路径，默认取最新一份")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--db", default=None, help="只看某个库")
    parser.add_argument("--tag", default=None, help="只看含某个标签的题")
    parser.add_argument("--width", type=int, default=100)
    args = parser.parse_args()

    run_path = Path(args.run) if args.run else latest_run(PROJECT_ROOT / "runs")
    report = json.loads(run_path.read_text(encoding="utf-8"))

    mistakes = [c for c in report["cases"] if not c["correct"]]
    if args.db:
        mistakes = [c for c in mistakes if args.db in c["tags"]]
    if args.tag:
        mistakes = [c for c in mistakes if args.tag in c["tags"]]

    print(f"报告 {run_path.name}｜{report['metrics']['n']} 题，错 {len(mistakes)} 题")
    print(f"显示前 {min(args.limit, len(mistakes))} 条\n")

    for case in mistakes[: args.limit]:
        print("=" * args.width)
        print(f"[{case['id']}] {case['difficulty']}  {case['tags']}")
        print(f"问题：{case['question']}")
        print(f"\n预测（{case['steps']} 步，{case['rows']} 行）：")
        print(textwrap.indent((case["predicted_sql"] or "(无)").strip(), "  "))
        print(f"\n标准（{case['gold_rows']} 行）：")
        print(textwrap.indent(case["gold_sql"].strip(), "  "))
        if case["detail"]:
            print(f"\n说明：{case['detail']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
