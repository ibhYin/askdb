"""命令行入口：python -m askdb.eval [--limit N] [--self-check]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bird import load_bird_cases, summarize
from .dataset import load_cases
from .runner import format_summary, run_eval, run_self_check, save_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "data" / "sample" / "shop.db"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "sample" / "questions.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m askdb.eval",
        description="跑一轮评测并输出报告（JSON 存到 runs/）",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="评测集 jsonl 路径")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite 数据库路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--ids", default=None, help="只跑指定 id，逗号分隔，如 q001,q002")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="用 gold SQL 冒充模型输出，验证评测器本身（必须 100%%）",
    )
    parser.add_argument("--bird", action="store_true", help="用 BIRD mini_dev 评测集")
    parser.add_argument("--db-id", default=None, help="只跑 BIRD 中某个库，如 toxicology")
    parser.add_argument(
        "--bird-difficulty",
        default=None,
        choices=["simple", "moderate", "challenging"],
        help="只跑 BIRD 某档难度",
    )
    parser.add_argument("--out", default=None, help="报告目录，默认 runs/")
    parser.add_argument("--no-save", action="store_true", help="不写报告文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ids = [i.strip() for i in args.ids.split(",")] if args.ids else None
    if args.bird:
        cases = load_bird_cases(
            limit=args.limit, db_id=args.db_id, difficulty=args.bird_difficulty
        )
        print(summarize(cases))
        dataset_name = "bird-mini-dev"
    else:
        cases = load_cases(args.dataset, limit=args.limit, ids=ids)
        dataset_name = "sample"
    if not cases:
        print("评测集为空，检查 --dataset / --limit / --ids", file=sys.stderr)
        return 2

    report = (
        run_self_check(cases, args.db)
        if args.self_check
        else run_eval(cases, args.db, dataset_name=dataset_name)
    )
    print(format_summary(report))

    if args.self_check and report.metrics["exact_match"] < 1.0:
        print("\n自检未达 100%，评测器或题目有问题：", file=sys.stderr)
        for case in report.mistakes[:5]:
            print(f"  {case.id} {case.category} {case.detail}", file=sys.stderr)
        return 1

    if report.mistakes:
        print(f"\n错题（{len(report.mistakes)}）：")
        for case in report.mistakes:
            print(f"  [{case.id}] {case.category}: {case.question}")
            if case.predicted_sql:
                print(f"        pred: {case.predicted_sql}")
            if case.detail:
                print(f"        why : {case.detail}")

    if not args.no_save:
        path = save_report(report, args.out)
        print(f"\n报告已保存：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
