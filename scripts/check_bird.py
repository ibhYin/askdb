"""体检 BIRD 题目：gold SQL 必须能跑通且非空。

公开基准也会有坏题（数据集版本与数据库版本对不上、口径有歧义）。
直接把它们算进分母，等于给模型扣分；正确做法是先体检、再隔离。

    python scripts/check_bird.py            # 全部 500 题
    python scripts/check_bird.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from askdb.db import open_db, run_query  # noqa: E402
from askdb.eval.bird import QUESTIONS_PATH, database_path  # noqa: E402

INVALID_PATH = PROJECT_ROOT / "data" / "bird" / "invalid_ids.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="体检 BIRD gold SQL")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    raw = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        raw = raw[: args.limit]

    conns: dict[str, object] = {}
    invalid: list[dict] = []
    by_db: Counter = Counter()

    for index, item in enumerate(raw, 1):
        db_id = item["db_id"]
        if db_id not in conns:
            conns[db_id] = open_db(database_path(db_id))
        result = run_query(
            conns[db_id], item["SQL"], timeout_seconds=args.timeout, max_rows=10_000
        )
        problem = None
        if not result.ok:
            problem = f"执行报错：{result.error}"
        elif not result.rows:
            problem = "结果为空"
        if problem:
            by_db[db_id] += 1
            invalid.append(
                {"question_id": item["question_id"], "db_id": db_id, "reason": problem}
            )
            print(f"[FAIL] bird-{item['question_id']:<4} {db_id:<24} {problem}"[:160])
        if index % 50 == 0:
            print(f"  … 已检查 {index}/{len(raw)}", flush=True)

    for conn in conns.values():
        conn.close()

    total = len(raw)
    bad = len(invalid)
    print(f"\n共 {total} 题，问题 {bad} 个（{bad / total:.1%}）")
    if by_db:
        print("按库：" + "  ".join(f"{k} {v}" for k, v in by_db.most_common()))

    INVALID_PATH.parent.mkdir(parents=True, exist_ok=True)
    INVALID_PATH.write_text(
        json.dumps(
            {"checked": total, "invalid": invalid, "generated_at": "check_bird.py"},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"已写入 {INVALID_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
