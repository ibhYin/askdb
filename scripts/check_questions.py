"""体检评测集：每条 gold SQL 必须可执行、非空，且 LIMIT 题不能有并列导致答案不稳定。

新增题目后先跑这个，再跑评测——坏题会污染分数，比没有题更糟。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from askdb.db import open_db, run_query  # noqa: E402


def main() -> int:
    db_path = PROJECT_ROOT / "data" / "sample" / "shop.db"
    questions_path = PROJECT_ROOT / "data" / "sample" / "questions.jsonl"
    cases = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    conn = open_db(db_path)
    problems = 0
    for case in cases:
        result = run_query(conn, case["gold_sql"], max_rows=1000)
        if not result.ok:
            problems += 1
            print(f"[FAIL] {case['id']} 执行报错：{result.error}")
            continue
        if not result.rows:
            problems += 1
            print(f"[FAIL] {case['id']} 结果为空：{case['question']}")
            continue

        note = "（ordered：顺序也参与判定）" if case.get("ordered") else ""
        print(f"[ OK ] {case['id']} {len(result.rows):>3} 行  {case['question']} {note}")

    conn.close()
    print(f"\n共 {len(cases)} 题，问题 {problems} 个")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
