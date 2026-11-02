"""离线评估 schema 检索质量：不算模型，只看选表准不准。

指标是 **表召回率**：gold SQL 里出现的表，有多少被检索结果覆盖。
这个数字决定 schema 检索值不值得开——召回不上 95%，
模型就根本没有正确的表可写，再强的模型也白搭。

    python scripts/eval_retrieval.py --top-k 6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from askdb.db import list_tables, open_db  # noqa: E402
from askdb.eval.bird import database_path, invalid_ids  # noqa: E402
from askdb.schema_index import build_index  # noqa: E402

QUESTIONS = PROJECT_ROOT / "data" / "bird" / "mini_dev_sqlite.jsonl"


def referenced_tables(sql: str, known: set[str]) -> set[str]:
    """从 gold SQL 里认出它用了哪些表（按库的实际表名做整词匹配）。"""
    found = set()
    for table in known:
        if re.search(rf"(?<![\w]){re.escape(table)}(?![\w])", sql, re.IGNORECASE):
            found.add(table)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="离线评估 schema 检索的表召回率")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    raw = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = invalid_ids()
    raw = [r for r in raw if f"bird-{r['question_id']}" not in bad]
    if args.limit:
        raw = raw[: args.limit]

    indexes: dict[str, object] = {}
    tables_by_db: dict[str, set[str]] = {}
    stats = {"good": 0, "partial": 0, "miss": 0}
    recall_sum = 0.0
    miss_by_db: Counter = Counter()

    for item in raw:
        db_id = item["db_id"]
        if db_id not in indexes:
            conn = open_db(database_path(db_id))
            indexes[db_id] = build_index(conn)
            tables_by_db[db_id] = set(list_tables(conn))
            conn.close()

        gold = referenced_tables(item["SQL"], tables_by_db[db_id])
        if not gold:
            continue

        query = f"{item['question']}\n{item.get('evidence', '')}".strip()
        picked = set(indexes[db_id].search_with_neighbors(query, top_k=args.top_k, hops=args.hops))
        hit = gold & picked
        recall = len(hit) / len(gold)
        recall_sum += recall

        if recall == 1.0:
            stats["good"] += 1
        elif recall > 0:
            stats["partial"] += 1
            miss_by_db[db_id] += 1
        else:
            stats["miss"] += 1
            miss_by_db[db_id] += 1

    total = sum(stats.values())
    print(f"题量 {total}  top_k={args.top_k} hops={args.hops}")
    print(f"表召回率（平均）: {recall_sum / total:.1%}")
    print(
        f"完全命中 {stats['good']} ({stats['good'] / total:.1%})  "
        f"部分命中 {stats['partial']}  完全没中 {stats['miss']}"
    )
    if miss_by_db:
        print("有漏表的库：" + "  ".join(f"{k} {v}" for k, v in miss_by_db.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
