"""BIRD mini_dev 评测集。

数据不在仓库里（11 个库合计几百 MB，且是 CC-BY-SA），由 scripts/fetch_bird.py 拉取：

    python scripts/fetch_bird.py --list          # 看有哪些库
    python scripts/fetch_bird.py                 # 下载并解压全部
    python scripts/fetch_bird.py --db toxicology # 只下一个

每题字段：question / evidence / SQL / difficulty / db_id。
evidence 是数据集给的外部知识（比如"SAT 优秀率 = 1500 分以上占比"），
属于题目的一部分，必须一起喂给模型——否则某些题人类也做不出来。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .dataset import Case

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIRD_DIR = PROJECT_ROOT / "data" / "bird"
QUESTIONS_PATH = BIRD_DIR / "mini_dev_sqlite.jsonl"
DATABASES_DIR = BIRD_DIR / "databases"


class BirdDataMissing(RuntimeError):
    """数据没准备好，提示用户去跑 fetch 脚本。"""


def database_path(db_id: str, databases_dir: Path = DATABASES_DIR) -> Path:
    return databases_dir / db_id / f"{db_id}.sqlite"


def load_bird_cases(
    *,
    questions_path: Path = QUESTIONS_PATH,
    databases_dir: Path = DATABASES_DIR,
    limit: int | None = None,
    db_id: str | None = None,
    difficulty: str | None = None,
    skip_missing: bool = True,
) -> list[Case]:
    if not questions_path.exists():
        raise BirdDataMissing(
            f"找不到 {questions_path}，先运行：python scripts/fetch_bird.py --list"
        )

    cases: list[Case] = []
    missing = 0
    for line in questions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if db_id and raw["db_id"] != db_id:
            continue
        if difficulty and raw.get("difficulty") != difficulty:
            continue

        path = database_path(raw["db_id"], databases_dir)
        if not path.exists():
            missing += 1
            if skip_missing:
                continue

        cases.append(
            Case(
                id=f"bird-{raw['question_id']}",
                question=raw["question"],
                gold_sql=raw["SQL"],
                ordered=False,  # BIRD 官方口径：无序比较
                tags=(raw["db_id"],),
                difficulty=raw.get("difficulty", "unknown"),
                db_path=str(path),
                evidence=raw.get("evidence", "") or "",
            )
        )

    if skip_missing and missing:
        print(f"[警告] 有 {missing} 题的数据库没下载，已跳过；跑 fetch_bird.py 补齐")
    if not cases:
        raise BirdDataMissing("没有可用的 BIRD 题目：检查 --db-id / --difficulty 或先下载数据库")
    if limit:
        cases = cases[:limit]
    return cases


def summarize(cases: list[Case]) -> str:
    by_db = Counter(case.tags[0] for case in cases)
    by_diff = Counter(case.difficulty for case in cases)
    lines = [f"题量 {len(cases)}"]
    lines.append("按库：" + "  ".join(f"{k} {v}" for k, v in by_db.most_common()))
    lines.append("按难度：" + "  ".join(f"{k} {v}" for k, v in by_diff.most_common()))
    return "\n".join(lines)
