"""评测集加载。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    gold_sql: str
    ordered: bool = False
    tags: tuple[str, ...] = ()
    difficulty: str = "medium"
    # 多库评测集（如 BIRD）里每道题绑定自己的库；单库数据集留空即可
    db_path: str | None = None
    # BIRD 的 evidence 提示（外部业务知识），参与组 prompt
    evidence: str = ""

    @property
    def key(self) -> str:
        return f"{self.difficulty}/{','.join(self.tags) or 'untagged'}"


def load_cases(
    path: str | Path,
    *,
    limit: int | None = None,
    ids: list[str] | None = None,
) -> list[Case]:
    path = Path(path)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases: list[Case] = []
    for line in lines:
        raw = json.loads(line)
        cases.append(
            Case(
                id=str(raw["id"]),
                question=raw["question"],
                gold_sql=raw["gold_sql"],
                ordered=bool(raw.get("ordered", False)),
                tags=tuple(raw.get("tags", ())),
                difficulty=raw.get("difficulty", "medium"),
                db_path=raw.get("db_path"),
                evidence=raw.get("evidence", ""),
            )
        )
    if ids:
        wanted = {str(i) for i in ids}
        cases = [c for c in cases if c.id in wanted]
    if limit:
        cases = cases[:limit]
    return cases
