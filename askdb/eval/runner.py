"""跑分器：把评测集跑一遍，算出指标、失败分布，存成 JSON。

一条铁律：**报告里必须有基线**。没有基线的百分比无法判断好坏。
"""

from __future__ import annotations

import functools
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .. import db as dbmod
from ..agent import AgentRun, run_agent
from ..config import RUNS_DIR, Settings, get_settings
from ..llm import BaseLLM, MockLLM, build_llm
from .dataset import Case
from .failure import CATEGORY_LABELS, classify, detail_of


@dataclass
class CaseResult:
    id: str
    question: str
    correct: bool
    category: str
    difficulty: str
    tags: list[str]
    gold_sql: str
    predicted_sql: str | None
    steps: int
    tokens: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    detail: str = ""
    rows: int = 0
    gold_rows: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    run_id: str
    started_at: str
    provider: str
    model: str
    dataset: str
    n: int
    metrics: dict = field(default_factory=dict)
    by_difficulty: dict = field(default_factory=dict)
    by_tag: dict = field(default_factory=dict)
    mistakes: list[CaseResult] = field(default_factory=list)
    cases: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["mistakes"] = [c.to_dict() for c in self.mistakes]
        data["cases"] = [c.to_dict() for c in self.cases]
        return data


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    low, high = math.floor(k), math.ceil(k)
    if low == high:
        return ordered[int(k)]
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def _db_for(case: Case, default: str | Path) -> str | Path:
    return case.db_path or default


@functools.lru_cache(maxsize=32)
def _schema_for(db_path: str) -> str:
    """同一个库只抽一次 schema：BIRD 一个库挂着几十道题，重复抽太浪费。"""
    conn = dbmod.open_db(db_path)
    try:
        return dbmod.schema_text(conn)
    finally:
        conn.close()


def gold_rows_of(case: Case, db_path: str | Path) -> list[tuple]:
    db_path = _db_for(case, db_path)
    conn = dbmod.open_db(db_path)
    try:
        result = dbmod.run_query(conn, case.gold_sql, max_rows=10_000)
    finally:
        conn.close()
    if not result.ok:
        raise RuntimeError(f"gold SQL 执行失败（{case.id}）：{result.error}")
    return result.rows


def run_case(
    case: Case,
    db_path: str | Path,
    llm: BaseLLM,
    settings: Settings,
    gold_rows: list[tuple],
) -> CaseResult:
    started = time.perf_counter()
    target_db = _db_for(case, db_path)
    try:
        run: AgentRun = run_agent(
            case.question,
            target_db,
            llm=llm,
            settings=settings,
            schema=_schema_for(str(Path(target_db).resolve())),
            evidence=case.evidence,
        )
        correct = run.ok and dbmod.results_equal(run.rows, gold_rows, ordered=case.ordered)
        category = classify(run, correct=correct)
        return CaseResult(
            id=case.id,
            question=case.question,
            correct=correct,
            category=category,
            difficulty=case.difficulty,
            tags=list(case.tags),
            gold_sql=case.gold_sql,
            predicted_sql=run.answer_sql,
            steps=len(run.steps),
            tokens=run.tokens,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            latency_ms=run.latency_ms,
            detail=detail_of(run),
            rows=len(run.rows),
            gold_rows=len(gold_rows),
        )
    except Exception as exc:  # noqa: BLE001 - 单题异常不能拖垮整轮评测
        return CaseResult(
            id=case.id,
            question=case.question,
            correct=False,
            category="llm_error",
            difficulty=case.difficulty,
            tags=list(case.tags),
            gold_sql=case.gold_sql,
            predicted_sql=None,
            steps=0,
            tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000,
            detail=f"内部异常：{type(exc).__name__}: {exc}",
            gold_rows=len(gold_rows),
        )


def _aggregate(cases: list[CaseResult]) -> dict:
    total = len(cases)
    correct = sum(1 for c in cases if c.correct)
    valid = [c for c in cases if c.category != "llm_error"]
    failures = Counter(c.category for c in cases if not c.correct)
    latencies = [c.latency_ms for c in cases]
    return {
        "n": total,
        "correct": correct,
        "exact_match": round(correct / total, 4) if total else 0.0,
        "exact_match_valid": round(sum(c.correct for c in valid) / len(valid), 4) if valid else 0.0,
        "avg_steps": round(sum(c.steps for c in cases) / total, 2) if total else 0.0,
        "avg_tokens": round(sum(c.tokens for c in cases) / total, 1) if total else 0.0,
        "total_tokens": sum(c.tokens for c in cases),
        "p50_latency_ms": round(_percentile(latencies, 50), 1),
        "p95_latency_ms": round(_percentile(latencies, 95), 1),
        "failures": {CATEGORY_LABELS.get(k, k): v for k, v in failures.most_common()},
    }


def _breakdown(cases: list[CaseResult], key) -> dict:
    groups: dict[str, list[CaseResult]] = {}
    for case in cases:
        for name in key(case):
            groups.setdefault(name, []).append(case)
    return {
        name: {
            "n": len(group),
            "exact_match": round(sum(c.correct for c in group) / len(group), 4),
        }
        for name, group in sorted(groups.items())
    }


def run_eval(
    cases: list[Case],
    db_path: str | Path,
    *,
    llm: BaseLLM | None = None,
    settings: Settings | None = None,
    dataset_name: str = "sample",
) -> Report:
    settings = settings or get_settings()
    llm = llm or build_llm(settings)
    started_at = datetime.now()

    gold_rows = {case.id: gold_rows_of(case, db_path) for case in cases}
    results = [run_case(case, db_path, llm, settings, gold_rows[case.id]) for case in cases]

    report = Report(
        # 带微秒与数据集名：同一秒内跑多次也不会互相覆盖（覆盖 = 丢失基线）
        run_id=f"{started_at:%Y%m%d-%H%M%S-%f}-{dataset_name}-{settings.provider}",
        started_at=started_at.isoformat(timespec="seconds"),
        provider=settings.provider,
        model=(settings.model or "mock") if settings.provider != "mock" else "mock",
        dataset=dataset_name,
        n=len(results),
        metrics=_aggregate(results),
        by_difficulty=_breakdown(results, lambda c: [c.difficulty]),
        by_tag=_breakdown(results, lambda c: c.tags or ["untagged"]),
        mistakes=[c for c in results if not c.correct],
        cases=results,
    )
    return report


def run_self_check(cases: list[Case], db_path: str | Path, settings: Settings | None = None) -> Report:
    """用 gold SQL 冒充模型输出，验证评测器本身——结果必须是 100%。

    这是整套评测的地基：如果自检不到 100%，说明题目或比较逻辑有问题，
    此时任何模型分数都不可信。
    """
    settings = settings or get_settings()
    by_question = {case.question: case.gold_sql for case in cases}
    llm = MockLLM(lambda payload: by_question.get(payload[-1]["content"], "SELECT 1"))
    return run_eval(cases, db_path, llm=llm, settings=settings, dataset_name="self-check")


def save_report(report: Report, out_dir: str | Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else RUNS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.run_id}.json"
    # 绝不覆盖已有报告：宁可多一个文件，也不能悄悄丢掉一次基线
    suffix = 1
    while path.exists():
        suffix += 1
        path = out_dir / f"{report.run_id}-{suffix}.json"
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def format_summary(report: Report) -> str:
    m = report.metrics
    lines = [
        f"数据集 {report.dataset}  |  provider={report.provider} model={report.model}",
        f"题目 {m['n']}  答对 {m['correct']}  "
        f"EX={m['exact_match']:.1%}（剔除调用失败 {m['exact_match_valid']:.1%}）",
        f"平均步数 {m['avg_steps']}  平均 token {m['avg_tokens']}  "
        f"P50 {m['p50_latency_ms']:.0f}ms  P95 {m['p95_latency_ms']:.0f}ms",
    ]
    if m["failures"]:
        lines.append("失败分布：" + "  ".join(f"{k} {v}" for k, v in m["failures"].items()))
    if report.by_difficulty:
        parts = [f"{k} {v['exact_match']:.0%}(n={v['n']})" for k, v in report.by_difficulty.items()]
        lines.append("按难度：" + "  ".join(parts))
    return "\n".join(lines)
