"""评测器测试：自检必须 100%，失败归因与报告结构正确。"""

from __future__ import annotations

import json

from askdb.agent import AgentRun, Step
from askdb.db import QueryResult
from askdb.eval.dataset import load_cases
from askdb.eval.failure import classify
from askdb.eval.runner import run_self_check, save_report
from askdb.llm import LLMResponse

DB_PATH = "data/sample/shop.db"
QUESTIONS = "data/sample/questions.jsonl"


def _run_with(result: QueryResult | None, sql: str | None = "SELECT 1") -> AgentRun:
    run = AgentRun(question="q")
    run.steps.append(Step(index=1, sql=sql, response=LLMResponse(text=sql or ""), result=result))
    return run


def test_self_check_is_100_percent():
    """评测器自检：gold SQL 作为模型输出，必须全对。低于 100% 说明尺子坏了。"""
    cases = load_cases(QUESTIONS)
    report = run_self_check(cases, DB_PATH)
    assert report.metrics["exact_match"] == 1.0
    assert report.metrics["correct"] == len(cases)


def test_report_contains_breakdowns_and_metrics():
    cases = load_cases(QUESTIONS, limit=5)
    report = run_self_check(cases, DB_PATH)
    assert report.metrics["n"] == 5
    assert set(report.by_difficulty) <= {"easy", "medium", "hard"}
    assert report.metrics["p95_latency_ms"] >= report.metrics["p50_latency_ms"]
    assert report.metrics["total_tokens"] >= 0


def test_save_report_writes_json(tmp_path):
    cases = load_cases(QUESTIONS, limit=2)
    report = run_self_check(cases, DB_PATH)
    path = save_report(report, tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["metrics"]["n"] == 2
    assert len(data["cases"]) == 2
    assert data["dataset"] == "self-check"


def test_classify_categories():
    ok = QueryResult(columns=["x"], rows=[(1,)])
    empty = QueryResult(columns=["x"], rows=[])
    broken = QueryResult(error="no such column: nope")

    assert classify(_run_with(ok), correct=True) == "correct"
    assert classify(_run_with(ok), correct=False) == "wrong_result"
    assert classify(_run_with(empty), correct=False) == "empty_result"
    assert classify(_run_with(broken), correct=False) == "sql_error"
    assert classify(_run_with(None, sql=None), correct=False) == "no_sql"


def test_reports_never_overwrite_each_other(tmp_path):
    """同一秒内跑两次，必须是两个文件：覆盖等于静默丢掉一次基线。"""
    cases = load_cases(QUESTIONS, limit=2)
    first = save_report(run_self_check(cases, DB_PATH), tmp_path)
    second = save_report(run_self_check(cases, DB_PATH), tmp_path)
    assert first != second
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_large_result_sets_are_not_truncated_by_eval(tmp_path):
    """回归：agent 默认只取 200 行，若评测沿用它，>200 行的题会被判错。"""
    from askdb.eval.dataset import Case

    case = Case(
        id="big",
        question="列出所有订单明细",
        gold_sql="SELECT * FROM order_items",
        ordered=False,
        difficulty="easy",
    )
    report = run_self_check([case], DB_PATH)
    assert report.metrics["exact_match"] == 1.0
    assert report.cases[0].gold_rows > 200


def test_report_records_the_config_it_ran_with():
    """报告必须自解释：否则两份结果的差异说不清是哪个开关造成的。"""
    from dataclasses import replace

    from askdb.config import get_settings

    cases = load_cases(QUESTIONS, limit=2)
    settings = replace(get_settings(), prompt_version="v1", schema_retrieval=True, schema_top_k=4)
    report = run_self_check(cases, DB_PATH, settings=settings)
    assert report.config["prompt_version"] == "v1"
    assert report.config["schema_retrieval"] is True
    assert report.config["schema_top_k"] == 4


def test_concurrent_run_preserves_case_order():
    """并发必须不能打乱结果顺序：报告里题目顺序变了，人工比对就失效。"""
    cases = load_cases(QUESTIONS, limit=6)
    sequential = run_self_check(cases, DB_PATH)
    from askdb.eval.runner import run_eval

    concurrent = run_eval(cases, DB_PATH, concurrency=3, dataset_name="self-check")
    assert [c.id for c in concurrent.cases] == [c.id for c in sequential.cases]
    assert concurrent.metrics["n"] == sequential.metrics["n"]
