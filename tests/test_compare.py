"""报告对比测试。"""

from __future__ import annotations

from askdb.eval.compare import compare, format_comparison


def _report(exact_match: float, n: int = 10, failures: dict | None = None, tokens: float = 500) -> dict:
    return {
        "metrics": {
            "n": n,
            "correct": round(exact_match * n),
            "exact_match": exact_match,
            "exact_match_valid": exact_match,
            "avg_steps": 1.2,
            "avg_tokens": tokens,
            "p50_latency_ms": 800.0,
            "p95_latency_ms": 2000.0,
            "failures": failures or {},
        }
    }


def test_compare_computes_delta_in_points():
    diff = compare(_report(0.52, failures={"结果不对": 5}), _report(0.78, failures={"结果不对": 2}))
    ex_row = next(r for r in diff["rows"] if r["key"] == "exact_match")
    assert ex_row["delta_text"] == "+26.0pt"

    failure = next(f for f in diff["failures"] if f["label"] == "结果不对")
    assert (failure["before"], failure["after"]) == (5, 2)


def test_compare_merges_failure_categories_from_both_sides():
    diff = compare(_report(0.5, failures={"结果不对": 3}), _report(0.5, failures={"SQL 执行报错": 2}))
    labels = {f["label"] for f in diff["failures"]}
    assert labels == {"结果不对", "SQL 执行报错"}


def test_compare_warns_on_different_question_counts():
    text = format_comparison(compare(_report(0.5, n=10), _report(0.5, n=20)))
    assert "题量不同" in text


def test_format_contains_metric_labels():
    text = format_comparison(compare(_report(0.5), _report(0.6, tokens=400)))
    assert "执行准确率" in text
    assert "平均 token" in text


def test_table_columns_are_visually_aligned():
    """中文占两列，表格必须按显示宽度对齐。"""
    from askdb.eval.compare import _display_width

    lines = format_comparison(compare(_report(0.5), _report(0.8))).splitlines()
    table = [line for line in lines if line and set(line) != {"-"}]
    assert len({_display_width(line) for line in table}) == 1
