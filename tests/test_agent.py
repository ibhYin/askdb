"""Agent 循环测试：SQL 抽取、报错回灌、空结果回灌、步数上限。"""

from __future__ import annotations

from askdb.agent import extract_sql, run_agent
from askdb.config import Settings
from askdb.eval.failure import classify
from askdb.llm import LLMError, MockLLM

DB_PATH = "data/sample/shop.db"


def make_settings(**overrides) -> Settings:
    base = dict(
        base_url="http://localhost:1/v1",
        api_key="",
        model="",
        max_steps=3,
        sql_timeout_seconds=2.0,
        max_rows=200,
        temperature=0.0,
    )
    base.update(overrides)
    return Settings(**base)


def test_extract_sql_from_fence_with_explanation():
    text = "好的，思路是先聚合。\n```sql\nSELECT COUNT(*) FROM customers;\n```\n完成。"
    assert extract_sql(text) == "SELECT COUNT(*) FROM customers"


def test_extract_sql_takes_first_statement_only():
    assert extract_sql("SELECT 1; SELECT 2;") == "SELECT 1"


def test_extract_sql_supports_cte_and_plain_text():
    assert extract_sql("WITH t AS (SELECT 1) SELECT * FROM t").startswith("WITH t")


def test_extract_sql_returns_none_without_select():
    assert extract_sql("我需要更多信息才能回答。") is None


def test_correct_first_try():
    run = run_agent("一共有多少个客户？", DB_PATH, llm=MockLLM("SELECT COUNT(*) FROM customers"), settings=make_settings())
    assert run.ok
    assert run.rows == [(120,)]
    assert len(run.steps) == 1
    assert run.tokens > 0


def test_sql_error_is_fed_back_and_retried():
    llm = MockLLM(["SELECT wrong_column FROM customers", "SELECT COUNT(*) FROM customers"])
    run = run_agent("一共有多少个客户？", DB_PATH, llm=llm, settings=make_settings())

    assert run.ok
    assert run.rows == [(120,)]
    assert len(run.steps) == 2
    # 第二轮请求里必须带上错误信息，模型才有机会改正
    assert "执行失败" in llm.calls[1][-1]["content"]


def test_truly_empty_result_triggers_retry():
    llm = MockLLM(
        [
            "SELECT name FROM customers WHERE city = '火星'",
            "SELECT name FROM customers WHERE city = '北京'",
        ]
    )
    run = run_agent("北京的客户有哪些？", DB_PATH, llm=llm, settings=make_settings())
    assert run.ok
    assert len(run.steps) == 2
    assert "0 行" in llm.calls[1][-1]["content"]


def test_gives_up_after_max_steps():
    llm = MockLLM("SELECT * FROM no_such_table")
    run = run_agent("随便问问", DB_PATH, llm=llm, settings=make_settings(max_steps=2))
    assert not run.ok
    assert len(run.steps) == 2
    assert run.error.startswith("达到最大步数")
    assert classify(run, correct=False) == "max_steps"


def test_model_never_outputs_sql():
    run = run_agent("随便问问", DB_PATH, llm=MockLLM("请再说得具体一些。"), settings=make_settings(max_steps=2))
    assert not run.ok
    assert classify(run, correct=False) == "no_sql"


def test_llm_failure_is_reported_not_raised():
    class BrokenLLM(MockLLM):
        def complete(self, messages, *, temperature=None, max_tokens=None):
            raise LLMError("调用 LLM 失败：网络不可用")

    run = run_agent("一共有多少个客户？", DB_PATH, llm=BrokenLLM(), settings=make_settings())
    assert not run.ok
    assert "网络不可用" in run.error
    assert classify(run, correct=False) == "llm_error"
