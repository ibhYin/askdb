"""Agent 循环：问题 → SQL → 执行 → 用执行结果纠错 → 再执行。

刻意比常见实现多两条约束，因为它们直接影响执行准确率：

1. **只接受 SELECT / WITH 开头的单条语句**，其余视为无效输出
2. **空结果也要回灌**：0 行通常意味着过滤条件写错或 JOIN 漏了，让模型看见再改

每一步都被记录下来（SQL、错误、耗时、token），评测器和前端 trace 都读它。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from . import db as dbmod
from . import prompts
from .config import Settings, get_settings
from .llm import BaseLLM, LLMError, LLMResponse, Message, build_llm

SYSTEM_PROMPT = prompts.get_prompt()


@dataclass
class Step:
    index: int
    sql: str | None
    response: LLMResponse
    result: dbmod.QueryResult | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.ok and bool(self.result.rows)


@dataclass
class AgentRun:
    question: str
    steps: list[Step] = field(default_factory=list)
    answer_sql: str | None = None
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def prompt_tokens(self) -> int:
        return sum(s.response.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.response.completion_tokens for s in self.steps)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def ok(self) -> bool:
        return self.error is None and self.answer_sql is not None


def extract_sql(text: str) -> str | None:
    """从模型回复里剥出 SQL。容忍 ```sql 代码块、前缀说明、末尾分号。"""
    if not text:
        return None
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    candidate = fenced.group(1) if fenced else text

    match = re.search(r"\b(select|with)\b", candidate, re.I)
    if not match:
        return None
    sql = candidate[match.start():].strip()
    sql = sql.split(";")[0].strip()  # 只取第一条语句
    return sql or None


def build_messages(
    question: str,
    schema: str,
    evidence: str = "",
    *,
    prompt_version: str | None = None,
) -> list[Message]:
    """evidence 是数据集给出的业务知识提示（BIRD 的标准字段），不是从库里能查到的东西。"""
    user = question
    if evidence:
        user = f"{question}\n\n参考信息：{evidence}"
    system_prompt = prompts.get_prompt(prompt_version) if prompt_version else SYSTEM_PROMPT
    return [
        Message("system", f"{system_prompt}\n\n数据库 schema：\n{schema}"),
        Message("user", user),
    ]


def stream_agent(
    question: str,
    db_path: str | Path,
    *,
    llm: BaseLLM | None = None,
    settings: Settings | None = None,
    schema: str | None = None,
    evidence: str = "",
    prompt_version: str | None = None,
) -> Iterator[dict]:
    """流式版本：每走一步就吐一个事件，Web 端直接转成 SSE。

    事件类型：
        step   一轮开始
        sql    模型给出的 SQL（可能是 None，表示没解析出来）
        result 执行结果（含列名、行、错误、耗时）
        error  调用失败
        finish 结束，携带完整 AgentRun（供评测/持久化）
    """
    settings = settings or get_settings()
    llm = llm or build_llm(settings)

    conn = dbmod.open_db(db_path)
    schema = schema if schema is not None else dbmod.schema_text(conn)
    messages = build_messages(question, schema, evidence, prompt_version=prompt_version)

    run = AgentRun(question=question)
    started = time.perf_counter()
    best: dbmod.QueryResult | None = None

    try:
        for index in range(1, settings.max_steps + 1):
            yield {"type": "step", "index": index}

            try:
                response = llm.complete(messages)
            except LLMError as exc:
                run.error = str(exc)
                yield {"type": "error", "message": run.error}
                break

            sql = extract_sql(response.text)
            step = Step(index=index, sql=sql, response=response)
            run.steps.append(step)
            yield {"type": "sql", "index": index, "sql": sql}

            if sql is None:
                messages.append(Message("assistant", response.text))
                messages.append(
                    Message("user", "你没有输出可执行的 SQL。请只输出一条 ```sql 代码块形式的 SELECT 查询。")
                )
                continue

            result = dbmod.run_query(
                conn,
                sql,
                timeout_seconds=settings.sql_timeout_seconds,
                max_rows=settings.max_rows,
            )
            step.result = result
            messages.append(Message("assistant", response.text))
            yield {
                "type": "result",
                "index": index,
                "columns": result.columns,
                "rows": result.rows,
                "error": result.error,
                "elapsed_ms": round(result.elapsed_ms, 1),
                "truncated": result.truncated,
            }

            if not result.ok:
                messages.append(
                    Message("user", f"SQL 执行失败：{result.error}\n请修正后重新输出完整的 SQL。")
                )
                continue

            best = result
            if result.rows:
                run.answer_sql = sql
                run.rows = result.rows
                break

            # 执行成功但 0 行：多半是过滤条件或 JOIN 写错，回灌让模型自查
            messages.append(
                Message(
                    "user",
                    "这条 SQL 执行成功但返回 0 行。请检查过滤条件是否过严、是否漏了 JOIN，"
                    "再输出修正后的 SQL。",
                )
            )
        else:
            run.error = f"达到最大步数（{settings.max_steps}）仍未得到有效结果"

        if run.answer_sql is None and best is not None:
            # 始终为空但有成功执行过：保留最后一条可执行 SQL 作为答案，交给评测判错
            run.answer_sql = run.steps[-1].sql
            run.rows = best.rows
    finally:
        conn.close()

    run.latency_ms = (time.perf_counter() - started) * 1000
    yield {"type": "finish", "run": run}


def run_agent(
    question: str,
    db_path: str | Path,
    *,
    llm: BaseLLM | None = None,
    settings: Settings | None = None,
    schema: str | None = None,
    evidence: str = "",
    prompt_version: str | None = None,
) -> AgentRun:
    """同步版本：把事件流跑完，返回最终结果（评测用）。"""
    for event in stream_agent(
        question,
        db_path,
        llm=llm,
        settings=settings,
        schema=schema,
        evidence=evidence,
        prompt_version=prompt_version,
    ):
        if event["type"] == "finish":
            return event["run"]
    raise RuntimeError("agent 事件流异常结束：没有 finish 事件")
