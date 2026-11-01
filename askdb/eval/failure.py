"""失败归因：把一次运行归到唯一类别。

归因的价值在于**下一步动作**——不同类别对应完全不同的改法：

- no_sql        模型没给 SQL       → 改 prompt / 输出解析
- sql_error     执行报错           → 加 schema 约束 / 错误回灌
- empty_result  执行成功但 0 行    → 加数据探查 / 过滤条件提示
- wrong_result  能跑但答案错       → 选表、JOIN、聚合、口径问题（最值得深挖）
- max_steps     反复失败不收敛     → 循环控制 / 重试上限
- llm_error     调用失败           → 与模型能力无关，需从分母里剔除
"""

from __future__ import annotations

from ..agent import AgentRun

CATEGORIES = (
    "correct",
    "no_sql",
    "sql_error",
    "empty_result",
    "wrong_result",
    "max_steps",
    "llm_error",
)

CATEGORY_LABELS = {
    "correct": "答对",
    "no_sql": "未输出 SQL",
    "sql_error": "SQL 执行报错",
    "empty_result": "结果为空",
    "wrong_result": "结果不对",
    "max_steps": "不收敛",
    "llm_error": "调用失败",
}


def classify(run: AgentRun, *, correct: bool) -> str:
    if correct:
        return "correct"
    if not run.steps:
        return "llm_error" if run.error else "no_sql"
    if all(step.sql is None for step in run.steps):
        # 模型全程没给过 SQL：问题在输出格式/提示，而不是 SQL 写错
        return "llm_error" if run.error and not run.error.startswith("达到最大步数") else "no_sql"
    if run.error:
        if run.error.startswith("达到最大步数"):
            return "max_steps"
        return "llm_error"

    last = run.steps[-1]
    if last.result is None or not last.result.ok:
        return "sql_error"
    if not last.result.rows:
        return "empty_result"
    return "wrong_result"


def detail_of(run: AgentRun) -> str:
    """一句人话，用于报告里定位问题。"""
    if run.error:
        return run.error
    for step in reversed(run.steps):
        if step.result is not None and not step.result.ok:
            return step.result.error or ""
    return ""
