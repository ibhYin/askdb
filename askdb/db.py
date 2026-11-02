"""SQLite 访问层：只读执行、schema 抽取、结果比较。

这里是整个项目的安全边界。模型生成的 SQL 是**不可信输入**，所以：

1. 连接以 `mode=ro` 打开，物理上不能写
2. 执行期挂 authorizer，只放行 SELECT / READ / FUNCTION，双保险
3. 带超时（progress handler 中断）与行数上限，防止一条笛卡尔积把评测卡死
4. 结果比较按多重集（无序）进行，浮点数按 4 位小数归一
"""

from __future__ import annotations

import math
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# 不允许模型调用的函数（sqlite 扩展相关，可能读写文件）
_BLOCKED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    elapsed_ms: float = 0.0
    truncated: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_text(self, max_chars: int = 2000) -> str:
        """回填给模型的文本形式。结果太长时截断，避免上下文被一条查询撑爆。"""
        if not self.ok:
            return f"SQL 执行失败：{self.error}"
        if not self.rows:
            return "查询成功，但结果为空（0 行）。请检查过滤条件是否过严。"
        head = " | ".join(self.columns)
        body = "\n".join(" | ".join(_cell_text(c) for c in row) for row in self.rows)
        text = f"{head}\n{body}"
        if self.truncated:
            text += f"\n（结果超过 {len(self.rows)} 行，已截断）"
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…（已截断）"
        return text


def _cell_text(value) -> str:
    return "NULL" if value is None else str(value)


def _authorizer(action, arg1, arg2, dbname, trigger):
    if action in _ALLOWED_ACTIONS:
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _BLOCKED_FUNCTIONS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    # 其余一律拒绝：INSERT/UPDATE/DELETE/DDL/PRAGMA/ATTACH…
    return sqlite3.SQLITE_DENY


def open_db(path: str | Path) -> sqlite3.Connection:
    """只读打开数据库。"""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = None  # 默认返回 tuple，比较时更直接
    return conn


def run_query(
    conn: sqlite3.Connection,
    sql: str,
    *,
    timeout_seconds: float = 10.0,
    max_rows: int = 200,
) -> QueryResult:
    """执行一条只读查询。任何失败都转成结构化结果，不抛异常。"""
    started = time.perf_counter()
    deadline = started + timeout_seconds

    def _watchdog() -> int:
        return 1 if time.perf_counter() > deadline else 0

    conn.set_authorizer(_authorizer)
    conn.set_progress_handler(_watchdog, 10_000)
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = [tuple(r) for r in fetched[:max_rows]]
        return QueryResult(
            columns=columns,
            rows=rows,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            truncated=truncated,
        )
    except sqlite3.Error as exc:
        message = str(exc)
        if "interrupted" in message:
            message = f"查询超时（>{timeout_seconds:g}s），请缩小范围或加 LIMIT"
        return QueryResult(
            rows=[],
            elapsed_ms=(time.perf_counter() - started) * 1000,
            error=message,
        )
    finally:
        conn.set_progress_handler(None, 0)
        conn.set_authorizer(None)


def list_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def schema_text(
    conn: sqlite3.Connection,
    *,
    include_row_counts: bool = True,
    tables: list[str] | None = None,
) -> str:
    """把库结构渲染成 CREATE TABLE 风格文本，附行数与外键——LLM 最容易读的格式。

    tables 不为空时只渲染这些表（schema 检索用），顺序按传入顺序。
    """
    tables = tables if tables is not None else list_tables(conn)
    blocks: list[str] = []
    for table in tables:
        cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        fks = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()

        lines: list[str] = []
        for cid, name, col_type, notnull, default, pk in cols:
            parts = [f"  {name} {col_type or 'TEXT'}"]
            if pk:
                parts.append("PRIMARY KEY")
            if notnull:
                parts.append("NOT NULL")
            if default is not None:
                parts.append(f"DEFAULT {default}")
            lines.append(" ".join(parts))

        block = f"CREATE TABLE {table} (\n" + ",\n".join(lines) + "\n);"
        annotations: list[str] = []
        if include_row_counts:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            annotations.append(f"-- {table}: {count} 行")
        for fk in fks:
            _, _, ref_table, from_col, to_col, *_ = fk
            annotations.append(f"-- 外键：{table}.{from_col} → {ref_table}.{to_col}")
        blocks.append(block + ("\n" + "\n".join(annotations) if annotations else ""))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 结果比较（评测的核心：执行准确率怎么判）
# ---------------------------------------------------------------------------
def normalize_cell(value):
    """把单元格归一化：浮点四舍五入、字符串去空白、布尔转整数。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return round(value, 4)
    if isinstance(value, bytes):
        return value.hex()
    return str(value).strip()


def _normalize_rows(rows: list[tuple]) -> Counter:
    return Counter(tuple(normalize_cell(v) for v in row) for row in rows)


def results_equal(left: list[tuple], right: list[tuple], *, ordered: bool = False) -> bool:
    """执行准确率判定：默认按无序多重集比较（BIRD 的口径）。

    ordered=True 时额外要求顺序一致，用于"必须排序"的题。
    """
    if len(left) != len(right):
        return False
    if ordered:
        return [tuple(normalize_cell(v) for v in r) for r in left] == [
            tuple(normalize_cell(v) for v in r) for r in right
        ]
    return _normalize_rows(left) == _normalize_rows(right)
