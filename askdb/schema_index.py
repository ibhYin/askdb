"""表级 schema 检索：从几十张表里挑出跟问题相关的那几张。

为什么不用向量检索：BIRD 的大库有 8~14 张表、上百个列，但**问题里出现的词
几乎都是标识符的字面片段**（"card" / "set" / "artist"）。词法匹配在这个任务上
信噪比很高，还能离线跑、可解释、可单测——而 embedding 要多一次网络调用、
每次评测都要重算，还引入一个模型作为变量。

评分是 IDF 加权的词命中：常见词（id、name）权重低，稀有词（artist、duel）权重高。
外键邻居可选一并带上，否则模型看到表却看不到连接路径。
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import db as dbmod

# 这些词在问题里高频但对选表没信息量
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "was", "were", "what", "which", "how", "many", "much", "list", "give", "show",
    "find", "me", "please", "that", "this", "with", "by", "from", "as", "at",
    "be", "been", "has", "have", "had", "do", "does", "did", "it", "its", "their",
    "there", "between", "than", "then", "all", "any", "each", "per", "number",
}

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_WORD = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """拆标识符与英文句子；中文用二元组（无需分词器）。"""
    if not text:
        return []

    tokens: list[str] = []
    spaced = _CAMEL.sub(" ", text)
    for chunk in _NON_WORD.split(spaced):
        if not chunk:
            continue
        if _CJK.search(chunk):
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
            continue
        lowered = chunk.lower()
        if lowered in STOPWORDS or len(lowered) < 2:
            continue
        tokens.append(lowered)
        # 复合词再拆一层：UnitPrice -> unit, price；Free Meal Count -> 已在上面拆过
        parts = [p for p in re.split(r"[_\-\s]+", lowered) if len(p) >= 2]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


@dataclass
class TableDoc:
    name: str
    tokens: Counter = field(default_factory=Counter)
    columns: list[str] = field(default_factory=list)
    foreign_keys: list[tuple[str, str]] = field(default_factory=list)

    def as_text(self) -> str:
        return " ".join(f"{tok}×{n}" for tok, n in self.tokens.most_common())


@dataclass
class Match:
    table: str
    score: float
    hits: list[str]


class SchemaIndex:
    """一个库的表级检索索引。构造时只读元数据与少量样例行，不扫全表。"""

    def __init__(self, docs: dict[str, TableDoc]):
        self.docs = docs
        self._idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        total = len(self.docs) or 1
        df: Counter = Counter()
        for doc in self.docs.values():
            df.update(doc.tokens.keys())
        # 出现在所有表里的词没有区分度
        return {tok: math.log(total / count) for tok, count in df.items() if count < total}

    def _better_match(self, token: str, doc: TableDoc) -> tuple[float, str] | None:
        """精确命中优先；否则允许前缀匹配（SAT → satscores），权重打折。

        没有前缀匹配时，像 satscores 这种全小写复合表名永远匹配不上问题里的 "SAT"。
        """
        if token in doc.tokens:
            return self._idf.get(token, 0.0), token
        if len(token) < 3:
            return None
        for candidate in doc.tokens:
            if len(candidate) >= len(token) and candidate.startswith(token):
                return self._idf.get(candidate, 0.0) * 0.6, candidate
        return None

    def search(self, question: str, top_k: int = 6) -> list[Match]:
        query = Counter(tokenize(question))
        if not query:
            return [Match(name, 0.0, []) for name in list(self.docs)[:top_k]]

        matches: list[Match] = []
        for name, doc in self.docs.items():
            score = 0.0
            hits: list[str] = []
            for token in query:
                matched = self._better_match(token, doc)
                if matched is None:
                    continue
                weight, actual = matched
                if weight <= 0:
                    continue
                # 表名命中比列名命中更有指示性
                boost = 1.6 if actual in tokenize(name) else 1.0
                score += weight * boost
                hits.append(actual)
            matches.append(Match(name, round(score, 3), sorted(set(hits))))

        matches.sort(key=lambda m: (-m.score, m.table))
        return matches[:top_k]

    def search_with_neighbors(self, question: str, top_k: int = 6, hops: int = 1) -> list[str]:
        """选中的表 + 它们的外键邻居：只给表不给连接路径，模型照样写不出 JOIN。"""
        chosen = [m.table for m in self.search(question, top_k) if m.score > 0]
        if not chosen:
            chosen = [m.table for m in self.search(question, top_k)]

        selected = list(chosen)
        for _ in range(hops):
            additions: list[str] = []
            for table in list(selected):
                doc = self.docs.get(table)
                if doc is None:
                    continue
                for _, ref in doc.foreign_keys:
                    if ref not in selected and ref not in additions:
                        additions.append(ref)
            selected.extend(additions)
        return selected


def _sample_values(conn: sqlite3.Connection, table: str, limit: int = 5) -> list[str]:
    """取少量样例行里的字符串值，帮助匹配"问题里出现的取值"（如 'DOUBLE' 这种枚举）。"""
    values: list[str] = []
    try:
        rows = conn.execute(f'SELECT * FROM "{table}" LIMIT {limit}').fetchall()
    except sqlite3.Error:
        return values
    for row in rows:
        for cell in row:
            if isinstance(cell, str) and 1 < len(cell) <= 30:
                values.append(cell)
    return values


def build_index(
    conn: sqlite3.Connection,
    *,
    sample_rows: int = 3,
    max_sample_values: int = 60,
) -> SchemaIndex:
    docs: dict[str, TableDoc] = {}
    for table in dbmod.list_tables(conn):
        doc = TableDoc(name=table)
        doc.tokens.update(tokenize(table))
        doc.tokens.update(tokenize(table) * 2)  # 表名权重加倍

        for col in conn.execute(f'PRAGMA table_info("{table}")').fetchall():
            column = col[1]
            doc.columns.append(column)
            doc.tokens.update(tokenize(column))

        for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall():
            doc.foreign_keys.append((fk[3], fk[2]))

        for value in _sample_values(conn, table, sample_rows)[:max_sample_values]:
            doc.tokens.update(tokenize(value))

        docs[table] = doc
    return SchemaIndex(docs)


@lru_cache(maxsize=32)
def cached_index(db_path: str) -> SchemaIndex:
    conn = dbmod.open_db(db_path)
    try:
        return build_index(conn)
    finally:
        conn.close()


def retrieve_schema(
    conn: sqlite3.Connection,
    question: str,
    *,
    evidence: str = "",
    db_path: str | Path | None = None,
    top_k: int = 6,
    hops: int = 1,
    include_row_counts: bool = False,
) -> tuple[str, list[str]]:
    """返回 (渲染好的 schema 文本, 选中的表名)。"""
    if db_path is not None:
        index = cached_index(str(Path(db_path).resolve()))
    else:
        index = build_index(conn)
    # evidence 也是题目的一部分（BIRD 官方口径），一起参与检索
    query = f"{question}\n{evidence}".strip()
    tables = index.search_with_neighbors(query, top_k=top_k, hops=hops)
    return (
        dbmod.schema_text(conn, include_row_counts=include_row_counts, tables=tables),
        tables,
    )
