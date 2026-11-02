"""表级 schema 检索测试。"""

from __future__ import annotations

import sqlite3

import pytest

from askdb.db import open_db
from askdb.schema_index import (
    SchemaIndex,
    TableDoc,
    build_index,
    retrieve_schema,
    tokenize,
)

DB_PATH = "data/sample/shop.db"


def test_tokenize_splits_identifiers_and_drops_stopwords():
    tokens = tokenize("How many UnitPrice values are in the order_items table?")
    assert "unit" in tokens and "price" in tokens
    assert "order" in tokens and "items" in tokens
    assert "the" not in tokens and "how" not in tokens


def test_tokenize_handles_chinese_without_segmenter():
    tokens = tokenize("每个城市的客户数量")
    assert "城市" in tokens
    assert "客户" in tokens


def test_tokenize_camel_case():
    assert tokenize("duelDeck") == ["duel", "deck"]


@pytest.fixture()
def shop_index():
    conn = open_db(DB_PATH)
    index = build_index(conn)
    conn.close()
    return index


def test_index_covers_every_table(shop_index):
    assert set(shop_index.docs) == {
        "customers",
        "products",
        "orders",
        "order_items",
        "payments",
        "refunds",
    }


def test_search_finds_table_by_question_words(shop_index):
    top = shop_index.search("How many customers are there in each city?", top_k=3)
    assert top[0].table == "customers"
    assert top[0].score > 0


def test_search_ranks_relevant_table_first(shop_index):
    top = shop_index.search("What is the total refund amount?", top_k=3)
    assert top[0].table == "refunds"


def test_idf_downweights_common_tokens(shop_index):
    # "id" 出现在所有表的列名里，不该有区分度
    assert shop_index._idf.get("id", 0.0) == 0.0


def test_prefix_match_links_abbreviation_to_table():
    """SAT 必须能匹配到 satscores：全小写复合表名没有分词，只能靠前缀。"""
    index = SchemaIndex(
        {
            "satscores": TableDoc(name="satscores", tokens={"satscores": 2}),
            "frpm": TableDoc(name="frpm", tokens={"frpm": 2, "meal": 1}),
        }
    )
    top = index.search("school-level SAT excellence rate", top_k=1)
    assert top[0].table == "satscores"


def test_neighbors_pull_in_foreign_key_tables(shop_index):
    tables = shop_index.search_with_neighbors("What is the highest refund amount?", top_k=1, hops=1)
    assert "refunds" in tables
    assert "orders" in tables  # refunds.order_id → orders


def test_retrieve_schema_returns_text_and_tables():
    conn = open_db(DB_PATH)
    text, tables = retrieve_schema(conn, "What is the highest refund amount?", top_k=2)
    conn.close()
    assert "refunds" in tables
    rendered = [line for line in text.splitlines() if line.startswith("CREATE TABLE")]
    assert len(rendered) <= len(tables)


def test_retrieved_schema_is_smaller_than_full():
    conn = open_db(DB_PATH)
    from askdb.db import schema_text

    full = schema_text(conn)
    trimmed, _ = retrieve_schema(conn, "What is the highest refund amount?", top_k=2, include_row_counts=False)
    conn.close()
    assert len(trimmed) < len(full)


def test_chinese_question_cannot_match_english_schema():
    """已知局限：样例库是英文 schema + 中文题面，词法匹配跨语言失效。

    这正是 schema 检索只对 BIRD（英文题 + 英文 schema）开启、默认关闭的原因。
    """
    conn = open_db(DB_PATH)
    _, tables = retrieve_schema(conn, "退款金额最高的三笔退款", top_k=2)
    conn.close()
    assert "refunds" not in tables


def test_evidence_participates_in_retrieval():
    conn = open_db(DB_PATH)
    text, tables = retrieve_schema(
        conn, "What are the top three amounts?", evidence="Refunds are stored in the refunds table", top_k=2
    )
    conn.close()
    assert "refunds" in tables
