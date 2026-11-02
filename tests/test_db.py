"""SQL 执行层测试：安全边界、超时、结果比较。"""

from __future__ import annotations

import pytest

from askdb import db

DB_PATH = "data/sample/shop.db"


@pytest.fixture()
def conn():
    connection = db.open_db(DB_PATH)
    yield connection
    connection.close()


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE orders SET status = 'x'",
        "DROP TABLE orders",
        "CREATE TABLE evil (x)",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA writable_schema = 1",
        "SELECT load_extension('evil')",
    ],
)
def test_write_and_dangerous_sql_are_denied(conn, sql):
    result = db.run_query(conn, sql)
    assert not result.ok
    assert "not authorized" in result.error


def test_database_survives_denied_write(conn):
    before = db.run_query(conn, "SELECT COUNT(*) FROM customers").rows[0][0]
    db.run_query(conn, "DELETE FROM customers")
    after = db.run_query(conn, "SELECT COUNT(*) FROM customers").rows[0][0]
    assert before == after == 120


def test_timeout_interrupts_runaway_query(conn):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    result = db.run_query(conn, sql, timeout_seconds=0.05)
    assert not result.ok
    assert "超时" in result.error


def test_row_limit_truncates(conn):
    result = db.run_query(conn, "SELECT * FROM order_items", max_rows=10)
    assert result.ok
    assert len(result.rows) == 10
    assert result.truncated is True


def test_schema_text_has_tables_counts_and_foreign_keys(conn):
    text = db.schema_text(conn)
    assert "CREATE TABLE customers" in text
    assert "-- customers: 120 行" in text
    assert "外键：orders.customer_id → customers.customer_id" in text


def test_results_equal_is_order_insensitive_by_default():
    left = [(1, 2.0), (3, 4.0)]
    right = [(3, 4.0), (1, 2.0)]
    assert db.results_equal(left, right)
    assert not db.results_equal(left, right, ordered=True)


def test_results_equal_tolerates_float_noise_and_ints_as_floats():
    assert db.results_equal([(12.0000001,)], [(12.0,)])
    assert db.results_equal([(12,)], [(12.0,)])
    assert not db.results_equal([(12.5,)], [(12.4,)])


def test_results_equal_counts_duplicates():
    assert not db.results_equal([(1,), (1,)], [(1,), (2,)])
    assert db.results_equal([(1,), (1,)], [(1,), (1,)])


def test_value_samples_only_annotate_low_cardinality_columns(conn):
    text = db.schema_text(conn, include_row_counts=False, tables=["customers"], value_samples=4)
    # level / city 在抽样行内取值有限，给出来有助于写对过滤条件
    assert "customers.level 取值示例" in text
    assert "customers.city 取值示例" in text
    # name 高基数，绝不能当枚举写进 prompt
    assert "customers.name 取值示例" not in text


def test_value_samples_are_off_by_default(conn):
    assert "取值示例" not in db.schema_text(conn)


def test_high_cardinality_column_returns_no_samples(conn):
    assert db.sample_column_values(conn, "customers", "name") == []
    assert db.sample_column_values(conn, "customers", "level") == ["普通", "银卡", "金卡"]
    assert 2 <= len(db.sample_column_values(conn, "customers", "city")) <= 4
