"""Web 层测试：页面、数据库列表、SSE 事件流。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from askdb.config import get_settings
from askdb.llm import MockLLM
from askdb.web import SAMPLE_DB, answer_stream, app, list_databases, resolve_db


@pytest.fixture()
def client():
    return TestClient(app)


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "askdb" in response.text
    assert "/api/ask" in response.text


def test_database_list_contains_sample(client):
    data = client.get("/api/databases").json()
    ids = [item["id"] for item in data["databases"]]
    assert "sample" in ids
    assert all(item["id"] != "bird:missing" for item in data["databases"])


def test_unknown_database_is_rejected(client):
    response = client.post("/api/ask", json={"question": "x", "db": "nope"})
    assert response.status_code == 400


def test_empty_question_is_rejected(client):
    response = client.post("/api/ask", json={"question": "", "db": "sample"})
    assert response.status_code == 422


def test_resolve_sample_database():
    assert resolve_db("sample") == SAMPLE_DB


def test_answer_stream_emits_sse_events():
    llm = MockLLM("SELECT COUNT(*) FROM customers")
    chunks = list(answer_stream("一共有多少个客户？", SAMPLE_DB, llm=llm))

    payloads = [json.loads(chunk.removeprefix("data: ").strip()) for chunk in chunks]
    types = [p["type"] for p in payloads]
    assert types == ["step", "sql", "result", "finish"]
    assert payloads[2]["rows"] == [[120]]  # JSON 化后 tuple 变 list
    assert payloads[-1]["row_count"] == 1
    assert payloads[-1]["error"] is None
    assert all(chunk.endswith("\n\n") for chunk in chunks)


def test_answer_stream_reports_errors_as_events():
    llm = MockLLM("SELECT * FROM no_such_table")

    class Boom(MockLLM):
        def complete(self, messages, *, temperature=None, max_tokens=None):
            raise RuntimeError("炸了")

    chunks = list(answer_stream("随便问问", SAMPLE_DB, llm=Boom()))
    payloads = [json.loads(chunk.removeprefix("data: ").strip()) for chunk in chunks]
    assert payloads[-1]["type"] == "error"
    assert "炸了" in payloads[-1]["message"]
