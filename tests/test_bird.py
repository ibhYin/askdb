"""BIRD 评测集加载测试。"""

from __future__ import annotations

import json
import sqlite3

import pytest

from askdb.eval.bird import BirdDataMissing, database_path, load_bird_cases, summarize
from askdb.eval.runner import run_self_check


def _make_db(path, table="t", rows=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE {table} (x INTEGER)")
    conn.executemany(f"INSERT INTO {table} VALUES (?)", [(i,) for i in range(rows)])
    conn.commit()
    conn.close()


def _write_questions(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n", encoding="utf-8"
    )


@pytest.fixture()
def bird_dir(tmp_path):
    dbs = tmp_path / "databases"
    _make_db(database_path("shop", dbs), "orders", 3)
    _make_db(database_path("clinic", dbs), "patients", 2)
    questions = tmp_path / "mini_dev_sqlite.jsonl"
    _write_questions(
        questions,
        [
            {
                "question_id": 0,
                "db_id": "shop",
                "question": "How many orders?",
                "evidence": "",
                "SQL": "SELECT COUNT(*) FROM orders",
                "difficulty": "simple",
            },
            {
                "question_id": 1,
                "db_id": "clinic",
                "question": "How many patients?",
                "evidence": "patients 是患者表",
                "SQL": "SELECT COUNT(*) FROM patients",
                "difficulty": "moderate",
            },
            {
                "question_id": 2,
                "db_id": "ghost",
                "question": "How many ghosts?",
                "evidence": "",
                "SQL": "SELECT COUNT(*) FROM ghosts",
                "difficulty": "challenging",
            },
        ],
    )
    return questions, dbs


def test_loads_cases_with_per_case_database(bird_dir):
    questions, dbs = bird_dir
    cases = load_bird_cases(questions_path=questions, databases_dir=dbs)
    # ghost 库没下载，默认跳过
    assert [c.id for c in cases] == ["bird-0", "bird-1"]
    assert cases[0].db_path.endswith("shop/shop.sqlite")
    assert cases[1].evidence == "patients 是患者表"
    assert cases[1].tags == ("clinic",)
    assert all(c.ordered is False for c in cases)


def test_filters_by_db_and_difficulty(bird_dir):
    questions, dbs = bird_dir
    only_shop = load_bird_cases(questions_path=questions, databases_dir=dbs, db_id="shop")
    assert len(only_shop) == 1

    only_moderate = load_bird_cases(
        questions_path=questions, databases_dir=dbs, difficulty="moderate"
    )
    assert [c.id for c in only_moderate] == ["bird-1"]


def test_missing_questions_file_raises(tmp_path):
    with pytest.raises(BirdDataMissing):
        load_bird_cases(questions_path=tmp_path / "nope.jsonl", databases_dir=tmp_path)


def test_summarize_counts_by_db_and_difficulty(bird_dir):
    questions, dbs = bird_dir
    text = summarize(load_bird_cases(questions_path=questions, databases_dir=dbs))
    assert "题量 2" in text
    assert "shop 1" in text
    assert "moderate 1" in text


def test_multi_database_self_check_is_100(bird_dir):
    """评测器在多库场景下同样要 100%：证明每道题确实连到了自己的库。"""
    questions, dbs = bird_dir
    cases = load_bird_cases(questions_path=questions, databases_dir=dbs)
    report = run_self_check(cases, dbs)
    assert report.metrics["exact_match"] == 1.0


def test_invalid_questions_are_quarantined_by_default(bird_dir, monkeypatch):
    from askdb.eval import bird

    questions, dbs = bird_dir
    invalid_file = questions.parent / "invalid_ids.json"
    invalid_file.write_text(
        json.dumps({"checked": 2, "invalid": [{"question_id": 1, "db_id": "clinic"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(bird, "INVALID_PATH", invalid_file)

    cases = bird.load_bird_cases(questions_path=questions, databases_dir=dbs)
    assert [c.id for c in cases] == ["bird-0"]

    kept = bird.load_bird_cases(
        questions_path=questions, databases_dir=dbs, include_invalid=True
    )
    assert [c.id for c in kept] == ["bird-0", "bird-1"]
