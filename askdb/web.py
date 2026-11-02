"""Web 界面：一个问题一个页面，SQL 与执行结果逐步流式展示。

    uvicorn askdb.web:app --reload --port 8020

之所以把 SSE 做在一等公民的位置：这个项目卖的是"过程可见"，
用户能看到模型写了什么 SQL、执行报了什么错、第几轮改对的。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import stream_agent
from .config import PROJECT_ROOT, get_settings
from .eval.bird import DATABASES_DIR, database_path
from .llm import BaseLLM, build_llm

STATIC_DIR = Path(__file__).resolve().parent / "web_static"
SAMPLE_DB = PROJECT_ROOT / "data" / "sample" / "shop.db"

app = FastAPI(title="askdb", description="自然语言问数 Agent")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    db: str = "sample"


def list_databases() -> list[dict]:
    items = [{"id": "sample", "label": "样例电商库"}]
    if DATABASES_DIR.exists():
        for path in sorted(DATABASES_DIR.glob("*/*.sqlite")):
            items.append({"id": f"bird:{path.stem}", "label": f"BIRD · {path.stem}"})
    return items


def resolve_db(db_id: str) -> Path:
    if db_id == "sample":
        if not SAMPLE_DB.exists():
            raise HTTPException(500, "样例库不存在，先运行 python scripts/build_sample_db.py")
        return SAMPLE_DB
    if db_id.startswith("bird:"):
        path = database_path(db_id.split(":", 1)[1])
        if not path.exists():
            raise HTTPException(404, f"数据库未下载：{db_id}")
        return path
    raise HTTPException(400, f"未知数据库：{db_id}")


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def answer_stream(question: str, db_path: Path, llm: BaseLLM | None = None):
    """把 agent 事件流翻译成 SSE 文本流。抽成独立函数，方便不启服务就测。"""
    settings = get_settings()
    llm = llm or build_llm(settings)
    try:
        for event in stream_agent(question, db_path, llm=llm, settings=settings):
            if event["type"] == "finish":
                run = event["run"]
                yield _sse(
                    {
                        "type": "finish",
                        "sql": run.answer_sql,
                        "row_count": len(run.rows),
                        "steps": len(run.steps),
                        "tokens": run.tokens,
                        "latency_ms": round(run.latency_ms),
                        "error": run.error,
                    }
                )
                continue
            yield _sse(event)
    except Exception as exc:  # noqa: BLE001 - 流里出错要让前端看到，而不是静默断开
        yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/databases")
def databases() -> dict:
    return {"databases": list_databases()}


@app.post("/api/ask")
def ask(payload: AskRequest):
    db_path = resolve_db(payload.db)
    return StreamingResponse(
        answer_stream(payload.question.strip(), db_path),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
