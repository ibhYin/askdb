"""配置：全部来自环境变量 / .env，无隐式默认值。

原则：没有 API Key 就明确走 mock，而不是悄悄用某个在线服务。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    max_steps: int = 6
    sql_timeout_seconds: float = 10.0
    max_rows: int = 200
    temperature: float = 0.0
    # schema 检索：只把相关的几张表放进 prompt
    schema_retrieval: bool = False
    schema_top_k: int = 6
    schema_hops: int = 1

    @property
    def provider(self) -> str:
        """有 key 走真实模型，否则 mock（离线可跑，CI 不需要密钥）。"""
        return "openai" if self.api_key and self.model else "mock"

    @property
    def redacted(self) -> dict:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key": "***" if self.api_key else "",
            "provider": self.provider,
        }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        base_url=(os.getenv("BASE_URL") or "https://api.deepseek.com/v1").strip(),
        api_key=(os.getenv("API_KEY") or "").strip(),
        model=(os.getenv("MODEL") or "").strip(),
        max_steps=_env_int("MAX_STEPS", 6),
        sql_timeout_seconds=_env_float("SQL_TIMEOUT_SECONDS", 10.0),
        max_rows=_env_int("MAX_ROWS", 200),
        temperature=_env_float("TEMPERATURE", 0.0),
        schema_retrieval=(os.getenv("SCHEMA_RETRIEVAL", "").strip().lower() in {"1", "true", "yes"}),
        schema_top_k=_env_int("SCHEMA_TOP_K", 6),
        schema_hops=_env_int("SCHEMA_HOPS", 1),
    )
