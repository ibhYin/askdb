"""LLM 客户端：OpenAI 兼容协议 + 离线 mock。

为什么抽一层：评测要能离线跑（CI、没网、没 key），也要能一行切换到
DeepSeek / Moonshot / Qwen / 本地 vLLM。Agent 只依赖 `BaseLLM.complete`。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .config import Settings, get_settings


class LLMError(RuntimeError):
    """调用失败（网络 / 鉴权 / 限流），消息已转成可读中文。"""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class BaseLLM:
    name = "base"

    def complete(
        self,
        messages: Sequence[Message | dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:  # pragma: no cover - 抽象方法
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    """任何 OpenAI 兼容服务（DeepSeek / Kimi / Qwen / 硅基流动 / vLLM / Ollama）。"""

    name = "openai"

    def __init__(self, settings: Settings):
        from openai import OpenAI  # 延迟导入：mock 模式下不需要

        self._settings = settings
        self._client = OpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=60.0)
        self.name = f"openai:{settings.model}"

    def complete(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse:
        payload = [m.to_dict() if isinstance(m, Message) else dict(m) for m in messages]
        started = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self._settings.model,
                messages=payload,
                temperature=self._settings.temperature if temperature is None else temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - 统一转成可读错误
            raise LLMError(_friendly_error(exc)) from exc

        choice = resp.choices[0] if resp.choices else None
        text = (choice.message.content or "") if choice else ""
        usage = resp.usage
        return LLMResponse(
            text=text.strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=(time.perf_counter() - started) * 1000,
            model=getattr(resp, "model", "") or self._settings.model,
        )


class MockLLM(BaseLLM):
    """离线替身。可以喂固定回复列表，也可以喂一个函数，用于单测与流水线自检。"""

    name = "mock"

    def __init__(self, responder: Callable[[list[dict]], str] | Sequence[str] | str = "SELECT 1"):
        self._responder = responder
        self.calls: list[list[dict]] = []

    def _next_reply(self, payload: list[dict]) -> str:
        if callable(self._responder):
            return self._responder(payload)
        if isinstance(self._responder, str):
            return self._responder
        if not self._responder:
            return "SELECT 1"
        # 列表：按调用顺序消费，用完停在最后一条
        index = min(len(self.calls) - 1, len(self._responder) - 1)
        return self._responder[index]

    def complete(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse:
        payload = [m.to_dict() if isinstance(m, Message) else dict(m) for m in messages]
        self.calls.append(payload)
        text = self._next_reply(payload)
        return LLMResponse(
            text=text,
            prompt_tokens=sum(len(m["content"]) for m in payload) // 4,
            completion_tokens=len(text) // 4,
            model="mock",
        )


def build_llm(settings: Settings | None = None, *, mock_responder=None) -> BaseLLM:
    settings = settings or get_settings()
    if settings.provider == "mock":
        return MockLLM(mock_responder if mock_responder is not None else "SELECT 1")
    return OpenAICompatibleLLM(settings)


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "401" in text or "unauthorized" in lowered or "invalid_api_key" in lowered:
        return "认证失败（401）：API_KEY 无效，请检查 .env"
    if "404" in text or "not found" in lowered:
        return "接口地址错误（404）：检查 BASE_URL，应形如 https://api.xxx.com/v1"
    if "429" in text or "rate limit" in lowered:
        return "触发限流（429）：降低并发或稍后重试"
    if "timeout" in lowered:
        return "请求超时：检查网络或服务商状态"
    return f"调用 LLM 失败：{text}"
