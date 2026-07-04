# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import httpx

from .config import Settings


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _headers(self, stream: bool = False) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _client(self) -> httpx.AsyncClient:
        timeout = httpx.Timeout(10, read=self.settings.request_timeout, write=10, pool=10)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        return httpx.AsyncClient(timeout=timeout, limits=limits)

    def _candidate_base_urls(self) -> Iterable[str]:
        yield self.settings.base_url
        if "dashscope" in self.settings.base_url.lower():
            yield "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def _fallback_answer(self, messages: List[Dict[str, str]], reason: str) -> str:
        question = messages[-1]["content"] if messages else ""
        return (
            "当前在线大模型暂时不可用，已切换为本地兜底回答。"
            "我会先根据已检索到的知识文档给出求职建议；如果你需要更完整的生成式回答，"
            "请检查 API Key、Base URL、模型名和网络连接是否正确。\n\n"
            "建议你围绕问题补充岗位方向、目标城市、简历背景或面试场景，我可以继续帮你梳理可执行步骤。"
        )

    async def aclose(self) -> None:
        return None

    def _chunk_text(self, text: str, chunk_size: int = 15) -> Iterable[str]:
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int | None = None,
    ) -> str:
        if not self.settings.api_key or not self.settings.base_url:
            if self.settings.allow_llm_fallback:
                return self._fallback_answer(messages, "Qwen API Key or Base URL not detected")
            raise RuntimeError("Qwen API Key or Base URL not configured")
        if requests is None:
            if self.settings.allow_llm_fallback:
                return self._fallback_answer(messages, "requests not installed in current environment")
            raise RuntimeError("requests not installed in current environment, cannot call online LLM")

        payload = {
            "model": self.settings.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
        }
        response = self._post_chat(payload, stream=False)
        response.encoding = "utf-8"
        data = response.json()
        choice = self._first_choice(data)
        if not choice:
            raise RuntimeError(self._response_error_message(data, "LLM response has no choices"))
        return choice.get("message", {}).get("content", "").strip()

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int | None = None,
    ) -> Iterable[str]:
        if not self.settings.api_key or not self.settings.base_url or requests is None:
            fallback = self._fallback_answer(messages, "Online streaming model unavailable")
            yield from self._chunk_text(fallback)
            return

        payload = {
            "model": self.settings.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "stream": True,
        }
        headers = self._headers(stream=True)
        errors: list[Exception] = []
        for base_url in self._candidate_base_urls():
            url = f"{base_url.rstrip('/')}/chat/completions"
            try:
                with requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=(10, self.settings.request_timeout),
                ) as response:
                    response.raise_for_status()
                    for raw_line in response.iter_lines():
                        line = (raw_line or b"").decode("utf-8").strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            return
                        try:
                            data = json.loads(line)
                            choice = self._first_choice(data)
                            if not choice:
                                continue
                            delta = choice.get("delta", {}).get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue
                return
            except Exception as exc:
                errors.append(exc)
                continue
        if self.settings.allow_llm_fallback:
            fallback = self._fallback_answer(messages, f"Online streaming model call failed: {'; '.join(str(item) for item in errors[-2:])}")
            yield from self._chunk_text(fallback)
            return
        raise RuntimeError("; ".join(str(item) for item in errors[-2:]))

    async def astream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int | None = None,
    ):
        if not self.settings.api_key or not self.settings.base_url:
            fallback = self._fallback_answer(messages, "Online streaming model unavailable")
            for chunk in self._chunk_text(fallback):
                yield chunk
                await asyncio.sleep(0)
            return

        payload = {
            "model": self.settings.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "stream": True,
        }
        errors: list[Exception] = []
        headers = self._headers(stream=True)
        for base_url in self._candidate_base_urls():
            url = f"{base_url.rstrip('/')}/chat/completions"
            try:
                client = self._client()
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for raw_line in response.aiter_lines():
                        line = (raw_line or "").strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            return
                        try:
                            data = json.loads(line)
                            choice = self._first_choice(data)
                            if not choice:
                                continue
                            delta = choice.get("delta", {}).get("content", "")
                            if delta:
                                yield delta
                        except Exception:
                            continue
                return
            except Exception as exc:
                errors.append(exc)
                continue
        if self.settings.allow_llm_fallback:
            fallback = self._fallback_answer(messages, f"Online streaming model call failed: {'; '.join(str(item) for item in errors[-2:])}")
            for chunk in self._chunk_text(fallback):
                yield chunk
                await asyncio.sleep(0)
            return
        raise RuntimeError("; ".join(str(item) for item in errors[-2:]))

    def _post_chat(self, payload: dict, stream: bool):
        assert requests is not None
        headers = self._headers(stream)
        errors: list[Exception] = []
        for base_url in self._candidate_base_urls():
            url = f"{base_url.rstrip('/')}/chat/completions"
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=(10, self.settings.request_timeout),
                    stream=stream,
                )
                response.raise_for_status()
                response.encoding = "utf-8"
                return response
            except Exception as exc:
                errors.append(exc)
                continue
        raise RuntimeError("; ".join(str(item) for item in errors[-2:]))

    async def _apost_chat(self, payload: dict):
        headers = self._headers(stream=False)
        errors: list[Exception] = []
        for base_url in self._candidate_base_urls():
            url = f"{base_url.rstrip('/')}/chat/completions"
            try:
                async with self._client() as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    return response
            except Exception as exc:
                errors.append(exc)
                continue
        raise RuntimeError("; ".join(str(item) for item in errors[-2:]))

    @staticmethod
    def _first_choice(data: dict) -> dict | None:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        return choice if isinstance(choice, dict) else None

    @staticmethod
    def _response_error_message(data: dict, default: str) -> str:
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code")
            if message:
                return str(message)
        return default


requests = None
try:
    import requests
except ImportError:
    pass
