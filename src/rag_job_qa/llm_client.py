from __future__ import annotations

import json
from typing import Dict, Iterable, List

try:
    import requests
except Exception:  # pragma: no cover - requests 是可选依赖，缺失时走离线兜底
    requests = None

from .config import Settings


DASHSCOPE_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class LLMClient:
    """OpenAI-compatible Qwen client with a local fallback for internship demos."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 1000,
    ) -> str:
        if not self.settings.api_key or not self.settings.base_url:
            if self.settings.allow_llm_fallback:
                return self._fallback_answer(messages, "未检测到 Qwen API Key 或 Base URL")
            raise RuntimeError("未配置 Qwen API Key 或 Base URL")
        if requests is None:
            if self.settings.allow_llm_fallback:
                return self._fallback_answer(messages, "当前环境未安装 requests")
            raise RuntimeError("当前环境未安装 requests，无法调用在线大模型")

        payload = {
            "model": self.settings.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        try:
            response = self._post_chat(payload, stream=False)
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            if self.settings.allow_llm_fallback:
                return self._fallback_answer(messages, f"在线模型调用失败：{exc}")
            raise

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 1000,
    ) -> Iterable[str]:
        if not self.settings.api_key or not self.settings.base_url or requests is None:
            fallback = self._fallback_answer(messages, "在线流式模型不可用")
            yield from self._chunk_text(fallback)
            return

        payload = {
            "model": self.settings.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            with self._post_chat(payload, stream=True) as response:
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                        delta = data["choices"][0].get("delta", {}).get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
        except Exception as exc:
            if self.settings.allow_llm_fallback:
                fallback = self._fallback_answer(messages, f"在线流式模型调用失败：{exc}")
                yield from self._chunk_text(fallback)
                return
            raise

    def _post_chat(self, payload: dict, stream: bool):
        """Call configured base_url; retry official DashScope URL when Maas SSL endpoint fails."""
        assert requests is not None
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
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
                return response
            except Exception as exc:
                errors.append(exc)
                continue
        raise RuntimeError("; ".join(str(item) for item in errors[-2:]))

    def _candidate_base_urls(self) -> list[str]:
        urls = [self.settings.base_url.rstrip("/")]
        # 部分业务空间 CSV 给出的 maas.aliyuncs.com 地址在校园网环境可能 SSL EOF；
        # Qwen 的 OpenAI 兼容调用优先兜底到 DashScope 官方兼容地址。
        if self.settings.api_key.startswith("sk-") and DASHSCOPE_OPENAI_BASE_URL not in urls:
            urls.append(DASHSCOPE_OPENAI_BASE_URL)
        return urls

    @staticmethod
    def _chunk_text(text: str, size: int = 8) -> Iterable[str]:
        for index in range(0, len(text), size):
            yield text[index : index + size]

    def _fallback_answer(self, messages: List[Dict[str, str]], reason: str) -> str:
        user_content = messages[-1]["content"] if messages else ""
        marker = "【检索到的知识片段】"
        context = user_content.split(marker, 1)[-1] if marker in user_content else user_content
        compact = "\n".join(line.strip() for line in context.splitlines() if line.strip())
        excerpt = compact[:900] or "当前知识库没有召回到足够内容。"
        return (
            f"【离线演示回答】当前未使用在线大模型生成，原因：{reason}\n\n"
            "根据知识库可参考的信息，建议你先围绕岗位要求、个人经历匹配度、简历表达和面试准备四个方面梳理问题。"
            "如果问题涉及薪资、具体政策或公司要求，需要以知识库原文和官方招聘信息为准。\n\n"
            f"可参考片段：\n{excerpt}"
        )
