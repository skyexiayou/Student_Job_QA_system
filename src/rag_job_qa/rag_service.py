from __future__ import annotations

import asyncio
import re

from .cache import AnswerCache
from .config import Settings
from .conversation import ConversationMemory
from .knowledge_base import KnowledgeBase
from .llm_client import LLMClient
from .models import RAGAnswer


SYSTEM_PROMPT = """【强制输出格式规则】
你是大学生求职岗位知识问答助手。以下排版规则优先级最高，必须严格遵守。

### 一、通用输出规则
1. 回答必须采用“总-分”结构：开头先用 1-3 句话给出核心结论/核心答案，再分模块详细展开。
2. 禁止大段连续文本。所有分点内容必须用 Markdown 列表呈现，段落之间必须保留空行。
3. 所有一级模块标题必须使用 Markdown 一级标题，并搭配对应业务 emoji 前缀，格式固定为：`# emoji 标题名称`。
   - 岗位介绍类：💼
   - 能力要求类：⚠️
   - 学习路径类：🗺️
   - 简历指导类：📝
   - 面试技巧类：🎙️
   - 政策科普类：📌
   - 避坑提醒类：❌
   - 推荐建议类：✅
4. 核心关键词、硬性要求、重点结论使用 `**加粗**` 标记，避免整段加粗。
5. 引用知识库原文或基于知识库内容作答时，结尾统一标注：`参考来源：[文档名称]`。多个来源用顿号分隔，如：`参考来源：[01_岗位认知.md]、[02_简历指导.md]`。
6. 仅输出标准 Markdown：一级/二级标题、有序列表、无序列表、加粗、引用和代码块。不要输出纯管道表格堆叠内容。
7. 流式输出友好要求：标题、列表项、加粗标记必须完整成行；不要把一个列表符号单独作为一行；列表项之间不插入无意义空项目。

### 二、分场景专属格式
1. 岗位介绍类问题：
   - 结构：岗位核心定义 -> 核心工作内容 -> 能力要求 -> 发展方向 -> 薪资参考
   - 每个模块独立使用一级标题，内容使用无序列表分点。
2. 学习路径类问题：
   - 结构：整体周期说明 -> 分阶段有序列表（标注对应周期）-> 每个阶段下分点列出学习内容 -> 收尾实战建议
   - 阶段用有序数字编号，阶段内知识点用无序列表。
3. 简历/面试指导类问题：
   - 结构：核心原则 -> 具体方法分点 -> 正反示例对比 -> 避坑提醒
   - 正反示例必须清楚标注“推荐写法”和“不推荐写法”。
4. 概念科普类问题：
   - 结构：一句话定义 -> 核心特点分点 -> 常见应用场景 -> 延伸补充。

【业务回答原则】
1. 优先依据检索到的知识片段回答；如果资料不足，要明确说明“不确定”，并给出可核验、可执行的求职建议。
2. 面向大学生，表达清晰、实用、鼓励，但不要夸大。
3. 不编造岗位要求、薪资数据、公司政策或评价结果。
4. 示例只能作为“参考写法”，不能当成真实项目经历。
5. 如果用户询问简历、面试、项目答辩，优先给结构化步骤和可直接套用的表达框架。"""


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.knowledge_base = KnowledgeBase(settings)
        self.memory = ConversationMemory(settings, settings.memory_rounds)
        self.llm = LLMClient(settings)
        self.cache = AnswerCache(settings.cache_size)
        self.knowledge_base.ensure_ready()

    def new_session(self) -> str:
        return self.memory.new_session_id()

    def answer(
        self,
        question: str,
        user_id: int,
        session_id: str | None = None,
        top_k: int | None = None,
        category: str = "",
        session_type: str = "chat",
    ) -> RAGAnswer:
        question, session_id, top_k = self._prepare_request(question, session_id, top_k)
        cached = self.cache.get(question, top_k, category)
        if cached is not None:
            self.memory.append(session_id, question, cached.answer, user_id, session_type)
            return cached

        retrieved = self.knowledge_base.search(question, top_k, category)
        messages = self._build_messages(question, session_id, retrieved, user_id, session_type)
        raw_answer = self.llm.chat(messages, max_tokens=self._max_tokens_for(session_type))
        answer = self._append_sources(raw_answer, retrieved)
        result = RAGAnswer(answer=answer, session_id=session_id, sources=retrieved, cached=False)
        self.cache.set(question, top_k, result, category)
        title = self.generate_title(question, answer)
        self.memory.append(session_id, question, answer, user_id, session_type, title)
        return result

    async def aanswer(
        self,
        question: str,
        user_id: int,
        session_id: str | None = None,
        top_k: int | None = None,
        category: str = "",
        session_type: str = "chat",
    ) -> RAGAnswer:
        return await asyncio.to_thread(self.answer, question, user_id, session_id, top_k, category, session_type)

    def stream_answer(
        self,
        question: str,
        user_id: int,
        session_id: str | None = None,
        top_k: int | None = None,
        category: str = "",
        session_type: str = "chat",
    ):
        question, session_id, top_k = self._prepare_request(question, session_id, top_k)
        yield {"type": "status", "content": "\u6b63\u5728\u68c0\u67e5\u5386\u53f2\u5bf9\u8bdd\u548c\u7f13\u5b58..."}
        cached = self.cache.get(question, top_k, category)
        if cached is not None:
            yield from self._stream_cached(session_id, question, cached, user_id, session_type)
            return

        yield {"type": "status", "content": "\u6b63\u5728\u68c0\u7d22\u77e5\u8bc6\u5e93..."}
        retrieved = self.knowledge_base.search(question, top_k, category)
        messages = self._build_messages(question, session_id, retrieved, user_id, session_type)
        yield self._meta_event(session_id, retrieved, cached=False)
        yield {"type": "status", "content": "\u5df2\u627e\u5230\u76f8\u5173\u6750\u6599\uff0c\u6b63\u5728\u751f\u6210\u56de\u7b54..."}

        parts: list[str] = []
        for delta in self.llm.stream_chat(messages, max_tokens=self._max_tokens_for(session_type)):
            delta = self._clean_system_message(delta)
            if not delta:
                continue
            parts.append(delta)
            yield {"type": "delta", "content": delta}

        yield from self._finish_stream(question, session_id, retrieved, parts, user_id, session_type, category, top_k)

    async def astream_answer(
        self,
        question: str,
        user_id: int,
        session_id: str | None = None,
        top_k: int | None = None,
        category: str = "",
        session_type: str = "chat",
    ):
        question, session_id, top_k = self._prepare_request(question, session_id, top_k)
        yield {"type": "status", "content": "\u6b63\u5728\u68c0\u67e5\u5386\u53f2\u5bf9\u8bdd\u548c\u7f13\u5b58..."}
        cached = self.cache.get(question, top_k, category)
        if cached is not None:
            for event in self._stream_cached(session_id, question, cached, user_id, session_type):
                yield event
                await asyncio.sleep(0)
            return

        yield {"type": "status", "content": "\u6b63\u5728\u68c0\u7d22\u77e5\u8bc6\u5e93..."}
        retrieved = await asyncio.to_thread(self.knowledge_base.search, question, top_k, category)
        messages = self._build_messages(question, session_id, retrieved, user_id, session_type)
        yield self._meta_event(session_id, retrieved, cached=False)
        yield {"type": "status", "content": "\u5df2\u627e\u5230\u76f8\u5173\u6750\u6599\uff0c\u6b63\u5728\u751f\u6210\u56de\u7b54..."}

        parts: list[str] = []
        async for delta in self.llm.astream_chat(messages, max_tokens=self._max_tokens_for(session_type)):
            delta = self._clean_system_message(delta)
            if not delta:
                continue
            parts.append(delta)
            yield {"type": "delta", "content": delta}

        for event in self._finish_stream(question, session_id, retrieved, parts, user_id, session_type, category, top_k):
            yield event
            await asyncio.sleep(0)

    def _prepare_request(self, question: str, session_id: str | None, top_k: int | None) -> tuple[str, str, int]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        return question, session_id or self.memory.new_session_id(), top_k or self.settings.default_top_k

    def _stream_cached(self, session_id: str, question: str, cached: RAGAnswer, user_id: int, session_type: str):
        yield {"type": "meta", "session_id": session_id, "sources": cached.source_payload(), "cached": True}
        yield {"type": "status", "content": "\u547d\u4e2d\u7f13\u5b58\uff0c\u6b63\u5728\u5feb\u901f\u8f93\u51fa..."}
        for token in self.llm._chunk_text(cached.answer):
            yield {"type": "delta", "content": token}
        self.memory.append(session_id, question, cached.answer, user_id, session_type)
        yield {"type": "done"}

    def _finish_stream(
        self,
        question: str,
        session_id: str,
        retrieved,
        parts: list[str],
        user_id: int,
        session_type: str,
        category: str,
        top_k: int,
    ):
        raw_answer = self._clean_system_message("".join(parts).strip())
        if not raw_answer:
            raw_answer = self._empty_answer_for(session_type)
        answer = self._append_sources(raw_answer, retrieved)
        appended = answer[len(raw_answer) :]
        if appended:
            yield {"type": "delta", "content": appended}
        result = RAGAnswer(answer=answer, session_id=session_id, sources=retrieved, cached=False)
        self.cache.set(question, top_k, result, category)
        title = self.generate_title(question, answer)
        self.memory.append(session_id, question, answer, user_id, session_type, title)
        yield {"type": "done", "title": title}

    def _meta_event(self, session_id: str, retrieved, cached: bool) -> dict:
        return {
            "type": "meta",
            "session_id": session_id,
            "sources": RAGAnswer("", session_id, retrieved).source_payload(),
            "cached": cached,
        }

    def _max_tokens_for(self, session_type: str) -> int:
        if session_type == "interview":
            return min(self.settings.llm_max_tokens, 520)
        if session_type == "job":
            return min(self.settings.llm_max_tokens, 620)
        return self.settings.llm_max_tokens

    def _build_messages(self, question: str, session_id: str, retrieved, user_id: int, session_type: str = "chat") -> list[dict[str, str]]:
        history = self.memory.format_context(session_id, user_id)
        context = self._format_retrieved(retrieved)
        mode_hint = {
            "interview": "\u5f53\u524d\u573a\u666f\u662f\u6a21\u62df\u9762\u8bd5\u3002\u8bf7\u4e25\u683c\u4f7f\u7528\u4e24\u4e2a\u6807\u9898\uff1a\u201c### \u9762\u8bd5\u5b98\u56de\u5e94\uff1a\u201d\u548c\u201c### \u7b80\u77ed\u5206\u6790\uff1a\u201d\u3002\u4e24\u4e2a\u6807\u9898\u90fd\u5fc5\u987b\u5355\u72ec\u6210\u884c\uff0c\u6807\u9898\u540e\u5fc5\u987b\u6362\u884c\u518d\u5199\u5177\u4f53\u5185\u5bb9\u3002\u5148\u4ee5\u771f\u5b9e\u9762\u8bd5\u5b98\u53e3\u543b\u76f4\u63a5\u56de\u5e94\u5019\u9009\u4eba\u5e76\u63d0\u51fa 1 \u4e2a\u8ffd\u95ee\uff0c\u518d\u7528 2-4 \u6761\u8981\u70b9\u7ed9\u7b80\u77ed\u5206\u6790\u3002\u4e0d\u8981\u4e00\u5f00\u59cb\u5c31\u957f\u7bc7\u8bb2\u89e3\u3002",
            "job": "\u5f53\u524d\u573a\u666f\u662f\u5c97\u4f4d\u667a\u80fd\u52a9\u624b\u3002\u8bf7\u56f4\u7ed5\u5c97\u4f4d\u65b9\u5411\u7ed9\u51fa\u5339\u914d\u5efa\u8bae\u3001\u80fd\u529b\u8981\u6c42\u548c\u5b66\u4e60\u8def\u5f84\u3002",
        }.get(session_type, "\u5f53\u524d\u573a\u666f\u662f\u77e5\u8bc6\u95ee\u7b54\u3002\u8bf7\u7ed3\u5408\u77e5\u8bc6\u5e93\u76f4\u63a5\u56de\u7b54\u7528\u6237\u95ee\u9898\u3002")
        user_prompt = f"""[\u573a\u666f]
{mode_hint}

[\u5386\u53f2\u5bf9\u8bdd]
{history}

[\u68c0\u7d22\u5230\u7684\u77e5\u8bc6]
{context}

[\u7528\u6237\u95ee\u9898]
{question}

\u8bf7\u4f7f\u7528\u4e2d\u6587\u56de\u7b54\u3002\u683c\u5f0f\u8981\u9002\u5408\u524d\u7aef\u9605\u8bfb\uff1a\u7528\u77ed\u6bb5\u843d\u3001\u6807\u9898\u548c\u5217\u8868\uff0c\u4e0d\u8981\u7528\u7eaf\u7ba1\u9053\u8868\u683c\u5806\u53e0\u5185\u5bb9\u3002"""
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]

    def _format_retrieved(self, retrieved) -> str:
        if not retrieved:
            return "(\u6ca1\u6709\u68c0\u7d22\u5230\u76f8\u5173\u77e5\u8bc6)"
        parts = []
        for i, item in enumerate(retrieved[:10], 1):
            chunk = getattr(item, "chunk", item)
            source = getattr(chunk, "source", getattr(chunk, "filename", f"Source{i}"))
            text = getattr(chunk, "content", getattr(chunk, "text", str(chunk))).strip()[:1200]
            score = getattr(item, "score", 0)
            parts.append(f"[Source {i}] file={source}; score={score:.4f}\n{text}")
        return "\n\n".join(parts)

    def _append_sources(self, answer: str, retrieved) -> str:
        if not retrieved:
            return answer
        sources = set()
        for item in retrieved[:10]:
            chunk = getattr(item, "chunk", item)
            source = getattr(chunk, "source", getattr(chunk, "filename", ""))
            if source:
                sources.add(str(source))
        if sources:
            answer = answer.rstrip()
            if answer and not answer.endswith(("\u3002", ".", "\uff01", "!", "\uff1f", "?")):
                answer += "\u3002"
            if "\u53c2\u8003\u6765\u6e90" not in answer:
                answer += "\n\n\u53c2\u8003\u6765\u6e90\uff1a" + "\u3001".join(f"[{source}]" for source in sorted(sources))
        return answer

    def _empty_answer_for(self, session_type: str) -> str:
        if session_type == "interview":
            return (
                "### 面试官回应：\n"
                "我刚才没有收到稳定的模型输出。我们先继续面试，请你再补充一下：你做过的项目里，"
                "最能体现后端能力的一项是什么？请重点说明你的职责、技术选型和结果。\n\n"
                "### 简短分析：\n"
                "- 这次更像是模型服务临时没有返回正文，不代表你的回答有问题。\n"
                "- 你可以继续按真实面试节奏回答，系统会接着追问。"
            )
        return "当前模型服务没有返回有效正文，请稍后重试，或换一种问法继续提问。"

    def _clean_system_message(self, text: str) -> str:
        if not text:
            return text
        patterns = [
            r"\[System Message\].*?\n",
            r"\[System Message\].*?$",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.S)
        return text.strip()

    def generate_title(self, question: str, answer: str = "") -> str:
        text = re.sub(r"\s+", " ", question or "").strip()
        text = re.sub(r"^(你现在是一名真实的大学生求职模拟面试官。)?面试官背景：.*?候选人刚才回答：", "", text)
        text = re.sub(r"请严格按以下顺序输出：.*$", "", text).strip()
        return (text or question.strip() or "新对话")[:15]

    def clear_session(self, session_id: str, user_id: int) -> None:
        self.memory.clear(session_id, user_id)

    def get_history(self, session_id: str, user_id: int):
        return self.memory.get(session_id, user_id)
