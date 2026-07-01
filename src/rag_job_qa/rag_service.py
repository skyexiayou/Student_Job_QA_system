from __future__ import annotations

from .cache import AnswerCache
from .config import Settings
from .conversation import ConversationMemory
from .knowledge_base import KnowledgeBase
from .llm_client import LLMClient
from .models import RAGAnswer


SYSTEM_PROMPT = """你是大学生求职岗位知识问答助手。
请严格基于检索到的知识片段回答问题；如果知识片段不足，请明确说明“不确定”，并给出可验证、可执行的求职建议。

回答要求：
1. 面向大学生，表达清晰、实用、鼓励，但不要夸大。
2. 不编造不存在的岗位要求、薪资数据、公司政策或评价结果。
3. 示例要写成“可参考写法”，不要把示例当成真实项目成果。
4. 如果用户询问简历、面试、项目答辩，优先给出结构化步骤。
5. 结尾列出“参考来源”，标注使用到的文档名称。"""


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.knowledge_base = KnowledgeBase(settings)
        self.memory = ConversationMemory(settings.conversations_dir, settings.memory_rounds)
        self.llm = LLMClient(settings)
        self.cache = AnswerCache(settings.cache_size)
        self.knowledge_base.ensure_ready()

    def answer(self, question: str, session_id: str | None = None, top_k: int | None = None, category: str = "") -> RAGAnswer:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        session_id = session_id or self.memory.new_session_id()
        top_k = top_k or self.settings.default_top_k

        cached = self.cache.get(question, top_k, category)
        if cached is not None:
            self.memory.append(session_id, question, cached.answer)
            return RAGAnswer(answer=cached.answer, session_id=session_id, sources=cached.sources, cached=True)

        retrieved = self.knowledge_base.search(question, top_k, category)
        messages = self._build_messages(question, session_id, retrieved)
        answer = self.llm.chat(messages=messages)
        answer = self._append_sources(answer, retrieved)
        result = RAGAnswer(answer=answer, session_id=session_id, sources=retrieved, cached=False)
        self.cache.set(question, top_k, result, category)
        self.memory.append(session_id, question, answer)
        return result

    def stream_answer(self, question: str, session_id: str | None = None, top_k: int | None = None, category: str = ""):
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        session_id = session_id or self.memory.new_session_id()
        top_k = top_k or self.settings.default_top_k

        cached = self.cache.get(question, top_k, category)
        if cached is not None:
            yield {"type": "meta", "session_id": session_id, "sources": cached.source_payload(), "cached": True}
            for token in self.llm._chunk_text(cached.answer):
                yield {"type": "delta", "content": token}
            self.memory.append(session_id, question, cached.answer)
            yield {"type": "done"}
            return

        retrieved = self.knowledge_base.search(question, top_k, category)
        messages = self._build_messages(question, session_id, retrieved)
        yield {
            "type": "meta",
            "session_id": session_id,
            "sources": RAGAnswer("", session_id, retrieved).source_payload(),
            "cached": False,
        }

        parts = []
        for delta in self.llm.stream_chat(messages):
            parts.append(delta)
            yield {"type": "delta", "content": delta}

        raw_answer = "".join(parts).strip()
        answer = self._append_sources(raw_answer, retrieved)
        appended = answer[len(raw_answer) :]
        if appended:
            yield {"type": "delta", "content": appended}
        result = RAGAnswer(answer=answer, session_id=session_id, sources=retrieved, cached=False)
        self.cache.set(question, top_k, result, category)
        self.memory.append(session_id, question, answer)
        yield {"type": "done"}

    def _build_messages(self, question: str, session_id: str, retrieved) -> list[dict[str, str]]:
        history = self.memory.format_context(session_id)
        context = self._format_retrieved(retrieved)
        user_prompt = f"""【历史对话】
{history}

【检索到的知识片段】
{context}

【用户当前问题】
{question}

请基于以上知识片段回答。"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _format_retrieved(retrieved) -> str:
        if not retrieved:
            return "未召回到相关知识片段。"
        lines = []
        for index, item in enumerate(retrieved, start=1):
            lines.append(
                f"[{index}] 来源：{item.chunk.source}；相似度：{item.score:.3f}\n{item.chunk.content}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _append_sources(answer: str, retrieved) -> str:
        if "参考来源" in answer:
            return answer
        if not retrieved:
            return answer + "\n\n参考来源：知识库未召回到明确片段。"
        names = []
        for item in retrieved:
            if item.chunk.source not in names:
                names.append(item.chunk.source)
        return f"{answer}\n\n参考来源：{'、'.join(names)}"

    def new_session(self) -> str:
        return self.memory.new_session_id()

    def clear_session(self, session_id: str) -> None:
        self.memory.clear(session_id)
