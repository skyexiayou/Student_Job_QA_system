from __future__ import annotations

import re
import tempfile
import threading
from pathlib import Path

from .config import Settings


class SpeechService:
    """Lazy FunASR wrapper for browser-recorded audio."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._model = None

    def recognize_bytes(self, content: bytes, suffix: str = ".webm") -> dict:
        if not content:
            raise ValueError("音频内容为空")
        suffix = suffix if suffix.startswith(".") else f".{suffix}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(content)
            audio_path = Path(temp.name)
        try:
            text = self._recognize_file(audio_path)
            return {"text": self._punctuate(text), "raw_text": text}
        finally:
            try:
                audio_path.unlink()
            except OSError:
                pass

    def recognize_stream_bytes(self, content: bytes, suffix: str = ".webm") -> dict:
        """Recognize currently buffered websocket audio.

        MediaRecorder emits container fragments. Concatenating the fragments is
        good enough for short browser-recorded WebM sessions and keeps the API
        independent from a specific FunASR streaming runtime.
        """
        return self.recognize_bytes(content, suffix)

    def _recognize_file(self, audio_path: Path) -> str:
        model = self._get_model()
        result = model.generate(input=str(audio_path), language="zh", use_itn=True)
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return str(first.get("text") or "").strip()
            return str(first).strip()
        if isinstance(result, dict):
            return str(result.get("text") or "").strip()
        return str(result or "").strip()

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from funasr import AutoModel
            except Exception as exc:
                raise RuntimeError("未安装 FunASR，请先执行 pip install funasr modelscope") from exc
            self._model = AutoModel(
                model=self.settings.funasr_model,
                trust_remote_code=True,
                disable_update=True,
            )
            return self._model

    @staticmethod
    def _punctuate(text: str) -> str:
        text = re.sub(r"\s+", "", text or "").strip("，。？！； ")
        if not text:
            return ""
        text = text.replace("？", "?").replace("！", "!")
        text = re.sub(r"(吗|呢|么|什么|为什么|怎么|如何)$", r"\1?", text)
        if text.endswith("?"):
            return text[:-1] + "？"
        if text.endswith("!"):
            return text[:-1] + "！"
        if text[-1] not in "。！？；":
            text += "。"
        return text
