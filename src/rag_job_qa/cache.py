from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Optional


class AnswerCache:
    def __init__(self, max_size: int = 128):
        self.max_size = max_size
        self._items: OrderedDict[str, object] = OrderedDict()

    @staticmethod
    def key(question: str, top_k: int, category: str = "") -> str:
        normalized = " ".join(question.lower().strip().split())
        return hashlib.sha256(f"{normalized}|{top_k}|{category}".encode("utf-8")).hexdigest()

    def get(self, question: str, top_k: int, category: str = "") -> Optional[object]:
        key = self.key(question, top_k, category)
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def set(self, question: str, top_k: int, value: object, category: str = "") -> None:
        key = self.key(question, top_k, category)
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)
