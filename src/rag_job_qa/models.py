from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    title: str
    content: str
    metadata: Dict[str, str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DocumentChunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            source=str(data["source"]),
            title=str(data["title"]),
            content=str(data["content"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


@dataclass
class RAGAnswer:
    answer: str
    session_id: str
    sources: List[RetrievedChunk]
    cached: bool = False

    def source_payload(self) -> List[Dict[str, object]]:
        return [
            {
                "chunk_id": item.chunk.chunk_id,
                "source": item.chunk.source,
                "title": item.chunk.title,
                "score": round(item.score, 4),
                "content": item.chunk.content[:360],
                "category": item.chunk.metadata.get("job_category", ""),
                "retrieval_mode": item.chunk.metadata.get("retrieval_mode", ""),
                "graph_entities": item.chunk.metadata.get("graph_entities", ""),
                "path": item.chunk.metadata.get("path", ""),
            }
            for item in self.sources
        ]
