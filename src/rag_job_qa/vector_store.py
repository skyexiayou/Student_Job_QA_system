from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from .models import DocumentChunk, RetrievedChunk


class EmbeddingBackend:
    """Hugging Face sentence embedding with a no-network classroom fallback."""

    def __init__(self, model_name: str, use_hf_embedding: bool = False, local_files_only: bool = True):
        self.model_name = model_name
        self.backend_name = "hashing"
        self.model = None
        self.status_message = "使用本地轻量检索"
        if not use_hf_embedding:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
            self.backend_name = "sentence-transformers"
            self.status_message = "使用 Hugging Face Embedding"
        except Exception as exc:
            self.model = None
            self.status_message = f"Hugging Face 模型未就绪，已自动使用本地轻量检索：{exc.__class__.__name__}"

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype="float32")
        if self.model is not None:
            vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype="float32")
        return self._hashing_encode(texts)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
        bigrams = [text[i : i + 2] for i in range(max(0, len(text) - 1)) if text[i : i + 2].strip()]
        return words + bigrams[:800]

    def _hashing_encode(self, texts: List[str], dim: int = 384) -> np.ndarray:
        matrix = np.zeros((len(texts), dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in self._tokenize(text):
                index = hash(token) % dim
                matrix[row, index] += 1.0
            norm = math.sqrt(float(np.dot(matrix[row], matrix[row])))
            if norm > 0:
                matrix[row] /= norm
        return matrix


class VectorStore:
    """Persistent vector index. FAISS is used when installed; numpy search is fallback."""

    def __init__(
        self,
        index_dir: Path,
        embedding_model: str,
        use_hf_embedding: bool = False,
        hf_local_files_only: bool = True,
    ):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.index_dir / "chunks.json"
        self.matrix_path = self.index_dir / "vectors.npy"
        self.faiss_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "metadata.json"
        self.embedding = EmbeddingBackend(embedding_model, use_hf_embedding, hf_local_files_only)
        self.chunks: List[DocumentChunk] = []
        self.matrix: Optional[np.ndarray] = None
        self.faiss_index = None
        self.needs_rebuild = False
        self._load()

    def _try_import_faiss(self):
        try:
            import faiss

            return faiss
        except Exception:
            return None

    def _load(self) -> None:
        if self.meta_path.exists():
            metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            old_backend = metadata.get("embedding_backend")
            old_model = metadata.get("embedding_model")
            if old_backend != self.embedding.backend_name:
                self.needs_rebuild = True
                return
            if self.embedding.backend_name == "sentence-transformers" and old_model != self.embedding.model_name:
                self.needs_rebuild = True
                return
        if self.chunks_path.exists():
            data = json.loads(self.chunks_path.read_text(encoding="utf-8"))
            self.chunks = [DocumentChunk.from_dict(item) for item in data]
        if self.matrix_path.exists():
            self.matrix = np.load(self.matrix_path)
        faiss = self._try_import_faiss()
        if faiss and self.faiss_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(self.faiss_path))
            except Exception:
                self.faiss_index = None

    def save(self) -> None:
        self.chunks_path.write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.matrix is not None:
            np.save(self.matrix_path, self.matrix)
        metadata = {
            "chunk_count": len(self.chunks),
            "embedding_backend": self.embedding.backend_name,
            "embedding_model": self.embedding.model_name,
        }
        self.meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def rebuild(self, chunks: Iterable[DocumentChunk]) -> None:
        self.chunks = list(chunks)
        texts = [chunk.content for chunk in self.chunks]
        self.matrix = self.embedding.encode(texts)
        faiss = self._try_import_faiss()
        self.faiss_index = None
        if faiss and self.matrix is not None and len(self.matrix) > 0:
            index = faiss.IndexFlatIP(self.matrix.shape[1])
            index.add(self.matrix)
            try:
                faiss.write_index(index, str(self.faiss_path))
                self.faiss_index = index
            except Exception:
                self.faiss_index = None
        self.save()

    def is_empty(self) -> bool:
        return not self.chunks or self.matrix is None or len(self.chunks) == 0

    def search(self, query: str, top_k: int = 4) -> List[RetrievedChunk]:
        if self.is_empty():
            return []
        query_vector = self.embedding.encode([query])
        return self.search_by_vector(query_vector[0], top_k)

    def encode_query(self, query: str) -> np.ndarray:
        return self.embedding.encode([query])[0]

    def search_by_vector(self, query_vector: np.ndarray, top_k: int = 4) -> List[RetrievedChunk]:
        if self.is_empty():
            return []
        top_k = max(1, min(top_k, len(self.chunks)))
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(np.asarray([query_vector], dtype="float32"), top_k)
            return [
                RetrievedChunk(chunk=self.chunks[int(index)], score=float(score))
                for score, index in zip(scores[0], indices[0])
                if int(index) >= 0
            ]
        scores = np.dot(self.matrix, query_vector)
        indices = np.argsort(scores)[::-1][:top_k]
        return [RetrievedChunk(chunk=self.chunks[int(index)], score=float(scores[int(index)])) for index in indices]

    def stats(self) -> dict:
        sources = sorted({chunk.source for chunk in self.chunks})
        return {
            "document_count": len(sources),
            "chunk_count": len(self.chunks),
            "sources": sources,
            "embedding_backend": self.embedding.backend_name,
            "index_type": "FAISS" if self.faiss_index is not None else "numpy-cosine",
            "embedding_status": self.embedding.status_message,
        }
