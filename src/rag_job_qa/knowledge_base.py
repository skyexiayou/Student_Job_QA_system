from __future__ import annotations

import shutil
import time
import hashlib
from pathlib import Path
from typing import Iterable, List

from .config import Settings
from .models import DocumentChunk
from .neo4j_store import JOB_CATEGORIES, Neo4jKnowledgeStore, category_by_key, classify_job_category
from .text_processing import SUPPORTED_SUFFIXES, build_chunks, read_document as parse_document
from .vector_store import VectorStore


class KnowledgeBase:
    """知识库门面：负责文档导入、索引重建、检索和文档列表管理。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = VectorStore(
            settings.index_dir,
            settings.embedding_model,
            settings.use_hf_embedding,
            settings.hf_local_files_only,
        )
        self.graph_store = Neo4jKnowledgeStore(settings)

    def source_paths(self) -> List[Path]:
        """返回种子知识库和用户上传目录下所有支持的知识文档。"""
        paths: List[Path] = []
        for folder in [self.settings.seed_knowledge_dir, self.settings.documents_dir]:
            if folder.exists():
                paths.extend(path for path in folder.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)
        return sorted(paths)

    def rebuild(self) -> dict:
        """重新解析全部知识文档并重建向量索引。"""
        chunks = build_chunks(
            self.source_paths(),
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        for chunk in chunks:
            chunk.metadata["job_category"] = classify_job_category(f"{chunk.source}\n{chunk.title}\n{chunk.content}")
        self.store.rebuild(chunks)
        self.graph_store.rebuild(self.store.chunks, self.store.matrix)
        return self.stats()

    def ensure_ready(self) -> None:
        """索引不存在或需要重建时自动初始化，方便学生本地直接启动。"""
        if (self.store.is_empty() or self.store.needs_rebuild) and self.source_paths():
            self.rebuild()

    def import_files(self, files: Iterable[Path]) -> dict:
        """导入用户上传的 PDF/TXT/Markdown 文件并重建索引。"""
        self.settings.documents_dir.mkdir(parents=True, exist_ok=True)
        imported = 0
        skipped_duplicates = 0
        skipped_unsupported = 0
        existing_hashes = self._document_md5_values()
        for file in files:
            if file.suffix.lower() not in SUPPORTED_SUFFIXES:
                skipped_unsupported += 1
                continue
            file_md5 = self._file_md5(file)
            if file_md5 in existing_hashes:
                skipped_duplicates += 1
                continue
            safe_name = f"{int(time.time())}_{file.name}"
            target = self.settings.documents_dir / safe_name
            if file.resolve() != target.resolve():
                shutil.copy2(file, target)
            existing_hashes.add(file_md5)
            imported += 1
        stats = self.rebuild()
        stats.update(
            {
                "imported": imported,
                "skipped_duplicates": skipped_duplicates,
                "skipped_unsupported": skipped_unsupported,
                "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
            }
        )
        return stats

    def search(self, query: str, top_k: int, category: str = ""):
        self.ensure_ready()
        query_vector = self.store.encode_query(query)
        graph_results = self.graph_store.search(query, query_vector, top_k, category)
        if graph_results:
            return self.graph_store.enrich(graph_results)
        results = self.store.search_by_vector(query_vector, max(top_k * 4, top_k))
        if category:
            results = [item for item in results if item.chunk.metadata.get("job_category") == category]
        return results[:top_k]

    def stats(self) -> dict:
        return self.store.stats()

    def chunks(self) -> List[DocumentChunk]:
        return self.store.chunks

    def list_documents(self, keyword: str = "", page: int = 1, page_size: int = 10, category: str = "all") -> dict:
        """分页查询用户上传的知识文档，支持按文件名搜索。"""
        page = max(int(page or 1), 1)
        page_size = min(max(int(page_size or 10), 1), 50)
        keyword = (keyword or "").strip().lower()
        category = (category or "all").strip()

        files = []
        if self.settings.documents_dir.exists():
            for path in self.settings.documents_dir.glob("*"):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                if keyword and keyword not in path.name.lower():
                    continue
                doc_category = self.classify_document(path)
                if category != "all" and doc_category != category:
                    continue
                stat = path.stat()
                files.append(
                    {
                        "id": path.name,
                        "filename": path.name,
                        "category": doc_category,
                        "size": stat.st_size,
                        "created_at": stat.st_ctime,
                        "chunk_count": self._count_chunks_for_file(path.name),
                    }
                )

        files.sort(key=lambda item: item["created_at"], reverse=True)
        total = len(files)
        start = (page - 1) * page_size
        return {"documents": files[start : start + page_size], "total": total}

    def read_document(self, filename: str) -> dict:
        """读取上传文档正文，供前端文献阅读弹窗使用。"""
        safe_name = Path(filename).name
        filepath = self.settings.documents_dir / safe_name
        if not filepath.exists() or not filepath.is_file():
            raise FileNotFoundError("document not found")
        if filepath.suffix.lower() in {".pdf", ".docx", ".xlsx"}:
            text = "\n\n".join(chunk.content for chunk in self.store.chunks if safe_name in chunk.source)
        else:
            text = parse_document(filepath)
        return {
            "id": safe_name,
            "filename": safe_name,
            "category": self.classify_document(filepath),
            "content": text[:50000],
        }

    def categories(self) -> list[dict]:
        """返回前端筛选栏使用的分类。分类是轻量规则推断，适合作为实习项目演示。"""
        counts = {item["key"]: 0 for item in JOB_CATEGORIES}
        for path in self.settings.documents_dir.glob("*") if self.settings.documents_dir.exists() else []:
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                counts[self.classify_document(path)] += 1
        categories = [{"key": "all", "label": "全部", "count": sum(counts.values())}]
        categories.extend(
            {
                "key": item["key"],
                "label": item["name"],
                "count": counts[item["key"]],
                "work": item["work"],
                "skills": item["skills"],
            }
            for item in JOB_CATEGORIES
        )
        return categories

    def classify_document(self, path: Path) -> str:
        """根据文件名和少量正文关键词推断岗位分类。"""
        sample = path.name.lower()
        if path.suffix.lower() != ".pdf":
            try:
                sample += "\n" + path.read_text(encoding="utf-8", errors="ignore")[:1500].lower()
            except Exception:
                pass
        return classify_job_category(sample)

    def graph_data(self, category: str = "", limit: int = 240) -> dict:
        data = self.graph_store.graph_data("" if category == "all" else category, limit)
        if data.get("nodes") or data.get("neo4j_enabled"):
            return data
        nodes = [
            {
                "id": f"category:{item['key']}",
                "label": item["name"],
                "type": "JobCategory",
                "category": item["key"],
                "properties": item,
            }
            for item in JOB_CATEGORIES
            if not category or category == "all" or item["key"] == category
        ]
        return {"nodes": nodes, "edges": [], "neo4j_enabled": False}

    def node_detail(self, node_id: str, relation_type: str = "") -> dict:
        if node_id.startswith("category:"):
            item = category_by_key(node_id.split(":", 1)[1])
            return {
                "node": {"id": node_id, "label": item["name"], "type": "JobCategory", "properties": item},
                "relations": [],
                "neo4j_enabled": self.graph_store.enabled,
            }
        return self.graph_store.node_detail(node_id, relation_type)

    def graph_categories(self) -> list[dict]:
        stats = self.graph_store.category_stats()
        if stats and any(item.get("entity_count") or item.get("document_count") for item in stats):
            by_key = {item["key"]: item for item in stats}
            return [by_key.get(item["key"], dict(item, entity_count=0, document_count=0)) for item in JOB_CATEGORIES]
        doc_counts = {item["key"]: 0 for item in JOB_CATEGORIES}
        for path in self.settings.documents_dir.glob("*") if self.settings.documents_dir.exists() else []:
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                doc_counts[self.classify_document(path)] += 1
        return [dict(item, entity_count=doc_counts[item["key"]], document_count=doc_counts[item["key"]]) for item in JOB_CATEGORIES]

    def search_graph_nodes(self, keyword: str, category: str = "", limit: int = 30) -> list[dict]:
        keyword = (keyword or "").strip()
        if not keyword:
            return []
        rows = self.graph_store.search_nodes(keyword, "" if category == "all" else category, limit)
        if rows:
            return rows
        return [
            {
                "id": f"category:{item['key']}",
                "label": item["name"],
                "type": "JobCategory",
                "properties": item,
            }
            for item in JOB_CATEGORIES
            if keyword.lower() in item["name"].lower() and (not category or category == "all" or item["key"] == category)
        ][:limit]

    def delete_document(self, filename: str) -> None:
        """删除单个上传文档并重建索引。"""
        safe_name = Path(filename).name
        filepath = self.settings.documents_dir / safe_name
        if filepath.exists() and filepath.is_file():
            filepath.unlink()
            self.rebuild()

    def _count_chunks_for_file(self, filename: str) -> int:
        return sum(1 for chunk in self.store.chunks if filename in chunk.source)

    @staticmethod
    def _file_md5(path: Path) -> str:
        digest = hashlib.md5()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _document_md5_values(self) -> set[str]:
        values = set()
        if not self.settings.documents_dir.exists():
            return values
        for path in self.settings.documents_dir.glob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                values.add(self._file_md5(path))
        return values
