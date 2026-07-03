from __future__ import annotations

import shutil
import time
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from .config import Settings
from .models import DocumentChunk
from .neo4j_store import JOB_CATEGORIES, Neo4jKnowledgeStore, category_by_key, classify_job_category
from .text_processing import SUPPORTED_SUFFIXES, build_chunks, read_document as parse_document
from .vector_store import VectorStore


logger = logging.getLogger(__name__)


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
        upload_roots = self._upload_roots()
        if self.settings.seed_knowledge_dir.exists():
            for path in self.settings.seed_knowledge_dir.rglob("*"):
                if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                if any(self._is_inside(path, root) for root in upload_roots):
                    continue
                paths.append(path)
        for root in upload_roots:
            if root.exists():
                paths.extend(path for path in root.rglob("*") if path.suffix.lower() in SUPPORTED_SUFFIXES)
        return sorted(paths)

    def rebuild(self) -> dict:
        """重新解析全部知识文档并重建向量索引。"""
        started = time.perf_counter()
        chunks = build_chunks(
            self.source_paths(),
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        for chunk in chunks:
            chunk.metadata["job_category"] = classify_job_category(f"{chunk.source}\n{chunk.title}\n{chunk.content}")
        self.store.rebuild(chunks)
        self.graph_store.rebuild(self.store.chunks, self.store.matrix)
        stats = self.stats()
        stats["rebuild_seconds"] = round(time.perf_counter() - started, 3)
        stats["neo4j"] = self.graph_store.diagnostics()
        logger.info("Knowledge base rebuilt: %s", stats)
        return stats

    def ensure_ready(self) -> None:
        """索引不存在或需要重建时自动初始化，方便学生本地直接启动。"""
        if (self.store.is_empty() or self.store.needs_rebuild) and self.source_paths():
            self.rebuild()

    def import_files(self, files: Iterable[Path]) -> dict:
        """导入用户上传文件，完成解析、分类、向量化并重建索引。"""
        upload_dir = self.settings.documents_dir / datetime.now().strftime("%Y-%m-%d")
        upload_dir.mkdir(parents=True, exist_ok=True)
        imported = 0
        skipped_duplicates = 0
        skipped_unsupported = 0
        skipped_parse_failed = 0
        imported_documents = []
        failed_documents = []
        existing_hashes = self._document_md5_values()
        for file in files:
            if file.suffix.lower() not in SUPPORTED_SUFFIXES:
                skipped_unsupported += 1
                failed_documents.append({"filename": file.name, "reason": "unsupported file type"})
                continue
            try:
                parsed_text = parse_document(file)
            except Exception as exc:
                skipped_parse_failed += 1
                failed_documents.append({"filename": file.name, "reason": f"parse failed: {exc}"})
                logger.exception("Failed to parse uploaded file: %s", file)
                continue
            if not parsed_text.strip():
                skipped_parse_failed += 1
                failed_documents.append({"filename": file.name, "reason": "empty parsed text"})
                continue
            file_md5 = self._file_md5(file)
            if file_md5 in existing_hashes:
                skipped_duplicates += 1
                failed_documents.append({"filename": file.name, "reason": "duplicate md5"})
                continue
            safe_name = f"{int(time.time())}_{file_md5[:8]}_{file.name}"
            target = upload_dir / safe_name
            if file.resolve() != target.resolve():
                shutil.copy2(file, target)
            existing_hashes.add(file_md5)
            imported += 1
            imported_documents.append(
                {
                    "filename": target.name,
                    "path": self._document_id(target),
                    "category": classify_job_category(f"{target.name}\n{parsed_text[:4000]}"),
                    "text_length": len(parsed_text),
                    "md5": file_md5,
                }
            )
        stats = self.rebuild()
        stats.update(
            {
                "imported": imported,
                "skipped_duplicates": skipped_duplicates,
                "skipped_unsupported": skipped_unsupported,
                "skipped_parse_failed": skipped_parse_failed,
                "imported_documents": imported_documents,
                "failed_documents": failed_documents,
                "supported_suffixes": sorted(SUPPORTED_SUFFIXES),
                "upload_dir": str(upload_dir),
            }
        )
        return stats

    def search(self, query: str, top_k: int, category: str = ""):
        self.ensure_ready()
        query_vector = self.store.encode_query(query)
        graph_results = self.graph_store.search(query, query_vector, top_k, category)
        if graph_results:
            return self._filter_by_threshold(self.graph_store.enrich(graph_results))
        results = self.store.search_by_vector(query_vector, max(top_k * 4, top_k))
        if category:
            results = [item for item in results if item.chunk.metadata.get("job_category") == category]
        for item in results:
            item.chunk.metadata["retrieval_mode"] = "local_vector_fallback"
        return self._filter_by_threshold(results)[:top_k]

    def stats(self) -> dict:
        stats = self.store.stats()
        stats.update(
            {
                "documents_dir": str(self.settings.documents_dir),
                "seed_knowledge_dir": str(self.settings.seed_knowledge_dir),
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "default_top_k": self.settings.default_top_k,
                "similarity_threshold": self.settings.similarity_threshold,
                "graph_hops": self.settings.graph_hops,
                "neo4j_enabled": self.graph_store.enabled,
                "neo4j_last_error": self.graph_store.last_error,
            }
        )
        return stats

    def chunks(self) -> List[DocumentChunk]:
        return self.store.chunks

    def list_documents(self, keyword: str = "", page: int = 1, page_size: int = 10, category: str = "all") -> dict:
        """分页查询用户上传的知识文档，支持按文件名搜索。"""
        page = max(int(page or 1), 1)
        page_size = min(max(int(page_size or 10), 1), 50)
        keyword = (keyword or "").strip().lower()
        category = (category or "all").strip()

        files = []
        upload_roots = self._upload_roots()
        for root in [*upload_roots, self.settings.seed_knowledge_dir]:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                if root == self.settings.seed_knowledge_dir and any(self._is_inside(path, upload_root) for upload_root in upload_roots):
                    continue
                if root == self.settings.seed_knowledge_dir and path.parent == self.settings.seed_knowledge_dir:
                    continue
                if keyword and keyword not in path.name.lower():
                    continue
                doc_category = self.classify_document(path)
                if category != "all" and doc_category != category:
                    continue
                stat = path.stat()
                files.append(
                    {
                        "id": self._document_id(path),
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
        filepath = self._find_document(filename)
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
            "suffix": filepath.suffix.lower(),
            "file_url": f"/api/knowledge/document-file/{self._document_id(filepath)}",
        }

    def categories(self) -> list[dict]:
        """返回前端筛选栏使用的分类。分类是轻量规则推断，适合作为实习项目演示。"""
        counts = {item["key"]: 0 for item in JOB_CATEGORIES}
        for path in self._iter_user_documents():
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
        chunks = [chunk for chunk in self.store.chunks if chunk.source == path.name]
        if chunks:
            category_counts: dict[str, int] = {}
            for chunk in chunks:
                key = chunk.metadata.get("job_category") or classify_job_category(f"{chunk.source}\n{chunk.title}\n{chunk.content}")
                category_counts[key] = category_counts.get(key, 0) + 1
            return max(category_counts.items(), key=lambda item: item[1])[0]

        sample = path.name.lower()
        try:
            sample += "\n" + parse_document(path)[:4000].lower()
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
        edges = []
        seen_sources = set()
        for chunk in self.store.chunks:
            chunk_category = chunk.metadata.get("job_category") or classify_job_category(f"{chunk.source}\n{chunk.content}")
            if category and category != "all" and chunk_category != category:
                continue
            doc_id = f"document:{chunk.source}"
            if doc_id in seen_sources:
                continue
            seen_sources.add(doc_id)
            nodes.append(
                {
                    "id": doc_id,
                    "label": chunk.source,
                    "type": "Document",
                    "category": chunk_category,
                    "properties": {
                        "name": chunk.source,
                        "title": chunk.title,
                        "source": chunk.source,
                        "category": chunk_category,
                        "content": chunk.content[:600],
                    },
                }
            )
            edges.append({"id": f"{chunk_category}:{chunk.source}", "source": f"category:{chunk_category}", "target": doc_id, "type": "RELATED_TO"})
        return {"nodes": nodes[:limit], "edges": edges[:limit], "neo4j_enabled": False}

    def node_detail(self, node_id: str, relation_type: str = "") -> dict:
        if node_id.startswith("category:"):
            item = category_by_key(node_id.split(":", 1)[1])
            return {
                "node": {"id": node_id, "label": item["name"], "type": "JobCategory", "properties": item},
                "relations": [],
                "neo4j_enabled": self.graph_store.enabled,
            }
        if node_id.startswith("document:"):
            source = node_id.split(":", 1)[1]
            chunks = [chunk for chunk in self.store.chunks if chunk.source == source]
            if chunks:
                first = chunks[0]
                category = first.metadata.get("job_category", "")
                return {
                    "node": {
                        "id": node_id,
                        "label": source,
                        "type": "Document",
                        "properties": {
                            "source": source,
                            "title": first.title,
                            "category": category,
                            "content": "\n\n".join(chunk.content for chunk in chunks)[:2000],
                        },
                    },
                    "relations": [
                        {
                            "id": f"{node_id}:category",
                            "type": "RELATED_TO",
                            "direction": "out",
                            "node": {
                                "id": f"category:{category}",
                                "label": category_by_key(category)["name"],
                                "type": "JobCategory",
                                "properties": category_by_key(category),
                            },
                        }
                    ],
                    "neo4j_enabled": self.graph_store.enabled,
                }
        return self.graph_store.node_detail(node_id, relation_type)

    def graph_categories(self) -> list[dict]:
        stats = self.graph_store.category_stats()
        if stats and any(item.get("entity_count") or item.get("document_count") for item in stats):
            by_key = {item["key"]: item for item in stats}
            return [by_key.get(item["key"], dict(item, entity_count=0, document_count=0)) for item in JOB_CATEGORIES]
        doc_counts = {item["key"]: 0 for item in JOB_CATEGORIES}
        for root in self._upload_roots():
            for path in root.rglob("*") if root.exists() else []:
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                    doc_counts[self.classify_document(path)] += 1
        return [dict(item, entity_count=doc_counts[item["key"]], document_count=doc_counts[item["key"]]) for item in JOB_CATEGORIES]

    def graph_diagnostics(self) -> dict:
        diagnostics = self.graph_store.diagnostics()
        diagnostics["fallback_graph_available"] = bool(self.store.chunks or self.source_paths())
        diagnostics["vector_chunks"] = len(self.store.chunks)
        return diagnostics

    def search_graph_nodes(self, keyword: str, category: str = "", limit: int = 30) -> list[dict]:
        keyword = (keyword or "").strip()
        if not keyword:
            return []
        rows = self.graph_store.search_nodes(keyword, "" if category == "all" else category, limit)
        if rows:
            return rows
        lowered = keyword.lower()
        doc_rows = []
        seen_sources = set()
        for chunk in self.store.chunks:
            chunk_category = chunk.metadata.get("job_category") or classify_job_category(f"{chunk.source}\n{chunk.content}")
            if category and category != "all" and chunk_category != category:
                continue
            if lowered not in chunk.source.lower() and lowered not in chunk.title.lower() and lowered not in chunk.content.lower():
                continue
            if chunk.source in seen_sources:
                continue
            seen_sources.add(chunk.source)
            doc_rows.append(
                {
                    "id": f"document:{chunk.source}",
                    "label": chunk.source,
                    "type": "Document",
                    "properties": {
                        "source": chunk.source,
                        "title": chunk.title,
                        "category": chunk_category,
                        "content": chunk.content[:600],
                    },
                }
            )
        category_rows = [
            {
                "id": f"category:{item['key']}",
                "label": item["name"],
                "type": "JobCategory",
                "properties": item,
            }
            for item in JOB_CATEGORIES
            if keyword.lower() in item["name"].lower() and (not category or category == "all" or item["key"] == category)
        ]
        return (doc_rows + category_rows)[:limit]

    def delete_document(self, filename: str) -> None:
        """删除单个上传文档并重建索引。"""
        filepath = self._find_document(filename)
        if filepath.exists() and filepath.is_file():
            filepath.unlink()
            self.rebuild()

    def document_file_path(self, filename: str) -> Path:
        return self._find_document(filename)

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
        for path in self._iter_user_documents():
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                values.add(self._file_md5(path))
        return values

    def _iter_user_documents(self):
        upload_roots = self._upload_roots()
        roots = [*upload_roots, self.settings.seed_knowledge_dir]
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                    if root == self.settings.seed_knowledge_dir and any(self._is_inside(path, upload_root) for upload_root in upload_roots):
                        continue
                    if root == self.settings.seed_knowledge_dir and path.parent == self.settings.seed_knowledge_dir:
                        continue
                    yield path

    def _filter_by_threshold(self, results):
        threshold = float(self.settings.similarity_threshold or 0)
        if threshold <= 0:
            return results
        return [item for item in results if float(item.score) >= threshold]

    @staticmethod
    def _is_inside(path: Path, folder: Path) -> bool:
        try:
            path.resolve().relative_to(folder.resolve())
            return True
        except ValueError:
            return False

    def _document_id(self, path: Path) -> str:
        for root in self._upload_roots():
            try:
                return str(path.relative_to(root)).replace("\\", "/")
            except ValueError:
                pass
        try:
            return str(path.relative_to(self.settings.seed_knowledge_dir)).replace("\\", "/")
        except ValueError:
            return path.name

    def _find_document(self, filename: str) -> Path:
        normalized = str(filename).replace("\\", "/").strip("/")
        for root in self._upload_roots():
            direct_nested = root / normalized
            if direct_nested.exists():
                return direct_nested
        direct_seed = self.settings.seed_knowledge_dir / normalized
        if direct_seed.exists():
            return direct_seed
        for root in self._upload_roots():
            direct_docs = root / Path(normalized).name
            if direct_docs.exists():
                return direct_docs
        safe_name = Path(normalized).name
        for path in self._iter_user_documents():
            if path.name == safe_name or self._document_id(path) == normalized:
                return path
        return self.settings.documents_dir / safe_name

    def _upload_roots(self) -> list[Path]:
        roots = [self.settings.documents_dir]
        legacy_root = getattr(self.settings, "legacy_documents_dir", None)
        if legacy_root and legacy_root != self.settings.documents_dir and self._is_inside(legacy_root, self.settings.data_dir):
            roots.append(legacy_root)
        unique_roots = []
        seen = set()
        for root in roots:
            key = str(root.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique_roots.append(root)
        return unique_roots
