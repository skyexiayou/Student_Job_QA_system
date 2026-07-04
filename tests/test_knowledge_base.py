from pathlib import Path

from src.rag_job_qa.config import Settings
from src.rag_job_qa.knowledge_base import KnowledgeBase
from src.rag_job_qa.models import DocumentChunk


def test_default_upload_dir_is_under_knowledge_base():
    settings = Settings()
    assert settings.documents_dir == settings.seed_knowledge_dir / "uploads"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        documents_dir=tmp_path / "documents",
        seed_knowledge_dir=tmp_path / "knowledge_base",
        index_dir=tmp_path / "index",
        conversations_dir=tmp_path / "conversations",
        model_cache_dir=tmp_path / "models",
        use_hf_embedding=False,
    )


def test_import_files_parses_classifies_and_indexes_uploaded_docs(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    source = tmp_path / "测试工程师岗位说明.txt"
    source.write_text("测试工程师 需要 自动化测试 pytest selenium 测试用例 接口测试", encoding="utf-8")

    kb = KnowledgeBase(settings)
    stats = kb.import_files([source])

    assert stats["imported"] == 1
    assert stats["skipped_parse_failed"] == 0
    assert stats["document_count"] == 1
    assert stats["chunk_count"] >= 1
    assert stats["imported_documents"][0]["category"] == "testing"

    listed = kb.list_documents(category="testing")
    assert listed["total"] == 1
    assert listed["documents"][0]["category"] == "testing"
    assert listed["documents"][0]["chunk_count"] >= 1


def test_import_files_classifies_real_chinese_upload_and_filters_rag_results(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    backend_source = tmp_path / "后端开发岗位说明.txt"
    backend_source.write_text(
        "后端开发工程师需要熟悉 Java、Spring Boot、MySQL、Redis、接口设计、微服务和高并发系统。",
        encoding="utf-8",
    )
    embedded_source = tmp_path / "嵌入式底层开发资料.txt"
    embedded_source.write_text(
        "嵌入式工程师需要掌握 C 语言、STM32、ARM、Linux 驱动、单片机和硬件通信协议。",
        encoding="utf-8",
    )

    kb = KnowledgeBase(settings)
    stats = kb.import_files([backend_source, embedded_source])

    imported_categories = {item["category"] for item in stats["imported_documents"]}
    assert {"backend", "embedded"} <= imported_categories
    assert stats["chunk_count"] >= 2

    backend_docs = kb.list_documents(category="backend")
    embedded_docs = kb.list_documents(category="embedded")
    assert backend_docs["total"] == 1
    assert embedded_docs["total"] == 1

    backend_results = kb.search("Spring Boot 接口 高并发", top_k=3, category="backend")
    assert backend_results
    assert all(item.chunk.metadata["job_category"] == "backend" for item in backend_results)


def test_document_ids_work_for_nested_upload_directory(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    nested = settings.documents_dir / "2026-07-01"
    nested.mkdir(parents=True)
    doc = nested / "后端开发资料.txt"
    doc.write_text("后端开发 Java Spring MySQL Redis 接口 高并发", encoding="utf-8")

    kb = KnowledgeBase(settings)
    kb.rebuild()

    listed = kb.list_documents(category="backend")
    assert listed["total"] == 1
    doc_id = listed["documents"][0]["id"]
    assert doc_id == "2026-07-01/后端开发资料.txt"
    assert kb.document_file_path(doc_id) == doc


def test_delete_document_removes_nested_upload_and_rebuilds_index(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    nested = settings.documents_dir / "2026-07-01"
    nested.mkdir(parents=True)
    doc = nested / "backend.txt"
    doc.write_text("backend Java Spring MySQL", encoding="utf-8")

    kb = KnowledgeBase(settings)
    kb.rebuild()
    assert kb.list_documents()["total"] == 1

    kb.delete_document("2026-07-01/backend.txt")

    assert not doc.exists()
    assert kb.list_documents()["total"] == 0
    assert kb.stats()["document_count"] == 0
    assert kb.stats()["chunk_count"] == 0


def test_delete_documents_deduplicates_and_reports_missing(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    doc = settings.documents_dir / "backend.txt"
    doc.write_text("backend Java Spring MySQL", encoding="utf-8")

    kb = KnowledgeBase(settings)
    kb.rebuild()
    result = kb.delete_documents(["backend.txt", "backend.txt", "missing.txt"])

    assert result == {"deleted": ["backend.txt"], "missing": ["missing.txt"], "deleted_count": 1}
    assert not doc.exists()
    assert kb.list_documents()["total"] == 0


def test_delete_document_rejects_path_traversal_outside_document_roots(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    outside = tmp_path / "outside.txt"
    outside.write_text("must stay", encoding="utf-8")

    kb = KnowledgeBase(settings)
    kb.delete_document("../outside.txt")

    assert outside.exists()
    assert outside.read_text(encoding="utf-8") == "must stay"


def test_graph_data_clamps_display_node_limit(tmp_path):
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    kb = KnowledgeBase(settings)
    chunks = [
        DocumentChunk(
            chunk_id=f"chunk-{index}",
            source=f"doc-{index}.txt",
            title=f"doc-{index}",
            content="backend Java Spring MySQL",
            metadata={"job_category": "backend"},
        )
        for index in range(220)
    ]
    kb.store.rebuild(chunks)

    data = kb.graph_data(limit=999)

    assert len(data["nodes"]) <= 240
    assert all(edge["source"] in {node["id"] for node in data["nodes"]} for edge in data["edges"])
    assert all(edge["target"] in {node["id"] for node in data["nodes"]} for edge in data["edges"])
