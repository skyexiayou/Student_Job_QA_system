from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .models import DocumentChunk, RetrievedChunk


ENTITY_LABELS = {
    "position": "Position",
    "company": "Company",
    "skill": "Skill",
    "salary": "Salary",
    "city": "City",
    "education": "Education",
}


JOB_CATEGORIES = [
    {"key": "backend", "name": "后端开发", "work": "搭建服务接口、业务系统、分布式高并发后台", "skills": "Java/Go/C++、数据库、中间件、网络、操作系统"},
    {"key": "frontend", "name": "前端开发", "work": "网页、小程序、跨端页面、数据可视化交互开发", "skills": "Vue/React/TS、工程化、渲染、多端适配"},
    {"key": "client", "name": "客户端开发", "work": "安卓/iOS 原生 App、Windows 桌面软件开发", "skills": "Kotlin/Swift、QT/Electron、移动端适配"},
    {"key": "embedded", "name": "嵌入式底层开发", "work": "单片机、Linux 驱动、车载、物联网硬件软件开发", "skills": "C 语言、内核、驱动、硬件通信协议"},
    {"key": "ai", "name": "AI 算法", "work": "CV/NLP/大模型、推荐、AIGC 模型研发与落地", "skills": "深度学习、PyTorch、特征工程、向量数据库"},
    {"key": "bigdata", "name": "大数据开发", "work": "数据仓库、实时离线计算、数据平台建设", "skills": "Spark/Flink/Hive、SQL、数据建模"},
    {"key": "cloud", "name": "云计算云原生", "work": "云平台、容器、K8s、自动化运维、服务稳定性", "skills": "Docker/K8s、Go、CI/CD、监控容灾"},
    {"key": "security", "name": "网络安全", "work": "渗透攻防、安全工具开发、逆向、等保合规防护", "skills": "漏洞原理、逆向、WAF、等保规范"},
    {"key": "testing", "name": "测试体系", "work": "功能验证、自动化脚本、性能压测、自研测试平台", "skills": "测试框架、接口自动化、压测工具、SQL"},
    {"key": "game", "name": "游戏开发", "work": "游戏服务器、客户端、引擎、NPC 对战 AI 开发", "skills": "Unity/UE、网络编程、图形学、游戏服务"},
    {"key": "ic", "name": "芯片 IC 开发", "work": "数字芯片逻辑设计、功能验证、FPGA 开发", "skills": "Verilog、数字电路、时序、仿真验证"},
    {"key": "nondev", "name": "技术非开发岗", "work": "技术产品、售前解决方案、技术支持、开发者运营", "skills": "需求梳理、方案撰写、客户对接、基础技术"},
]


CATEGORY_KEYWORDS = {
    "backend": ["后端", "服务端", "java", "go", "c++", "spring", "fastapi", "django", "接口", "高并发", "微服务", "redis", "mysql", "中间件"],
    "frontend": ["前端", "vue", "react", "typescript", "javascript", "小程序", "可视化", "css", "html", "webpack", "vite"],
    "client": ["客户端", "android", "ios", "安卓", "kotlin", "swift", "qt", "electron", "桌面软件", "移动端"],
    "embedded": ["嵌入式", "单片机", "驱动", "linux 内核", "stm32", "arm", "硬件", "通信协议", "物联网", "车载"],
    "ai": ["算法", "机器学习", "深度学习", "nlp", "cv", "大模型", "推荐", "aigc", "pytorch", "tensorflow", "向量数据库"],
    "bigdata": ["大数据", "数仓", "数据仓库", "spark", "flink", "hive", "离线计算", "实时计算", "数据建模"],
    "cloud": ["云原生", "云计算", "kubernetes", "k8s", "docker", "devops", "ci/cd", "监控", "容灾", "sre"],
    "security": ["安全", "渗透", "漏洞", "逆向", "waf", "等保", "攻防", "ctf", "加固"],
    "testing": ["测试", "qa", "自动化测试", "压测", "性能测试", "测试用例", "pytest", "selenium", "jmeter"],
    "game": ["游戏", "unity", "ue", "unreal", "引擎", "图形学", "npc", "游戏服务器"],
    "ic": ["芯片", "ic", "fpga", "verilog", "数字电路", "时序", "仿真", "验证"],
    "nondev": ["产品", "售前", "解决方案", "技术支持", "开发者运营", "需求", "客户对接", "方案撰写"],
}


def classify_job_category(text: str) -> str:
    haystack = (text or "").lower()
    scores = {key: 0 for key in CATEGORY_KEYWORDS}
    for key, words in CATEGORY_KEYWORDS.items():
        for word in words:
            if word.lower() in haystack:
                scores[key] += 1
    best_key, best_score = max(scores.items(), key=lambda item: item[1])
    return best_key if best_score > 0 else "nondev"


def category_by_key(key: str) -> dict:
    for item in JOB_CATEGORIES:
        if item["key"] == key:
            return item
    return JOB_CATEGORIES[-1]


@dataclass(frozen=True)
class JobEntity:
    kind: str
    name: str


class JobEntityExtractor:
    position_words = ["工程师", "开发", "产品经理", "测试", "算法", "数据分析", "运营", "设计师", "实习生"]
    skill_words = [
        "Python", "Java", "C++", "Go", "Vue", "React", "SQL", "MySQL", "Linux", "Docker",
        "Kubernetes", "Redis", "NLP", "机器学习", "深度学习", "数据分析", "FastAPI", "Spring",
    ]
    education_words = ["大专", "本科", "硕士", "博士", "研究生", "不限学历"]
    city_words = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉", "西安", "苏州", "重庆", "天津"]

    def extract(self, text: str) -> List[JobEntity]:
        entities: Dict[tuple[str, str], JobEntity] = {}
        self._add_matches(entities, "skill", self.skill_words, text)
        self._add_matches(entities, "education", self.education_words, text)
        self._add_matches(entities, "city", self.city_words, text)

        for match in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,18}(?:公司|集团|科技|网络|信息|有限公司)", text):
            entities[("company", match)] = JobEntity("company", match)
        for match in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{1,16}(?:工程师|开发|产品经理|测试|算法|运营|设计师|实习生)", text):
            entities[("position", match)] = JobEntity("position", match)
        for match in re.findall(r"\d+(?:-\d+)?[kK]|[\d.]+万(?:/年|每年)?|薪资面议", text):
            entities[("salary", match)] = JobEntity("salary", match)
        return list(entities.values())

    @staticmethod
    def _add_matches(target: Dict[tuple[str, str], JobEntity], kind: str, words: Sequence[str], text: str) -> None:
        lower_text = text.lower()
        for word in words:
            if word.lower() in lower_text:
                target[(kind, word)] = JobEntity(kind, word)


class Neo4jKnowledgeStore:
    def __init__(self, settings):
        self.settings = settings
        self.enabled = bool(settings.neo4j_uri and settings.neo4j_password)
        self.driver = None
        self.extractor = JobEntityExtractor()
        if not self.enabled:
            return
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            self.driver.verify_connectivity()
        except Exception:
            self.driver = None
            self.enabled = False

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

    def rebuild(self, chunks: Iterable[DocumentChunk], vectors: np.ndarray | None) -> None:
        if not self.enabled or self.driver is None or vectors is None:
            return
        chunk_list = list(chunks)
        if len(chunk_list) != len(vectors):
            return
        rows = []
        for chunk, vector in zip(chunk_list, vectors):
            category_key = str(chunk.metadata.get("job_category") or classify_job_category(f"{chunk.source}\n{chunk.title}\n{chunk.content}"))
            category = category_by_key(category_key)
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "content": chunk.content,
                    "metadata_json": json.dumps(chunk.metadata, ensure_ascii=False),
                    "embedding": [float(value) for value in vector.tolist()],
                    "entities": [entity.__dict__ for entity in self.extractor.extract(chunk.content)],
                    "category": category,
                }
            )
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.execute_write(
                self._ensure_vector_index,
                int(vectors.shape[1]),
                self.settings.neo4j_vector_index,
            )
            session.execute_write(
                self._replace_graph,
                rows,
                JOB_CATEGORIES,
            )

    def search(self, query: str, query_vector: np.ndarray, top_k: int, category: str = "") -> List[RetrievedChunk]:
        if not self.enabled or self.driver is None:
            return []
        with self.driver.session(database=self.settings.neo4j_database) as session:
            rows = session.execute_read(
                self._vector_search,
                [float(value) for value in query_vector.tolist()],
                top_k,
                category,
            )
        return [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id=row["chunk_id"],
                    source=row["source"],
                    title=row["title"],
                    content=row["content"],
                    metadata=self._loads_metadata(row.get("metadata_json")),
                ),
                score=float(row["score"]),
            )
            for row in rows
        ]

    def graph_data(self, category: str = "", limit: int = 240) -> dict:
        if not self.enabled or self.driver is None:
            return {"nodes": [], "edges": [], "neo4j_enabled": False}
        with self.driver.session(database=self.settings.neo4j_database) as session:
            return session.execute_read(self._graph_data, category, limit)

    def node_detail(self, node_id: str, relation_type: str = "") -> dict:
        if not self.enabled or self.driver is None:
            return {"node": None, "relations": [], "neo4j_enabled": False}
        with self.driver.session(database=self.settings.neo4j_database) as session:
            return session.execute_read(self._node_detail, node_id, relation_type)

    def category_stats(self) -> list[dict]:
        if not self.enabled or self.driver is None:
            return [dict(item, entity_count=0, document_count=0) for item in JOB_CATEGORIES]
        with self.driver.session(database=self.settings.neo4j_database) as session:
            return session.execute_read(self._category_stats)

    def search_nodes(self, keyword: str, category: str = "", limit: int = 30) -> list[dict]:
        if not self.enabled or self.driver is None:
            return []
        with self.driver.session(database=self.settings.neo4j_database) as session:
            return session.execute_read(self._search_nodes, keyword, category, limit)

    def enrich(self, retrieved: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not retrieved or not self.enabled or self.driver is None:
            return retrieved
        ids = [item.chunk.chunk_id for item in retrieved]
        with self.driver.session(database=self.settings.neo4j_database) as session:
            related = session.execute_read(self._related_entities, ids)
        by_id: Dict[str, List[str]] = {}
        for row in related:
            by_id.setdefault(row["chunk_id"], []).append(f"{row['label']}:{row['name']}")
        enriched = []
        for item in retrieved:
            entities = by_id.get(item.chunk.chunk_id, [])
            if entities:
                metadata = dict(item.chunk.metadata)
                metadata["graph_entities"] = "；".join(sorted(set(entities)))
                content = f"{item.chunk.content}\n\n[Graph entities] {metadata['graph_entities']}"
                chunk = DocumentChunk(item.chunk.chunk_id, item.chunk.source, item.chunk.title, content, metadata)
                enriched.append(RetrievedChunk(chunk=chunk, score=item.score))
            else:
                enriched.append(item)
        return enriched

    @staticmethod
    def _index_identifier(name: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""):
            return "job_chunk_embedding_index"
        return name

    @staticmethod
    def _loads_metadata(value) -> dict[str, str]:
        if not value:
            return {}
        try:
            data = json.loads(str(value))
            return {key: str(item) for key, item in dict(data).items()}
        except Exception:
            return {}

    @classmethod
    def _ensure_vector_index(cls, tx, dimension: int, index_name: str) -> None:
        index_name = cls._index_identifier(index_name)
        tx.run(
            f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
              `vector.dimensions`: $dimension,
              `vector.similarity_function`: 'cosine'
            }}}}
            """,
            dimension=dimension,
        )

    @staticmethod
    def _replace_graph(tx, rows: List[dict], categories: List[dict]) -> None:
        tx.run("MATCH (c:Chunk) DETACH DELETE c")
        tx.run("MATCH (d:Document) WHERE NOT (d)--() DETACH DELETE d")
        for label in ENTITY_LABELS.values():
            tx.run(f"MATCH (e:{label}) WHERE NOT (e)--() DETACH DELETE e")
        tx.run(
            """
            UNWIND $categories AS category
            MERGE (jc:JobCategory {key: category.key})
            SET jc.name = category.name,
                jc.work = category.work,
                jc.skills = category.skills,
                jc.type = 'JobCategory'
            """,
            categories=categories,
        )
        tx.run(
            """
            UNWIND $rows AS row
            MERGE (d:Document {source: row.source})
            SET d.title = row.title,
                d.name = row.title,
                d.type = 'Document'
            MERGE (jc:JobCategory {key: row.category.key})
            MERGE (d)-[:RELATED_TO]->(jc)
            CREATE (c:Chunk {
              chunk_id: row.chunk_id,
              source: row.source,
              title: row.title,
              content: row.content,
              metadata_json: row.metadata_json,
              embedding: row.embedding
            })
            MERGE (d)-[:HAS_CHUNK]->(c)
            MERGE (c)-[:RELATED_TO]->(jc)
            WITH c, row, jc, d
            UNWIND row.entities AS entity
            CALL {
              WITH c, entity, jc, d
              WITH c, entity, jc, d WHERE entity.kind = 'position'
              MERGE (e:Position {name: entity.name})
              SET e.type = 'Position'
              MERGE (c)-[:MENTIONS]->(e)
              MERGE (jc)-[:CONTAINS]->(e)
              MERGE (d)-[:RELATED_TO]->(e)
              RETURN count(*) AS done
              UNION
              WITH c, entity, jc, d
              WITH c, entity WHERE entity.kind = 'company'
              MERGE (e:Company {name: entity.name})
              SET e.type = 'Company'
              MERGE (c)-[:MENTIONS]->(e)
              RETURN count(*) AS done
              UNION
              WITH c, entity, jc, d
              WITH c, entity, jc, d WHERE entity.kind = 'skill'
              MERGE (e:Skill {name: entity.name})
              SET e.type = 'Skill'
              MERGE (c)-[:REQUIRES]->(e)
              MERGE (jc)-[:REQUIRES]->(e)
              WITH c, entity, jc, d, e
              OPTIONAL MATCH (jc)-[:CONTAINS]->(p:Position)
              FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (p)-[:REQUIRES]->(e))
              RETURN count(*) AS done
              UNION
              WITH c, entity, jc, d
              WITH c, entity WHERE entity.kind = 'salary'
              MERGE (e:Salary {name: entity.name})
              SET e.type = 'Salary'
              MERGE (c)-[:OFFERS]->(e)
              RETURN count(*) AS done
              UNION
              WITH c, entity, jc, d
              WITH c, entity WHERE entity.kind = 'city'
              MERGE (e:City {name: entity.name})
              SET e.type = 'City'
              MERGE (c)-[:LOCATED_IN]->(e)
              RETURN count(*) AS done
              UNION
              WITH c, entity, jc, d
              WITH c, entity WHERE entity.kind = 'education'
              MERGE (e:Education {name: entity.name})
              SET e.type = 'Education'
              MERGE (c)-[:REQUIRES_EDUCATION]->(e)
              RETURN count(*) AS done
            }
            RETURN count(*)
            """,
            rows=rows,
        )

    def _vector_search(self, tx, embedding: List[float], top_k: int, category: str = "") -> List[dict]:
        result = tx.run(
            """
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node, score
            OPTIONAL MATCH (node)-[:RELATED_TO]->(jc:JobCategory)
            WITH node, score, jc
            WHERE $category = '' OR jc.key = $category
            OPTIONAL MATCH (node)-[]-(entity)<-[]-(neighbor:Chunk)
            WHERE neighbor.chunk_id <> node.chunk_id
            WITH node, score, collect(DISTINCT neighbor.content)[0..2] AS neighbor_context
            RETURN node.chunk_id AS chunk_id,
                   node.source AS source,
                   node.title AS title,
                   node.content +
                     CASE WHEN size(neighbor_context) = 0 THEN ''
                          ELSE reduce(text = '\n\n[Related graph context]\n', item IN neighbor_context | text + item + '\n---\n')
                     END AS content,
                   node.metadata_json AS metadata_json,
                   score AS score
            ORDER BY score DESC
            """,
            index_name=self.settings.neo4j_vector_index,
            top_k=max(1, int(top_k) * 4) if category else max(1, int(top_k)),
            embedding=embedding,
            category=category,
        )
        return [dict(record) for record in result][: max(1, int(top_k))]

    @staticmethod
    def _related_entities(tx, chunk_ids: List[str]) -> List[dict]:
        result = tx.run(
            """
            MATCH (c:Chunk)-[r]-(e)
            WHERE c.chunk_id IN $chunk_ids AND NOT e:Chunk AND NOT e:Document
            RETURN c.chunk_id AS chunk_id, labels(e)[0] AS label, e.name AS name
            """,
            chunk_ids=chunk_ids,
        )
        return [dict(record) for record in result]

    @staticmethod
    def _graph_data(tx, category: str, limit: int) -> dict:
        result = tx.run(
            """
            MATCH (n)
            WHERE n:JobCategory OR n:Position OR n:Skill OR n:Document
            OPTIONAL MATCH (n)-[*0..2]-(jc:JobCategory)
            WITH n, collect(DISTINCT jc.key) AS categories
            WHERE $category = '' OR n.key = $category OR $category IN categories
            WITH n LIMIT $limit
            OPTIONAL MATCH (n)-[r]-(m)
            WHERE m:JobCategory OR m:Position OR m:Skill OR m:Document
            WITH collect(DISTINCT n) AS base_nodes, collect(DISTINCT {source: n, rel: r, target: m}) AS rels
            UNWIND base_nodes AS node
            WITH base_nodes, rels, collect(DISTINCT {
              id: elementId(node),
              label: coalesce(node.name, node.title, node.source, node.key),
              type: labels(node)[0],
              category: node.key,
              properties: properties(node)
            }) AS base_payload
            UNWIND rels AS rel
            WITH base_payload, rels, collect(DISTINCT rel.target) AS neighbor_nodes
            WITH base_payload, rels, [node IN neighbor_nodes WHERE node IS NOT NULL][0..$limit] AS neighbor_nodes
            RETURN base_payload + [node IN neighbor_nodes | {
              id: elementId(node),
              label: coalesce(node.name, node.title, node.source, node.key),
              type: labels(node)[0],
              category: node.key,
              properties: properties(node)
            }] AS nodes,
            [rel IN rels WHERE rel.rel IS NOT NULL | {
              id: elementId(rel.rel),
              source: elementId(rel.source),
              target: elementId(rel.target),
              type: type(rel.rel)
            }] AS edges
            """,
            category=category,
            limit=max(20, min(int(limit or 240), 500)),
        )
        record = result.single()
        if not record:
            return {"nodes": [], "edges": [], "neo4j_enabled": True}
        nodes = {item["id"]: item for item in record["nodes"]}.values()
        edges = {item["id"]: item for item in record["edges"]}.values()
        return {"nodes": list(nodes), "edges": list(edges), "neo4j_enabled": True}

    @staticmethod
    def _node_detail(tx, node_id: str, relation_type: str) -> dict:
        result = tx.run(
            """
            MATCH (n)
            WHERE elementId(n) = $node_id
            OPTIONAL MATCH (n)-[r]-(m)
            WHERE $relation_type = '' OR type(r) = $relation_type
            RETURN {
              id: elementId(n),
              label: coalesce(n.name, n.title, n.source, n.key),
              type: labels(n)[0],
              properties: properties(n)
            } AS node,
            collect(DISTINCT {
              id: elementId(r),
              type: type(r),
              direction: CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END,
              node: {
                id: elementId(m),
                label: coalesce(m.name, m.title, m.source, m.key),
                type: labels(m)[0],
                properties: properties(m)
              }
            }) AS relations
            """,
            node_id=node_id,
            relation_type=relation_type,
        )
        record = result.single()
        if not record:
            return {"node": None, "relations": [], "neo4j_enabled": True}
        return {
            "node": record["node"],
            "relations": [item for item in record["relations"] if item.get("id")],
            "neo4j_enabled": True,
        }

    @staticmethod
    def _category_stats(tx) -> list[dict]:
        result = tx.run(
            """
            MATCH (jc:JobCategory)
            OPTIONAL MATCH (jc)--(entity)
            WHERE NOT entity:Chunk
            OPTIONAL MATCH (doc:Document)-[:RELATED_TO]->(jc)
            RETURN jc.key AS key,
                   jc.name AS name,
                   jc.work AS work,
                   jc.skills AS skills,
                   count(DISTINCT entity) AS entity_count,
                   count(DISTINCT doc) AS document_count
            ORDER BY key
            """
        )
        return [dict(record) for record in result]

    @staticmethod
    def _search_nodes(tx, keyword: str, category: str, limit: int) -> list[dict]:
        result = tx.run(
            """
            MATCH (n)
            WHERE (n:JobCategory OR n:Position OR n:Skill OR n:Document)
              AND toLower(coalesce(n.name, n.title, n.source, n.key, '')) CONTAINS toLower($keyword)
            OPTIONAL MATCH (n)-[*0..2]-(jc:JobCategory)
            WITH n, collect(DISTINCT jc.key) AS categories
            WHERE $category = '' OR n.key = $category OR $category IN categories
            RETURN elementId(n) AS id,
                   coalesce(n.name, n.title, n.source, n.key) AS label,
                   labels(n)[0] AS type,
                   properties(n) AS properties
            LIMIT $limit
            """,
            keyword=keyword,
            category=category,
            limit=max(1, min(int(limit or 30), 100)),
        )
        return [dict(record) for record in result]
