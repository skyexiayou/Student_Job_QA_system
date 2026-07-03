# 大学生求职岗位知识问答系统

这是一个面向生产实习和课程答辩的 RAG 智能问答系统。系统围绕大学生求职场景设计，支持用户注册登录、知识文档上传、RAG 流式问答、岗位助手、模拟面试、知识图谱、语音输入、学习记录和求职目标管理。

用户可以上传岗位 JD、企业招聘资料、面试题、简历指导文档等资料。系统会自动完成文档解析、文本清洗、文本分块、向量化和索引构建。提问时，系统先从知识库检索相关片段，再调用大模型生成结构化中文回答，并展示参考来源。

## 项目亮点

- 完整 RAG 链路：文档入库、文本分块、Embedding、向量检索、提示词增强、来源追溯。
- 流式智能问答：后端通过 NDJSON 流式返回，前端逐步展示回答。
- 多功能前端页面：首页、岗位助手、知识图谱、知识问答、模拟面试、知识文档、学习记录。
- 多格式知识库：支持 TXT、Markdown、PDF、DOCX、XLSX。
- 用户体系：基于 MySQL 保存用户、登录令牌、头像和求职目标。
- 图谱能力：可接入 Neo4j；未连接时使用本地知识库生成兜底图谱。
- 语音能力：支持 FunASR 语音识别和浏览器朗读。
- 演示友好：支持本机启动和校园网共享访问。

## 核心功能

| 模块 | 功能 |
| --- | --- |
| 用户管理 | 注册、登录、退出、头像上传、个人信息修改、密码修改。 |
| 首页 | 热门岗位入口、模拟面试入口、知识文档入口、求职目标倒计时。 |
| 知识问答 | RAG 多轮问答、流式输出、参考来源、语音输入、回答朗读、历史保存。 |
| 知识文档 | 上传、解析、分块、分类、索引重建、搜索、筛选、预览、删除。 |
| 岗位助手 | 按岗位方向检索资料，生成岗位要求、技能准备、学习路线和项目建议。 |
| 模拟面试 | AI 面试官根据用户回答进行追问，并给出简短分析。 |
| 知识图谱 | 展示岗位、文档、技能等节点关系，支持搜索、筛选和节点详情。 |
| 学习记录 | 查看、筛选、打开、重命名和删除历史会话。 |

## 技术栈

| 类型 | 技术 |
| --- | --- |
| 后端 | Python、FastAPI、Uvicorn |
| 前端 | Vue 3、HTML、CSS、JavaScript |
| 大模型 | OpenAI Compatible API，默认模型 `qwen-plus` |
| RAG | 文档解析、文本分块、Embedding、Top-K 召回、提示词增强 |
| 向量检索 | sentence-transformers、FAISS、numpy fallback、hashing embedding fallback |
| 文档解析 | pypdf、python-docx、openpyxl |
| 数据库 | MySQL、PyMySQL |
| 图数据库 | Neo4j、neo4j Python driver |
| 语音识别 | FunASR、ModelScope |
| 测试 | pytest |

## 快速启动

进入项目目录：

```powershell
cd "E:\26大三下\26课内学习\生产实习\project\rag_student_job_qa_system"
```

安装依赖：

```powershell
D:\Anaconda\python.exe -m pip install -r requirements.txt
```

启动系统：

```powershell
D:\Anaconda\python.exe run.py
```

浏览器访问：

```text
http://127.0.0.1:8000
```

接口文档：

```text
http://127.0.0.1:8000/docs
```

校园网共享访问：

```powershell
D:\Anaconda\python.exe run.py --host 0.0.0.0 --port 8000
```

伙伴访问：

```text
http://你的IPv4地址:8000
```

## 必要配置

系统至少需要 MySQL 用于用户登录相关功能。建议在项目根目录创建 `.env`：

```text
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=你的MySQL密码
MYSQL_DATABASE=rag_job_qa

QWEN_API_KEY=你的APIKey
QWEN_BASE_URL=你的OpenAI兼容接口地址
QWEN_MODEL=qwen-plus
```

创建 MySQL 数据库：

```sql
CREATE DATABASE IF NOT EXISTS rag_job_qa
DEFAULT CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;
```

系统启动时会自动创建或补齐用户相关表。

更多环境、模型、Neo4j 和排错说明见 [docs/部署手册.md](docs/部署手册.md)。

## 项目结构

```text
rag_student_job_qa_system/
├─ run.py                         # 启动入口，加载配置并启动 FastAPI + Vue 页面
├─ requirements.txt               # Python 依赖
├─ pytest.ini                     # pytest 配置
├─ README.md                      # 项目总览
├─ PRODUCT.md                     # 产品说明
├─ docs/
│  ├─ 使用说明.md                 # 系统框架、功能板块、文件职责、答辩说明
│  └─ 部署手册.md                 # 环境配置、启动部署、常见问题排查
├─ frontend/
│  ├─ index.html                  # Vue 单页前端主文件
│  └─ vendor/vue.global.prod.js   # 本地 Vue 运行时
├─ scripts/
│  ├─ start.bat                   # Windows 启动脚本
│  ├─ start_api.bat               # API 启动脚本
│  ├─ start_ui.bat                # 前端相关启动脚本
│  └─ download_hf_model.py        # Hugging Face 模型下载脚本
├─ src/rag_job_qa/
│  ├─ api.py                      # FastAPI 路由层
│  ├─ config.py                   # 配置读取和路径设置
│  ├─ rag_service.py              # RAG 问答编排
│  ├─ llm_client.py               # 大模型客户端
│  ├─ knowledge_base.py           # 知识库导入、检索、文档管理
│  ├─ vector_store.py             # 向量索引和相似度检索
│  ├─ text_processing.py          # 文档读取、清洗、分块
│  ├─ neo4j_store.py              # Neo4j 知识图谱
│  ├─ user_store.py               # MySQL 用户和目标管理
│  ├─ conversation.py             # 会话历史管理
│  ├─ speech_service.py           # FunASR 语音识别
│  ├─ cache.py                    # 问答缓存
│  └─ models.py                   # 数据模型
├─ data/
│  ├─ knowledge_base/             # 内置知识库和上传文档
│  ├─ index/                      # 向量索引和文本块
│  ├─ photo/                      # 用户头像和系统头像
│  ├─ api_key/                    # API Key CSV
│  ├─ conversations/              # 运行后生成的会话记录
│  └─ models/                     # 运行后生成的模型缓存
└─ tests/                         # 单元测试
```

## 主要数据流

文档入库：

```text
上传文档
  -> text_processing.py 解析和清洗
  -> split_text 分块
  -> knowledge_base.py 分类和导入
  -> vector_store.py 生成向量并保存索引
  -> neo4j_store.py 可选写入图谱
```

用户问答：

```text
用户提问
  -> api.py 接收请求
  -> rag_service.py 组织 RAG 流程
  -> knowledge_base.py 检索相关片段
  -> llm_client.py 调用大模型
  -> conversation.py 保存历史
  -> 前端流式展示回答和参考来源
```

## 运行测试

```powershell
cd "E:\26大三下\26课内学习\生产实习\project\rag_student_job_qa_system"
D:\Anaconda\python.exe -m pytest
```

## 文档导航

- [PRODUCT.md](PRODUCT.md)：产品定位、目标用户、核心价值和产品边界。
- [docs/使用说明.md](docs/使用说明.md)：系统框架、功能板块、文件职责、配置项和答辩讲解顺序。
- [docs/部署手册.md](docs/部署手册.md)：MySQL、API Key、Embedding、Neo4j、启动、共享访问和故障排查。

## 答辩建议

答辩时建议按以下顺序展示：

1. 项目背景：大学生求职资料分散，准备成本高。
2. 系统架构：前端、后端、MySQL、向量库、大模型、Neo4j、语音模块。
3. 文档入库：上传资料，说明解析、分块、向量化和索引。
4. RAG 问答：提问并展示流式回答和参考来源。
5. 岗位助手：选择岗位方向并继续追问。
6. 模拟面试：展示 AI 面试官追问和分析。
7. 知识图谱：展示岗位、文档和技能关系。
8. 学习记录与目标管理：展示系统完整性。
