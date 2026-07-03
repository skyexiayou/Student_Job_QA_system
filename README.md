# 大学生求职岗位知识问答系统

本项目是一个面向生产实习和课程答辩的 RAG 求职知识问答系统。系统支持用户注册登录、知识文档上传、流式智能问答、岗位分类检索、模拟面试、知识图谱、语音输入、历史会话和个人求职目标管理，适合在本地或校园网环境中演示。

## 项目定位

系统的目标是帮助大学生围绕实习、校招、简历、面试和岗位选择进行知识检索与问答。用户可以上传岗位 JD、企业招聘资料、面试题、简历指导材料等文档，系统会自动解析、分块、向量化并建立索引。提问时，系统先检索知识库，再调用大模型生成中文结构化回答，并展示参考来源。

## 核心功能

- 用户管理：注册、登录、退出、头像上传、个人信息维护、密码修改。
- 知识问答：基于 RAG 的多轮问答、流式输出、参考来源展示、回答缓存。
- 知识库管理：文档上传、解析、分块、分类、索引重建、列表检索、预览和删除。
- 岗位助手：按岗位方向筛选知识材料，提供岗位要求、技能路径和项目准备建议。
- 模拟面试：AI 面试官根据学生回答进行点评并继续追问。
- 知识图谱：展示岗位、文档、技能、职位等节点关系，支持搜索和分类筛选。
- 语音能力：支持语音识别输入、WebSocket 流式识别和浏览器朗读回答。
- 历史记录：保存不同类型会话，支持查看、重命名和删除。
- 目标管理：添加求职目标和日期，显示倒计时。

## 技术栈

- 后端框架：FastAPI、Uvicorn
- 前端框架：Vue 3 单页应用、原生 HTML/CSS/JavaScript
- 大模型接口：兼容 OpenAI 协议的接口，默认适配通义千问 `qwen-plus`
- RAG 检索：文档解析、文本分块、Embedding、相似度召回
- 向量检索：sentence-transformers、FAISS；不可用时自动降级到 numpy 本地向量检索
- 文档解析：pypdf、python-docx、openpyxl、Markdown/TXT 解析
- 数据库：MySQL，存储用户、登录令牌、个人目标等数据
- 知识图谱：Neo4j，可选组件；未连接时使用本地分类/文档图谱兜底
- 语音识别：FunASR
- 测试工具：pytest

## 项目结构

```text
rag_student_job_qa_system/
├─ run.py                         # 系统启动入口，创建 FastAPI 应用并启动 Uvicorn
├─ requirements.txt               # Python 依赖列表
├─ pytest.ini                     # pytest 测试配置
├─ PRODUCT.md                     # 产品定位与产品说明
├─ README.md                      # 项目总览
├─ docs/
│  ├─ 使用说明.md                 # 面向答辩和团队协作的完整系统说明
│  └─ 部署手册.md                 # 环境准备、部署、启动和排错步骤
├─ scripts/
│  ├─ start.bat                   # Windows 一键启动脚本
│  ├─ start_api.bat               # API 启动脚本
│  ├─ start_ui.bat                # 前端相关启动脚本
│  └─ download_hf_model.py        # Hugging Face 模型下载辅助脚本
├─ frontend/
│  ├─ index.html                  # Vue 单页前端，包含主要页面、样式和交互逻辑
│  └─ vendor/vue.global.prod.js   # 本地 Vue 运行时
├─ src/rag_job_qa/
│  ├─ api.py                      # FastAPI 路由：认证、问答、知识库、图谱、语音等接口
│  ├─ config.py                   # 配置读取、路径设置、API Key 加载
│  ├─ rag_service.py              # RAG 编排：检索、构造提示词、调用 LLM、追加来源
│  ├─ knowledge_base.py           # 知识库门面：导入、重建、检索、文档管理、图谱兜底
│  ├─ vector_store.py             # 向量库：Embedding、FAISS/numpy 检索、索引持久化
│  ├─ neo4j_store.py              # Neo4j 知识图谱：建图、图检索、节点关系管理
│  ├─ text_processing.py          # 文档读取、清洗、分块
│  ├─ llm_client.py               # 大模型 HTTP 客户端和流式调用
│  ├─ conversation.py             # 会话历史管理
│  ├─ user_store.py               # MySQL 用户、令牌、目标管理
│  ├─ speech_service.py           # 语音识别服务
│  ├─ cache.py                    # 问答缓存
│  └─ models.py                   # 数据模型
├─ data/
│  ├─ knowledge_base/             # 内置知识文档和上传文档目录
│  ├─ index/                      # 向量索引、文本块和元数据
│  ├─ photo/                      # 用户头像和系统头像
│  ├─ api_key/                    # API Key CSV 放置目录
│  └─ users.db                    # 早期本地数据文件，当前主要使用 MySQL
└─ tests/                         # 单元测试
```

## 快速启动

进入项目目录：

```powershell
cd "E:\26大三下\26课内学习\生产实习\project\rag_student_job_qa_system"
```

安装依赖：

```powershell
D:\Anaconda\python.exe -m pip install -r requirements.txt
```

启动本机服务：

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

同学访问：

```text
http://你的IPv4地址:8000
```

## 文档入口

- [PRODUCT.md](PRODUCT.md)：产品定位、目标用户、核心价值和边界。
- [docs/使用说明.md](docs/使用说明.md)：系统框架、功能板块、文件职责、技术栈和答辩讲解建议。
- [docs/部署手册.md](docs/部署手册.md)：环境安装、数据库、模型、API Key、Neo4j、启动和排错。
