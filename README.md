# 大学生求职岗位知识问答系统

面向生产实习交付的 RAG 求职知识问答系统，支持用户注册登录、文档知识库上传、流式回答、参考来源展示和校园网访问。

## 快速启动

```bash
cd E:\26大三下\26课内学习\生产实习\project\rag_student_job_qa_system
D:\Anaconda\python.exe run.py
```

访问：

```text
http://127.0.0.1:8000
```

校园网共享：

```bash
D:\Anaconda\python.exe run.py --host 0.0.0.0 --port 8000
```

同学访问：

```text
http://你的IPv4地址:8000
```

## 核心功能

- 用户注册、登录、退出
- 多轮求职问答
- 流式输出回答
- 知识库来源引用
- PDF/TXT/Markdown 文档入库
- 默认 SQLite 用户库，支持切换 MySQL
- Hugging Face Embedding 优先，未缓存模型时自动降级

更多说明见：

- [docs/部署手册.md](docs/部署手册.md)
- [docs/使用说明.md](docs/使用说明.md)
