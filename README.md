# 企业智能服务台（模块化版）

基于 Streamlit 的混合检索 RAG 问答系统

## 目录结构

| 文件 | 职责 |
|---|---|
| `app.py` | 应用入口：页面配置、会话状态、持久化知识库加载、侧边栏、快捷提问、提问入口 |
| `config.py` | 全局配置：HF 镜像端点、路径、Rerank 阈值、父子分块参数 |
| `llm.py` | 公共 LLM 客户端（DeepSeek） |
| `query_analysis.py` | 多轮对话改写 / 多跳分解 / 查询变体生成 |
| `document_loader.py` | 文件解析（pdf/docx/txt）+ 对话历史保存 |
| `ingest.py` | 上传入库：父子分块 → 向量化 → 增量入库 → 重建 BM25 |
| `qa_flow.py` | 提问主流程：改写 → 多跳/变体 → 混合检索 → RRF 融合 → 子块重排 → 父块回溯 → 拒答判断 → 流式生成 |

## 功能特性

- 父子分块检索：子块（250 字）做检索，父块（1200 字）做上下文
- 混合检索：向量 + BM25，RRF 融合（键为元组，防碰撞）
- 子块 Rerank（分批计算防 OOM）+ 父块回溯（保序取回）
- 拒答阈值：Rerank 分数 < 0.2 不回答，防幻觉
- 多轮改写、多查询变体、多跳问题自动拆解迭代检索
- 流式输出、对话历史持久化、快捷提问按钮

## 运行方式

```bash
pip install -r requirements.txt
streamlit run app.py
```

需在项目根目录准备 `.env` 文件：

```
API_KEY=你的DeepSeek_API_Key
BASE_URL=https://api.deepseek.com/v1
```

> 运行时会在项目目录生成 `chroma_db/`（向量库）与 `chat_history.json`（对话历史），属正常产物。
