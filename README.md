# rag-service

> 独立 RAG 服务：PDF 复杂解析、混合检索、Cross-Encoder 重排。
> 本科毕设 v2.0 重构 change 1 产物，对内供 ai-agent-learning 调用，对外可独立部署复用。

## 项目定位

rag-service 是从主系统 `ai-agent-learning` 抽离出来的独立 RAG 服务，专注以下职责：

- **复杂文档解析（PDF）**：双轨架构 — 配置 `MINERU_API_TOKEN` 时调用 [MinerU](https://mineru.net) 云端 SOTA 解析（论文核心，借鉴 airQA 项目）；未配置时降级为 PyMuPDF + 启发式版面分析
- 混合检索（Dense 向量 + Sparse BM25 + RRF 融合）
- Cross-Encoder 精排（BAAI/bge-reranker-v2-m3）
- Collection 与文档元数据 CRUD
- 健康检查与降级

详细设计见主系统 `docs/design-spec/01_正式设计/11_RAG服务独立项目设计.md`。

## PDF 解析能力（双轨架构）

| 模式 | 触发条件 | 实现 | 论文价值 |
| --- | --- | --- | --- |
| MinerU 云端 | `MINERU_API_TOKEN` 已配置 | [app/parser/mineru/](app/parser/mineru/) | 高 — MinerU 是当前 SOTA 的 PDF 复杂解析 SaaS（vlm 视觉语言模型） |
| PyMuPDF 启发式 | 未配置 token 或 MinerU 失败 | [app/parser/pdf_parser.py](app/parser/pdf_parser.py) + [layout/](app/parser/layout/) | 中 — 工程鲁棒性 fallback |

MinerU 模块借鉴自用户自有的 airQA 项目（[src/chunking/](https://github.com/ArtLjn/NSQA)），包含：
- [client.py](app/parser/mineru/client.py)：HTTP 上传 + 轮询 + zip 下载
- [constants.py](app/parser/mineru/constants.py)：50+ MinerU BlockType 与 chunk category 完整映射
- [latex_normalizer.py](app/parser/mineru/latex_normalizer.py)：MinerU LaTeX 噪声修复正则
- [parser.py](app/parser/mineru/parser.py)：content_list_v2 → Chunk 转换器

### 启用 MinerU

1. 到 https://mineru.net 注册账号（免费 100 页/天）
2. 在 [rag-service/.env](.env) 中填入：
   ```env
   MINERU_API_TOKEN=your-token-here
   MINERU_MODEL_VERSION=vlm  # vlm=视觉语言模型；pipeline=传统流水线
   ```
3. 重启服务，`POST /parse` 即走 MinerU

## 快速启动

### 方式 1：docker-compose 一键启动（推荐）

```bash
cd rag-service
cp .env.example .env
docker-compose up -d
```

启动后访问：
- rag-service：http://localhost:8001
- Qdrant 控制台：http://localhost:6333/dashboard

### 方式 2：本地开发（不使用 Docker）

依赖：
- Python 3.11+
- Tesseract OCR + 中文语言包（macOS：`brew install tesseract-lang`；Ubuntu：`apt install tesseract-ocr tesseract-ocr-chi-sim`）
- Qdrant 实例（可 docker 单独启动 `docker run -p 6333:6333 qdrant/qdrant`）

```bash
cd rag-service
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8001
```

首次启动时 Embedding 与 Reranker 模型会自动下载（约 2GB），网络较慢建议提前预下载到 HF 缓存。

## 与主系统集成

主系统 ai-agent-learning 通过 `tools/rag_client.py`（待实现）HTTP 调用：

```python
import httpx

async def retrieve(query: str):
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            "http://localhost:8001/retrieve",
            json={"query": query, "collection": "ticket_knowledge", "mode": "hybrid"},
        )
        return response.json()
```

主系统侧的 client 完整实现属于下一个 change `switch-main-system-to-rag-client`。

## API 列表

| 路径 | 方法 | 说明 |
| --- | --- | --- |
| `/parse` | POST | 解析与分块（不写库），预览分块效果 |
| `/ingest` | POST | 完整入库流水线（解析 → 分块 → 向量化 → 写 Qdrant + SQLite） |
| `/retrieve` | POST | vector / bm25 / hybrid 三模式检索 |
| `/rerank` | POST | Cross-Encoder 精排 |
| `/collections` | GET / POST | 列出 / 创建 collection |
| `/collections/{name}` | DELETE | 删除 collection |
| `/collections/{name}/documents` | GET | 分页查询文档 |
| `/collections/{name}/documents/{doc_id}` | DELETE | 删除指定文档 |
| `/health` | GET | 健康检查（qdrant/embedder/reranker 三组件状态） |

详细 API 契约见 [docs/api.md](docs/api.md)。

## 配置说明

通过环境变量或 `.env` 文件配置（默认值见 `.env.example`）。默认对齐主系统 `ai-agent-learning/config.yaml` 的端点与模型：

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | 8001 | 服务监听端口 |
| `QDRANT_URL` | http://localhost:6333 | Qdrant 地址（生产部署填主系统同款） |
| `QDRANT_API_KEY` | (空) | Qdrant api_key（生产部署填主系统同款） |
| `EMBEDDING_BASE_URL` | https://generativelanguage.googleapis.com/v1beta | Embedding API 网关（OpenAI 兼容） |
| `EMBEDDING_API_KEY` | (空) | Embedding API 密钥 |
| `EMBEDDING_MODEL` | gemini-embedding-001 | Embedding 模型 |
| `EMBEDDING_DIM` | 3072 | 向量维度（与模型一致） |
| `EMBEDDING_BATCH_SIZE` | 64 | 单次 API 请求批量 |
| `RERANKER_MODEL` | BAAI/bge-reranker-v2-m3 | 本地 Cross-Encoder 重排模型 |
| `MINERU_API_TOKEN` | (空) | MinerU 云端 PDF 解析 token；留空则降级 PyMuPDF |
| `MINERU_BASE_URL` | https://mineru.net/api/v4 | MinerU API 网关 |
| `MINERU_MODEL_VERSION` | vlm | vlm=视觉语言模型；pipeline=传统流水线 |
| `MINERU_TIMEOUT` | 600 | MinerU 解析超时（秒） |
| `MINERU_POLL_INTERVAL` | 3 | 轮询间隔（秒） |
| `DEFAULT_CHUNK_SIZE` | 500 | fixed 分块默认大小 |
| `DEFAULT_TOP_K` | 10 | 默认召回数量 |
| `RRF_K` | 60 | RRF 常数 |
| `RRF_VECTOR_WEIGHT` | 0.7 | 向量权重 |
| `RRF_SPARSE_WEIGHT` | 0.3 | BM25 权重 |
| `HTTP_TIMEOUT` | 30 | 客户端超时（秒） |
| `SCORE_THRESHOLD` | 0.3 | dense 检索默认 score 阈值 |
| `METADATA_DB_PATH` | data/rag_metadata.db | SQLite 元数据路径 |
| `HYDE_LLM_BASE_URL` | (空) | HyDE LLM 网关（OpenAI 兼容） |
| `HYDE_LLM_API_KEY` | (空) | HyDE LLM 密钥 |
| `HYDE_LLM_MODEL` | gpt-4o-mini | HyDE LLM 模型名 |

> 部署到与主系统共用同一套基础设施时，把主系统 `config.yaml` 中 `qdrant_url`/`qdrant_api_key`/`embedding_*`/`llm_*` 的值复制到 rag-service `.env` 即可。

## 测试

```bash
# 全部测试
pytest

# 仅单元测试
pytest tests/parser tests/retrieval tests/storage tests/services

# 集成测试
pytest tests/integration

# 覆盖率
pytest --cov=app --cov-report=term-missing
```

当前覆盖率：约 77%。

## 项目结构

```
rag-service/
├── app/
│   ├── api/          # 5 个 HTTP 路由 + collections 管理
│   ├── parser/       # 文档解析（layout/table/chunker/cleaner）
│   ├── retrieval/    # 检索与重排（embedder/dense/sparse/hybrid/hyde/reranker）
│   ├── storage/      # Qdrant 接入 + SQLite 元数据
│   ├── services/     # 编排层（parse/ingest/retrieve/rerank/collection/health）
│   ├── core/         # config/logging/exceptions/metrics/response
│   ├── models/       # Pydantic 数据模型
│   └── main.py       # FastAPI 入口
├── tests/
│   ├── api/          # API 端到端测试（TestClient + mock）
│   ├── parser/       # parser/chunker/layout/table 单测
│   ├── retrieval/    # dense/sparse/hybrid/reranker/hyde 单测
│   ├── storage/      # qdrant_client/metadata_store 单测
│   ├── services/     # parse/ingest/collection service 单测
│   └── integration/  # 端到端 + 降级 + 契约测试
├── docs/             # 部署与 API 文档
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## 降级策略

| 触发条件 | 行为 |
| --- | --- |
| Qdrant 不可用 | `/retrieve` `/ingest` 返回 503；`/health` 标记 qdrant=unavailable |
| Embedder 不可用 | `mode=vector` / `mode=hybrid` 自动退化为 bm25，返回 `warning` 字段 |
| Cross-Encoder 不可用 | `/rerank` 按原 score 排序，返回 `warning: "reranker_degraded"` |
| PDF 版面模型不可用 | 解析失败时降级为 fixed 分块并返回 `warning` |

## License

本科毕设项目，未对外发布。
