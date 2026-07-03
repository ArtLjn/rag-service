"""集中配置：环境变量加载与默认值。

设计原则：
- 默认值全部使用 localhost / 标准占位符，不嵌入任何 IP 或 API Key
- 部署时由 .env 注入实际端点与密钥
- 与主系统 ai-agent-learning/config.yaml 对齐的部署指引见 README.md「与主系统集成」
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    port: int = Field(default=8001, alias="PORT")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    # Embedding：在线 HTTP（默认 Google gemini-embedding-001 原生协议）
    embedding_provider: str = Field(default="google", alias="EMBEDDING_PROVIDER")  # google | openai
    embedding_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta", alias="EMBEDDING_BASE_URL")
    embedding_api_key: str | None = Field(default=None, alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="gemini-embedding-001", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=3072, alias="EMBEDDING_DIM")
    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")

    # Reranker：四种 provider 可选
    # - reranker_enabled=false：完全禁用（默认），不下载、不调 API；/rerank 走原 score 排序；/health 显示 disabled
    # - reranker_enabled=true + RERANKER_PROVIDER=jina：Jina 在线 API（推荐，免费 1M token/月，需注册 https://jina.ai）
    # - reranker_enabled=true + RERANKER_PROVIDER=llm：用主系统 LLM 网关给候选打分（不下载、复用 qwen3.6-flash）
    # - reranker_enabled=true + RERANKER_PROVIDER=local：下载本地 BAAI/bge-reranker-v2-m3（~568MB，论文实验场景）
    reranker_enabled: bool = Field(default=False, alias="RERANKER_ENABLED")
    reranker_provider: str = Field(default="local", alias="RERANKER_PROVIDER")  # local | llm | jina | flashrank
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    # local 兜底：失败时按原 score 排序（不依赖额外服务）
    # llm provider 配置（默认复用 HyDE/主系统 LLM）
    reranker_llm_base_url: str | None = Field(default=None, alias="RERANKER_LLM_BASE_URL")
    reranker_llm_api_key: str | None = Field(default=None, alias="RERANKER_LLM_API_KEY")
    reranker_llm_model: str | None = Field(default=None, alias="RERANKER_LLM_MODEL")
    # jina provider 配置（https://jina.ai 注册免费 1M token/月）
    reranker_jina_api_key: str | None = Field(default=None, alias="RERANKER_JINA_API_KEY")
    reranker_jina_model: str = Field(default="jina-reranker-v2-base-multilingual", alias="RERANKER_JINA_MODEL")
    reranker_jina_base_url: str = Field(default="https://api.jina.ai/v1/rerank", alias="RERANKER_JINA_BASE_URL")
    # FlashRank provider 配置（推荐 2核4G 部署，CPU ONNX 推理，~18-120MB）
    # 模型选项（按大小/精度排序）：
    #   - ms-marco-TinyBERT-L-2-v2     ~18MB  最快最省，适合工单类短文本
    #   - ms-marco-MiniLM-L-12-v2      ~78MB  平衡
    #   - rank-T5-flan                 ~120MB 默认，精度最高
    reranker_flashrank_model: str = Field(default="rank-T5-flan", alias="RERANKER_FLASHRANK_MODEL")
    reranker_flashrank_cache_dir: str = Field(default="data/flashrank_cache", alias="RERANKER_FLASHRANK_CACHE_DIR")

    # MinerU 云端 PDF 解析（毕设推荐启用：论文价值高，借鉴 airQA 项目接入方式）
    # 留空则降级为 PyMuPDF 启发式解析（离线 fallback）
    mineru_api_token: str | None = Field(default=None, alias="MINERU_API_TOKEN")
    mineru_base_url: str = Field(default="https://mineru.net/api/v4", alias="MINERU_BASE_URL")
    mineru_model_version: str = Field(default="vlm", alias="MINERU_MODEL_VERSION")
    mineru_timeout: float = Field(default=600.0, alias="MINERU_TIMEOUT")
    mineru_poll_interval: float = Field(default=3.0, alias="MINERU_POLL_INTERVAL")

    # 公式 LaTeX 深度规范化（sympy / latex2sympy2，懒加载；未装库自动降级）
    formula_validation_enabled: bool = Field(default=False, alias="FORMULA_VALIDATION_ENABLED")

    # 检索侧
    normalize_scores_before_fusion: bool = Field(default=True, alias="NORMALIZE_SCORES_BEFORE_FUSION")
    diversity_penalty: float = Field(default=0.1, alias="DIVERSITY_PENALTY")  # 0=关闭；0.1 默认
    diversity_floor: float = Field(default=0.5, alias="DIVERSITY_FLOOR")  # 同文档衰减下限
    dedup_jaccard_threshold: float = Field(default=0.7, alias="DEDUP_JACCARD_THRESHOLD")  # 1=禁用

    default_chunk_size: int = Field(default=500, alias="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=50, alias="DEFAULT_CHUNK_OVERLAP")
    default_top_k: int = Field(default=10, alias="DEFAULT_TOP_K")

    rrf_k: int = Field(default=60, alias="RRF_K")
    rrf_vector_weight: float = Field(default=0.7, alias="RRF_VECTOR_WEIGHT")
    rrf_sparse_weight: float = Field(default=0.3, alias="RRF_SPARSE_WEIGHT")

    http_timeout: int = Field(default=30, alias="HTTP_TIMEOUT")

    score_threshold: float = Field(default=0.3, alias="SCORE_THRESHOLD")
    metadata_db_path: str = Field(default="data/rag_metadata.db", alias="METADATA_DB_PATH")

    # 日志：text 开发友好 / json 生产 ELK 友好；文件留空则只写 stdout
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="text", alias="LOG_FORMAT")  # text | json
    log_file_path: str | None = Field(default=None, alias="LOG_FILE_PATH")  # 留空=不写文件
    log_rotation: str = Field(default="20 MB", alias="LOG_ROTATION")
    log_retention: str = Field(default="14 days", alias="LOG_RETENTION")
    log_backtrace: bool = Field(default=True, alias="LOG_BACKTRACE")
    slow_request_ms: float = Field(default=2000.0, alias="SLOW_REQUEST_MS")  # 慢请求阈值

    # HyDE LLM（OpenAI 兼容；默认未配置）
    hyde_enabled_by_default: bool = Field(default=False, alias="HYDE_ENABLED_BY_DEFAULT")
    hyde_llm_base_url: str | None = Field(default=None, alias="HYDE_LLM_BASE_URL")
    hyde_llm_api_key: str | None = Field(default=None, alias="HYDE_LLM_API_KEY")
    hyde_llm_model: str = Field(default="gpt-4o-mini", alias="HYDE_LLM_MODEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
