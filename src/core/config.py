"""
项目统一配置
- 所有路径从项目根目录推导，不硬编码绝对路径
- 所有可调参数集中管理（含六阶段 RAG 检索参数）
"""
from pathlib import Path

# ============================================================
# 项目根目录（从本文件位置向上推导：src/core/config.py → 项目根）
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ============================================================
# 数据路径
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LAWS_DIR = PROCESSED_DIR / "laws"
VECTORS_DIR = PROCESSED_DIR / "vectors"
MODELS_DIR = PROJECT_ROOT / "models"

# 数据库文件（向后兼容，保留旧配置）
MILVUS_DB = str(VECTORS_DIR / "laws_milvus.db")
SQLITE_DB = str(PROCESSED_DIR / "chunks.db")

# Milvus 集合名
COLLECTION_NAME = "labor_laws"

# ============================================================
# Multi-DB 检索源配置
# 每个数据源独立配置，Dense(语义) 和 BM25(关键词) 保持独立排名
# 跨数据源结果通过标准 RRF（Σ 1/(k+rank)）统一融合，不做加权
#
# 如需调整某数据源的影响力，可通过调整召回数量（top_k）间接实现
# ============================================================
DB_SOURCE_CONFIGS = [
    {
        "name": "laws",
        "milvus_db": str(VECTORS_DIR / "laws_milvus.db"),
        "sqlite_db": str(PROCESSED_DIR / "chunks.db"),
        "collection": "labor_laws",
        "db_type": "laws",
    },
    {
        "name": "non_laws",
        "milvus_db": str(VECTORS_DIR / "cases_milvus.db"),
        "sqlite_db": str(PROCESSED_DIR / "chunks_cases.db"),
        "collection": "labor_cases",
        "db_type": "cases",
    },
]

# ============================================================
# 六阶段 RAG 检索通用参数
# ============================================================
DEFAULT_TOP_K = 15
MAX_RETRIEVAL_TEXTS = 10

# ============================================================
# 输入校验
# ============================================================
MIN_INPUT_LENGTH = 10
MAX_INPUT_LENGTH = 2000
BANNED_PATTERNS = []

# ============================================================
# Token 估算
# ============================================================
CHINESE_CHARS_PER_TOKEN = 1.5
ENGLISH_CHARS_PER_TOKEN = 4.0
MAX_CONTEXT_TOKENS = 30000

# ============================================================
# LLM 调用
# ============================================================
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 8192

# ============================================================
# 六阶段 RAG 检索参数
# ============================================================

# --- 阶段一：前置路由 ---
# 规则引擎：法律关键词（正则匹配，<1ms）
LEGAL_KEYWORDS_PATTERN = (
    r"劳动合同|工资|开除|加班|工伤|社保|辞退|赔偿|仲裁|"
    r"经济补偿|竞业限制|试用期|年假|产假|病假|离职|"
    r"解除|终止|拖欠|克扣|调岗|降薪|裁员|职业病"
)

# 规则引擎：问候语/闲聊（直接返回，不走 RAG）
GREETING_PATTERNS = [
    r"^(你好|您好|hi|hello|hey)[\s!！。.]*$",
    r"^(谢谢|感谢|thanks|thank|多谢)[\s!！。.]*$",
    r"^(再见|拜拜|bye|goodbye|回见)[\s!！。.]*$",
    r"^(在吗|在不在|有人吗|上线了没)[\s!！。.]*$",
]

GREETING_REPLIES = {
    "greeting": "您好！我是劳动纠纷智能分析助手，请输入您遇到的劳动纠纷问题，我会为您提供专业的法律分析。",
    "thanks": "不客气！有任何劳动法相关问题随时问我。",
    "bye": "再见！祝您维权顺利。如有需要随时回来。",
    "presence": "在的！请直接描述您遇到的劳动纠纷问题，我会尽力帮您分析。",
}

# --- 阶段二：查询改写 ---
# 复杂度阈值（来自阶段一路由）
QUERY_REWRITE_SIMPLE_THRESHOLD = 0.5   # complexity < 0.5 → Simple（不改写）
QUERY_REWRITE_MEDIUM_THRESHOLD = 0.8   # 0.5-0.8 → Medium（Multi-Query）
                                        # > 0.8 → Complex（子问题分解）

# --- 阶段三：多路召回 ---
VECTOR_RECALL_TOP_K = 30           # 每个 Query 的向量召回数
BM25_RECALL_TOP_K = 30             # BM25 召回数
MAX_CONCURRENT_EMBED = 4           # 并行 embed 最大并发数（Semaphore）

# 检索缓存
VECTOR_CACHE_ENABLED = True        # 是否启用向量缓存
VECTOR_CACHE_TTL_SEC = 600         # Query→Embedding 缓存 10 分钟
RETRIEVAL_CACHE_ENABLED = True     # 是否启用检索结果缓存
RETRIEVAL_CACHE_TTL_SEC = 86400    # 检索结果缓存 24 小时

# 候选池上限自适应：min(绝对上限, 总 chunk 数 × 比例)
CANDIDATE_POOL_MAX_ABSOLUTE = 150  # 绝对上限
CANDIDATE_POOL_MAX_RATIO = 0.30    # 总 chunk 数比例

# --- 阶段四：RRF 融合 ---
RRF_K = 60                         # RRF 衰减因子（可配置: 20/60/100）
RRF_FUSION_TOP_K = 50              # RRF 融合后截断 Top-50

# --- 阶段五：Cross-Encoder 精排 ---
CE_ENABLED = True                  # CE 精排开关（数据量>=500条时开启，当前法条+案例=5400+条）
CE_MODEL_NAME = "BAAI/bge-reranker-base"  # CE 模型
CE_TOP_K_INPUT = 30                # CE 精排输入 Top-30
CE_TOP_K_OUTPUT = 10               # CE 精排输出 Top-K（可配置，方便实验）
CE_BATCH_SIZE = 8                  # CE 批量推理大小
CE_HYDE_THRESHOLD = 0.4            # CE Top-1 < 此值 → 触发 HyDE 二次检索
CE_MAX_RETRY_ROUNDS = 1            # 最多 1 轮补充检索（HyDE 或覆盖度）

# --- 阶段六：置信度自检 ---
CONFIDENCE_HIGH = 0.7              # 高置信度阈值
CONFIDENCE_MEDIUM = 0.4            # 中等置信度阈值（低于此值触发反问）
CONFIDENCE_GAP_THRESHOLD = 0.3     # Top-1 与 Top-3 分数差距阈值
CONFIDENCE_GAP_CAUTIOUS = 0.1      # 多个答案皆有可能的阈值

# 覆盖度检查
COVERAGE_LIGHTWEIGHT_ENABLED = True  # 轻量覆盖度检查（关键词匹配）
COVERAGE_SUPPLEMENT_RETRIEVAL_MAX = 1  # 补充检索最多 1 轮