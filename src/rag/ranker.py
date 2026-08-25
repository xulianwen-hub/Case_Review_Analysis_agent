"""
阶段三：并行多路召回 + 阶段四：RRF 融合粗排 + 阶段五：Cross-Encoder 精排

管线：
  候选 Query → 并行 Dense(BGE→Milvus) + Sparse(BM25→SQLite)
  → RRF 融合 + 去重 + 截断 → CE 精排 → Top-K

HyDE 独立闭环：CE Top-1 < 0.4 时触发二次检索
"""
import os
import re
import json
import time
import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

import numpy as np

from ..core.config import (
    MILVUS_DB, SQLITE_DB, MODELS_DIR, COLLECTION_NAME,
    VECTOR_RECALL_TOP_K, BM25_RECALL_TOP_K, DEFAULT_TOP_K,
    MAX_CONCURRENT_EMBED, VECTOR_CACHE_ENABLED, VECTOR_CACHE_TTL_SEC,
    RETRIEVAL_CACHE_ENABLED, RETRIEVAL_CACHE_TTL_SEC,
    CANDIDATE_POOL_MAX_ABSOLUTE, CANDIDATE_POOL_MAX_RATIO,
    RRF_K, RRF_FUSION_TOP_K,
    CE_ENABLED, CE_MODEL_NAME, CE_TOP_K_INPUT, CE_TOP_K_OUTPUT,
    CE_BATCH_SIZE, CE_HYDE_THRESHOLD, CE_MAX_RETRY_ROUNDS,
    DB_SOURCE_CONFIGS,
)
from ..core.logger import logger
from .router import QueryVariant


# ============================================================
# 数据类
# ============================================================

@dataclass
class RetrievedDoc:
    """检索到的文档"""
    chunk_id: str
    chunk_type: str       # child / parent
    text: str
    law_name: str
    chapter: str
    article: str
    parent_key: str
    source: str           # dense / sparse / fused
    source_query: str     # 来自哪个候选 Query
    score: float          # 归一化后分数
    rank: int = 0         # 在该召回链路中的排名
    source_db: str = ""   # 来自哪个数据库: laws / cases
    title: str = ""       # 案例标题（cases 库专用）


@dataclass 
class RankResult:
    """排序后的文档"""
    chunk_id: str
    text: str
    law_name: str
    chapter: str
    article: str
    parent_key: str
    score: float          # RRF 或 CE 分数
    rank: int
    source: str
    source_db: str = ""   # 来自哪个数据库
    title: str = ""       # 案例标题

# ============================================================
# 检索缓存（线程安全）
# ============================================================

class RetrievalCache:
    """两阶段检索缓存"""

    def __init__(self):
        self._lock = threading.Lock()
        # L1: Query → Embedding 缓存
        self._embed_cache: Dict[str, Tuple[float, List[List[float]]]] = {}
        # L2: Query → 检索结果缓存
        self._result_cache: Dict[str, Tuple[float, List[Dict]]] = {}

    def get_embedding(self, query: str) -> Optional[List[List[float]]]:
        if not VECTOR_CACHE_ENABLED:
            return None
        with self._lock:
            entry = self._embed_cache.get(query)
            if entry and time.time() - entry[0] < VECTOR_CACHE_TTL_SEC:
                return entry[1]
            if entry:
                del self._embed_cache[query]
        return None

    def set_embedding(self, query: str, embedding: List[List[float]]):
        if not VECTOR_CACHE_ENABLED:
            return
        with self._lock:
            self._embed_cache[query] = (time.time(), embedding)

    def get_results(self, query: str) -> Optional[List[Dict]]:
        if not RETRIEVAL_CACHE_ENABLED:
            return None
        with self._lock:
            entry = self._result_cache.get(query)
            if entry and time.time() - entry[0] < RETRIEVAL_CACHE_TTL_SEC:
                return entry[1]
            if entry:
                del self._result_cache[query]
        return None

    def set_results(self, query: str, results: List[Dict]):
        if not RETRIEVAL_CACHE_ENABLED:
            return
        with self._lock:
            self._result_cache[query] = (time.time(), results)

    def invalidate_all(self):
        """新法条入库时清空全量缓存"""
        with self._lock:
            self._embed_cache.clear()
            self._result_cache.clear()
            logger.info("[缓存] 已清空全部检索缓存（新法条入库触发）")


# 全局缓存单例
_cache = RetrievalCache()

# Embedding 并发控制
_DENSE_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_EMBED)


# ============================================================
# DBSource：单个数据源封装（Milvus + SQLite + 权重）
# ============================================================

class DBSource:
    """
    封装一个检索数据源（法条 / 案例 / 裁判文书 / 法规）

    每个数据源：
      - 有自己的 Milvus 向量库 和 SQLite 文本库
      - 支持 Dense(语义) 和 BM25(关键词) 独立检索
      - 两种结果保持独立排名，后续由 RRF 统一融合
    """

    def __init__(self, name: str, milvus_db: str, sqlite_db: str,
                 collection: str, db_type: str, model):
        self.name = name
        self.milvus_db_path = milvus_db
        self.sqlite_db_path = sqlite_db
        self.collection = collection
        self.db_type = db_type  # "laws" or "cases"
        self._model = model
        self._milvus_client = None
        self._sqlite_conn = None

    def _init_milvus(self):
        if self._milvus_client is None:
            from pymilvus import MilvusClient
            self._milvus_client = MilvusClient(uri=self.milvus_db_path)

    def _init_sqlite(self):
        if self._sqlite_conn is None:
            self._sqlite_conn = sqlite3.connect(self.sqlite_db_path)
            self._sqlite_conn.row_factory = sqlite3.Row

    def vector_search(self, query: str, top_k: int = None) -> List[RetrievedDoc]:
        if top_k is None:
            top_k = VECTOR_RECALL_TOP_K

        self._init_milvus()

        with _DENSE_SEMAPHORE:
            query_vec = self._model.encode(
                [query], normalize_embeddings=True
            ).tolist()

            try:
                results = self._milvus_client.search(
                    collection_name=self.collection,
                    data=query_vec,
                    anns_field="vector",
                    search_params={"metric_type": "IP", "params": {"nprobe": 8}},
                    limit=top_k,
                    output_fields=["id"],
                )
            except Exception:
                self._milvus_client.load_collection(self.collection)
                results = self._milvus_client.search(
                    collection_name=self.collection,
                    data=query_vec,
                    anns_field="vector",
                    search_params={"metric_type": "IP", "params": {"nprobe": 8}},
                    limit=top_k,
                    output_fields=["id"],
                )

        docs = []
        for i, hits in enumerate(results):
            for k, hit in enumerate(hits):
                docs.append(RetrievedDoc(
                    chunk_id=hit["id"],
                    chunk_type="child",
                    text="",
                    law_name="",
                    chapter="",
                    article="",
                    parent_key="",
                    source="dense",
                    source_query=query,
                    score=hit["distance"],
                    rank=k + 1,
                    source_db=self.name,
                ))
        return docs

    def bm25_search(self, query: str, top_k: int = None) -> List[RetrievedDoc]:
        if top_k is None:
            top_k = BM25_RECALL_TOP_K

        self._init_sqlite()

        escaped_query = re.sub(r'[|*"()\[\]{}<>~^:;,!，。！？、；：（）【】「」]+', ' ', query)
        words = escaped_query.split()

        # 中文查询：使用 2-字滑动窗口分词（适配 FTS5 unicode61 分词器）
        has_chinese = any(re.search(r'[\u4e00-\u9fff]', w) for w in words[:3]) if words else True
        if has_chinese:
            chinese_only = re.sub(r'[^\u4e00-\u9fff]', '', query)
            chinese_words = set()
            for i in range(len(chinese_only) - 1):
                chinese_words.add(chinese_only[i:i+2])
            if chinese_words:
                words = sorted(chinese_words)[:10]

        if not words:
            return []

        match_query = " OR ".join(f'"{w}"' for w in words[:10])

        cur = self._sqlite_conn.cursor()
        try:
            cur.execute(
                """SELECT c.id, rank
                   FROM chunks_fts f
                   JOIN chunks c ON f.rowid = c.rowid
                   WHERE chunks_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (match_query, top_k),
            )
        except Exception:
            cur.execute(
                """SELECT c.id, rank
                   FROM chunks_fts f
                   JOIN chunks c ON f.rowid = c.rowid
                   WHERE chunks_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                ('"{}"'.format(' '.join(words[:3])), top_k),
            )

        docs = []
        for k, row in enumerate(cur.fetchall()):
            score = 1.0 / (1.0 + float(row["rank"]))
            docs.append(RetrievedDoc(
                chunk_id=row["id"],
                chunk_type="child",
                text="",
                law_name="",
                chapter="",
                article="",
                parent_key="",
                source="sparse",
                source_query=query,
                score=score,
                rank=k + 1,
                source_db=self.name,
            ))
        return docs

    def enrich_docs(self, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        """从 SQLite 填充完整文本信息"""
        if not docs:
            return docs
        self._init_sqlite()
        cur = self._sqlite_conn.cursor()

        if self.db_type == "laws":
            for doc in docs:
                cur.execute(
                    "SELECT chunk_type, text, law_name, chapter, article, parent_key "
                    "FROM chunks WHERE id = ?",
                    (doc.chunk_id,),
                )
                row = cur.fetchone()
                if row:
                    doc.chunk_type = row["chunk_type"]
                    doc.text = row["text"]
                    doc.law_name = row["law_name"]
                    doc.chapter = row["chapter"]
                    doc.article = row["article"]
                    doc.parent_key = row["parent_key"]
        else:
            for doc in docs:
                cur.execute(
                    "SELECT chunk_type, text, title, publish_source, data_time, parent_key "
                    "FROM chunks WHERE id = ?",
                    (doc.chunk_id,),
                )
                row = cur.fetchone()
                if row:
                    doc.chunk_type = row["chunk_type"]
                    doc.text = row["text"]
                    doc.title = row["title"] or ""
                    doc.law_name = row["title"] or ""  # 映射：title → law_name（兼容显示）
                    doc.chapter = row["publish_source"] or ""
                    doc.article = row["data_time"] or ""
                    doc.parent_key = row["parent_key"]
        return docs

    def get_parent_text(self, parent_key: str) -> str:
        """通过 parent_key 获取父块完整文本"""
        self._init_sqlite()
        cur = self._sqlite_conn.cursor()
        cur.execute(
            "SELECT text FROM chunks WHERE parent_key = ? AND chunk_type = 'parent'",
            (parent_key,),
        )
        row = cur.fetchone()
        return row["text"] if row else ""

    def close(self):
        if self._sqlite_conn:
            self._sqlite_conn.close()
        if self._milvus_client:
            self._milvus_client.close()


# ============================================================
# 阶段三：并行多路召回
# ============================================================

class MultiPathRetriever:
    """
    阶段三：并行多路召回（多数据库支持）

    架构：
      - 持有多个 DBSource（法条 + 案例/裁判文书/法规）
      - 每个 DBSource 内部 Dense 和 BM25 独立检索，不做加权融合
      - 跨数据源结果通过标准 RRF（Σ 1/(k+rank)）合并

    基础设施：
      - 检索缓存（L1: Embedding 缓存, L2: 结果缓存）
      - Semaphore 并发控制
    """

    _embed_semaphore = threading.Semaphore(MAX_CONCURRENT_EMBED)  # 保留兼容，实际由 _DENSE_SEMAPHORE 控制

    def __init__(self, db_sources: List[DBSource] = None):
        self._model = None
        self._sources: List[DBSource] = []

        if db_sources is not None:
            self._sources = list(db_sources)
        else:
            # 兼容旧代码：默认仅加载 laws
            self._init_model()
            self._sources = [
                DBSource(
                    name="laws",
                    milvus_db=MILVUS_DB,
                    sqlite_db=SQLITE_DB,
                    collection=COLLECTION_NAME,
                    db_type="laws",
                    model=self._model,
                )
            ]

        logger.info(
            f"[多路召回] 数据源: {[s.name for s in self._sources]}"
        )

    def _init_model(self):
        if self._model is None:
            model_path = os.path.join(
                MODELS_DIR, "models", "BAAI--bge-base-zh-v1.5", "snapshots", "master"
            )
            if not os.path.exists(model_path):
                from modelscope import snapshot_download
                model_path = snapshot_download("BAAI/bge-base-zh-v1.5", cache_dir=MODELS_DIR)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_path)

    @classmethod
    def from_config(cls) -> "MultiPathRetriever":
        """从 DB_SOURCE_CONFIGS 配置创建多数据源检索器"""
        retriever = cls(db_sources=[])
        retriever._init_model()
        for src_cfg in DB_SOURCE_CONFIGS:
            source = DBSource(
                name=src_cfg["name"],
                milvus_db=src_cfg["milvus_db"],
                sqlite_db=src_cfg["sqlite_db"],
                collection=src_cfg["collection"],
                db_type=src_cfg["db_type"],
                model=retriever._model,
            )
            retriever._sources.append(source)
        logger.info(
            f"[多路召回] 从配置加载 {len(retriever._sources)} 个数据源: "
            f"{[s.name for s in retriever._sources]}"
        )
        return retriever

    def retrieve_all(self, queries: List[QueryVariant],
                     original_query: str = None,
                     top_k_per_source: int = None) -> Tuple[List[RetrievedDoc], List[RetrievedDoc]]:
        """
        多数据源多 Query 检索（Dense 和 BM25 独立返回）

        流程：
          1. 对每个候选 Query，在每个数据源上分别做 Dense 和 BM25 检索
          2. Dense 和 BM25 各自保持独立排名，不做加权融合
          3. 后续由 RRFMerger 做标准 RRF 粗排（不加权，Σ 1/(k+rank)）

        Returns:
            (dense_docs, bm25_docs): 两个独立列表，已填充文本
        """
        if top_k_per_source is None:
            top_k_dense = VECTOR_RECALL_TOP_K
            top_k_bm25 = BM25_RECALL_TOP_K
        else:
            top_k_dense = top_k_per_source
            top_k_bm25 = top_k_per_source

        if original_query is None:
            original_query = queries[0].text if queries else ""

        all_dense: List[RetrievedDoc] = []
        all_bm25: List[RetrievedDoc] = []
        seen_ids: Set[str] = set()

        for qv in queries:
            for source in self._sources:
                try:
                    dense_docs = source.vector_search(qv.text, top_k=top_k_dense)
                    for d in dense_docs:
                        d.source_query = qv.text
                    all_dense.extend(dense_docs)
                except Exception as e:
                    logger.error(
                        f"[Dense] 数据源 '{source.name}' Query '{qv.text[:50]}...' 失败: {e}"
                    )

                try:
                    bm25_docs = source.bm25_search(qv.text, top_k=top_k_bm25)
                    for d in bm25_docs:
                        d.source_query = qv.text
                    all_bm25.extend(bm25_docs)
                except Exception as e:
                    logger.error(
                        f"[BM25] 数据源 '{source.name}' Query '{qv.text[:50]}...' 失败: {e}"
                    )

        # 去重：同一 chunk_id 在 Dense 和 BM25 中可能重复
        # 不去重——RRF 需要它们各自独立的排名信息
        # 但需要填充文本
        all_docs = all_dense + all_bm25
        self._enrich_all(all_docs)

        logger.info(
            f"[多路召回] {len(queries)} Query × {len(self._sources)} 数据源 "
            f"→ Dense={len(all_dense)} | BM25={len(all_bm25)}"
        )
        return all_dense, all_bm25

    def _enrich_all(self, docs: List[RetrievedDoc]):
        """按数据源分组填充文本"""
        by_source: Dict[str, List[RetrievedDoc]] = {}
        for d in docs:
            by_source.setdefault(d.source_db, []).append(d)
        for source in self._sources:
            if source.name in by_source:
                source.enrich_docs(by_source[source.name])

    def get_parent_text(self, chunk_id: str, source_db: str) -> str:
        """通过 chunk_id 和来源数据库获取父块文本"""
        for source in self._sources:
            if source.name == source_db:
                return source.get_parent_text(chunk_id)
        return ""

    def close(self):
        for source in self._sources:
            source.close()


# ============================================================
# 阶段四：RRF 融合与粗排
# ============================================================

class RRFMerger:
    """
    阶段四：RRF 融合与粗排

    公式：RRF(d) = Σ(1 / (k + rank_i(d)))
    对所有召回源（Dense + Sparse + 各数据源）的排名合并

    多数据源模式：使用 SUM 聚合（同一文档出现在多个库中 = 强信号）
    单数据源模式：使用 MAX 聚合（兼容旧行为）
    """

    def __init__(self, k: int = None):
        self.k = k or RRF_K

    def merge(self, dense_results: List[RetrievedDoc],
              sparse_results: List[RetrievedDoc],
              top_k: int = None) -> List[RankResult]:
        """单数据源 RRF 合并（向后兼容）"""
        if top_k is None:
            top_k = RRF_FUSION_TOP_K

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, RetrievedDoc] = {}

        all_docs = dense_results + sparse_results

        for doc in all_docs:
            rrf = 1.0 / (self.k + doc.rank)
            cid = doc.chunk_id
            if cid not in rrf_scores or rrf > rrf_scores[cid]:
                rrf_scores[cid] = rrf
                doc_map[cid] = doc

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        results = []
        for rank, cid in enumerate(sorted_ids, 1):
            doc = doc_map[cid]
            results.append(RankResult(
                chunk_id=cid,
                text=doc.text,
                law_name=doc.law_name,
                chapter=doc.chapter,
                article=doc.article,
                parent_key=doc.parent_key,
                score=rrf_scores[cid],
                rank=rank,
                source=doc.source,
                source_db=doc.source_db,
                title=doc.title,
            ))

        return results

    def merge_multi_source(self, dense_docs: List[RetrievedDoc],
                           bm25_docs: List[RetrievedDoc],
                           top_k: int = None) -> List[RankResult]:
        """
        标准 RRF 多路融合（Dense + BM25，不加权）

        公式：RRF(d) = Σ(1 / (k + rank_i))

        同一文档出现在多个召回源中 → RRF 分数累加（SUM 聚合）

        Args:
            dense_docs: 所有 Dense 检索结果（保持独立排名）
            bm25_docs: 所有 BM25 检索结果（保持独立排名）
            top_k: 截断数量

        Returns:
            RankResult 列表
        """
        if top_k is None:
            top_k = RRF_FUSION_TOP_K

        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, RetrievedDoc] = {}

        for doc in list(dense_docs) + list(bm25_docs):
            rrf = 1.0 / (self.k + doc.rank)
            cid = doc.chunk_id
            if cid not in rrf_scores:
                rrf_scores[cid] = 0.0
                doc_map[cid] = doc
            rrf_scores[cid] += rrf

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        results = []
        for rank, cid in enumerate(sorted_ids, 1):
            doc = doc_map[cid]
            results.append(RankResult(
                chunk_id=cid,
                text=doc.text,
                law_name=doc.law_name,
                chapter=doc.chapter,
                article=doc.article,
                parent_key=doc.parent_key,
                score=rrf_scores[cid],
                rank=rank,
                source=doc.source,
                source_db=doc.source_db,
                title=doc.title,
            ))

        dense_by_src = {}
        bm25_by_src = {}
        for d in dense_docs:
            dense_by_src[d.source_db] = dense_by_src.get(d.source_db, 0) + 1
        for d in bm25_docs:
            bm25_by_src[d.source_db] = bm25_by_src.get(d.source_db, 0) + 1

        logger.info(
            f"[RRF] Dense({dense_by_src}) + BM25({bm25_by_src}) "
            f"→ 融合后 {len(results)} 条"
        )

        return results


# ============================================================
# 阶段五：Cross-Encoder 深度精排
# ============================================================

class CrossEncoderRanker:
    """
    阶段五：Cross-Encoder 深度精排

    对 RRF 结果 Top-N 做 Cross-Encoder 成对打分 → 重新排序 → 截断 Top-K

    CE 模型：bge-reranker-base（可配置）
    """

    def __init__(self):
        self._ce_model = None
        self._initialized = False

    def _init_model(self):
        if self._initialized:
            return
        try:
            from sentence_transformers import CrossEncoder
            import os as _os

            # 优先本地 modelscope 快照（HuggingFace 大陆访问超时），和 Dense 模型加载策略一致
            # snapshot 目录示例：MODELS_DIR/BAAI--bge-reranker-base/snapshots/master/
            model_dir = _os.path.join(
                MODELS_DIR, "models",
                CE_MODEL_NAME.replace("/", "--"),
                "snapshots", "master"
            )
            if not _os.path.exists(model_dir) or not _os.listdir(model_dir):
                logger.info(f"[CE] 本地快照不存在 ({model_dir})，调用 modelscope 下载 {CE_MODEL_NAME}...")
                from modelscope import snapshot_download as _snapshot_download
                model_dir = _snapshot_download(CE_MODEL_NAME, cache_dir=MODELS_DIR)

            self._ce_model = CrossEncoder(model_dir)
            self._initialized = True
            logger.info(f"[CE] 模型加载成功: {CE_MODEL_NAME} (本地路径={model_dir})")
        except Exception as e:
            logger.warning(f"[CE] 模型加载失败，CE 精排降级: {e}")
            self._ce_model = None
            self._initialized = True

    def rerank(self, query: str, candidates: List[RankResult],
               top_k: int = None, input_top_n: int = None) -> List[RankResult]:
        """
        CE 精排

        Args:
            query: 原始查询
            candidates: RRF 粗排结果
            top_k: 精排后保留多少条
            input_top_n: 对前多少条做精排

        Returns:
            重新排序后的 Top-K
        """
        if top_k is None:
            top_k = CE_TOP_K_OUTPUT
        if input_top_n is None:
            input_top_n = CE_TOP_K_INPUT

        if not CE_ENABLED or not self._initialized:
            logger.debug("[CE] 未启用或模型未加载，跳过精排，直接返回 RRF Top-K")
            for i, c in enumerate(candidates[:top_k]):
                c.rank = i + 1
            return candidates[:top_k]

        self._init_model()

        if self._ce_model is None:
            # 降级：直接返回 RRF 结果
            return candidates[:top_k]

        # 取前 input_top_n 做精排
        to_rerank = candidates[:min(input_top_n, len(candidates))]

        # 成对打分
        pairs = [(query, doc.text) for doc in to_rerank]
        scores = self._ce_model.predict(pairs, batch_size=CE_BATCH_SIZE)

        # 重新排序
        for doc, score in zip(to_rerank, scores):
            doc.score = float(score)

        to_rerank.sort(key=lambda x: x.score, reverse=True)

        for i, doc in enumerate(to_rerank[:top_k]):
            doc.rank = i + 1

        return to_rerank[:top_k]

    def get_top_score(self, results: List[RankResult]) -> float:
        """获取精排后 Top-1 分数"""
        return results[0].score if results else 0.0


# ============================================================
# 管线封装
# ============================================================

def get_cache():
    """获取全局缓存实例"""
    return _cache