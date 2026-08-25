"""
检索效果评估脚本

功能：
  1. 加载 test_queries.json 中的测试查询
  2. 对每个查询执行向量检索（Dense + Sparse 可选）
  3. 计算 Recall@K、MRR、NDCG@K 等指标
  4. 输出评估报告

用法：
  python scripts/eval_retrieval.py                    # 评估法条库
  python scripts/eval_retrieval.py --db cases          # 评估案例库
  python scripts/eval_retrieval.py --db all            # 评估全部
  python scripts/eval_retrieval.py --db laws --top-k 5 # 指定 Top-K
"""
import json
import os
import sys
import time
from pathlib import Path

# Windows 控制台 GBK 编码兼容
if sys.stdout.encoding and sys.stdout.encoding.lower() in ("gbk", "gb2312"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# =============================================
# 配置
# =============================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_QUERIES_FILE = PROJECT_ROOT / "data" / "eval" / "test_queries.json"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "eval"

# 数据库配置
DB_CONFIGS = {
    "laws": {
        "milvus_db": str(PROJECT_ROOT / "data" / "processed" / "vectors" / "laws_milvus.db"),
        "collection": "labor_laws",
        "sqlite_db": str(PROJECT_ROOT / "data" / "processed" / "chunks.db"),
        "child_chunks": str(PROJECT_ROOT / "data" / "processed" / "laws" / "child_chunks.json"),
        "parent_chunks": str(PROJECT_ROOT / "data" / "processed" / "laws" / "parent_chunks.json"),
    },
    "cases": {
        "milvus_db": str(PROJECT_ROOT / "data" / "processed" / "vectors" / "cases_milvus.db"),
        "collection": "labor_cases",
        "sqlite_db": str(PROJECT_ROOT / "data" / "processed" / "chunks_cases.db"),
        "child_chunks": str(PROJECT_ROOT / "data" / "processed" / "cases" / "child_chunks.json"),
        "parent_chunks": str(PROJECT_ROOT / "data" / "processed" / "cases" / "parent_chunks.json"),
    },
}

DEFAULT_TOP_K = 10
DEFAULT_METRIC = "IP"
RRF_K = 60  # RRF 融合参数（和 src/config.py 一致）


def build_bm25_query(query, max_terms=6):
    """构建 FTS5 安全查询：去掉特殊字符，提取关键词用 OR 连接"""
    import re as _re
    escaped = _re.sub(r'[|*"()\[\]{}<>~^:;,!，。！？、；：（）【】「」]+', ' ', query)
    words = escaped.split()
    if not words:
        chinese_words = _re.findall(r'[\u4e00-\u9fff]{2,8}', query)
        words = chinese_words[:max_terms]
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in words[:max_terms])


def bm25_search(db_path, query, top_k=20):
    """对指定 SQLite 执行 BM25 检索，返回 [(chunk_id, score), ...]"""
    import sqlite3 as _sq

    match_q = build_bm25_query(query)
    if not match_q:
        return []

    conn = _sq.connect(db_path)
    conn.row_factory = _sq.Row
    cur = conn.cursor()
    try:
        cur.execute(
            """SELECT c.id, rank FROM chunks_fts f
               JOIN chunks c ON f.rowid = c.rowid
               WHERE chunks_fts MATCH ?
               ORDER BY rank LIMIT ?""",
            (match_q, top_k),
        )
        hits = [(row["id"], 1.0 / (1.0 + float(row["rank"]))) for row in cur.fetchall()]
        conn.close()
        return hits
    except Exception:
        conn.close()
        return []


def rrf_merge(dense_ids, sparse_ids, top_k):
    """标准 RRF 融合：RRF(d) = Σ(1 / (k + rank_i(d)))"""
    rrf_scores = {}
    for rank, cid in enumerate(dense_ids, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in merged[:top_k]]


def load_model():
    """加载 BGE 模型"""
    from sentence_transformers import SentenceTransformer

    model_path = os.path.join(
        MODELS_DIR, "models", "BAAI--bge-base-zh-v1.5", "snapshots", "master"
    )
    if not os.path.exists(model_path):
        model_path = os.path.join(
            MODELS_DIR, "models", "BAAI--bge-small-zh-v1.5", "snapshots", "master"
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型未找到: {model_path}")

    print(f"[模型] 加载: {model_path}")
    model = SentenceTransformer(model_path)
    return model


def load_test_queries(db_filter="all"):
    """加载测试查询，按 db_filter 过滤"""
    with open(TEST_QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if db_filter == "all":
        return queries

    # 按 category 过滤
    category_map = {
        "laws": ["laws"],
        "cases": ["cases", "judgments", "regulations"],
    }
    allowed = category_map.get(db_filter, [db_filter])
    filtered = [q for q in queries if q.get("category", "") in allowed]
    return filtered


def evaluate(db_name, queries, top_k=DEFAULT_TOP_K, hybrid_mode=False):
    """对指定数据库执行检索评估"""
    from pymilvus import MilvusClient

    config = DB_CONFIGS[db_name]
    model = load_model()

    print(f"\n{'='*60}")
    print(f"评估数据库: {db_name}")
    print(f"  Milvus: {config['milvus_db']}")
    print(f"  Collection: {config['collection']}")
    print(f"  Top-K: {top_k}")
    print(f"  模式: {'混合(向量+BM25+RRF)' if hybrid_mode else '纯向量'}")
    print(f"{'='*60}")

    # 连接向量库
    mc = MilvusClient(uri=config["milvus_db"])
    mc.load_collection(config["collection"])
    stats = mc.get_collection_stats(config["collection"])
    total_docs = stats["row_count"]
    print(f"  总向量数: {total_docs}")

    # 统计
    recall_at_k = {k: 0.0 for k in [1, 3, 5, 10]}
    mrr_total = 0.0
    ndcg_total = 0.0
    evaluated = 0
    details = []

    for q in queries:
        query_text = q["query"]
        query_id = q["id"]
        relevant_ids = set(q.get("relevant_chunk_ids", []))

        if not relevant_ids:
            # 跳过无 ground truth 的查询（但仍执行检索供人工检查）
            details.append({
                "id": query_id,
                "query": query_text,
                "category": q.get("category", ""),
                "difficulty": q.get("difficulty", ""),
                "relevant_ids": [],
                "retrieved": [],
                "metrics": {"note": "无 ground truth，已跳过指标计算"},
                "error": None,
            })
            continue

        # 向量检索
        try:
            query_embedding = model.encode([query_text], normalize_embeddings=True)
            results = mc.search(
                collection_name=config["collection"],
                data=query_embedding.tolist(),
                anns_field="vector",
                search_params={"metric_type": DEFAULT_METRIC, "params": {"nprobe": 16}},
                limit=top_k,
                output_fields=["id", "text", "title", "law_name", "chapter", "article"],
            )
        except Exception as e:
            details.append({
                "id": query_id,
                "query": query_text,
                "category": q.get("category", ""),
                "difficulty": q.get("difficulty", ""),
                "relevant_ids": list(relevant_ids),
                "retrieved": [],
                "metrics": {},
                "error": str(e),
            })
            continue

        # 提取检索结果
        retrieved = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                retrieved.append({
                    "chunk_id": entity.get("id", ""),
                    "score": hit["distance"],
                    "title": entity.get("title", entity.get("law_name", "")),
                    "text_preview": (entity.get("text", "") or "")[:100],
                })

        retrieved_ids = [r["chunk_id"] for r in retrieved]

        # ========== 混合检索模式：向量 + BM25 → RRF 融合 ==========
        if hybrid_mode:
            sqlite_db = config.get("sqlite_db", "")
            if sqlite_db and os.path.exists(sqlite_db):
                sparse_hits = bm25_search(sqlite_db, query_text, top_k=top_k * 2)
                # 向量检索按距离降序
                dense_ids = [r["chunk_id"] for r in sorted(retrieved, key=lambda x: x["score"], reverse=True)]
                sparse_ids = [cid for cid, _ in sparse_hits]
                # RRF 融合
                merged_ids = rrf_merge(dense_ids, sparse_ids, top_k)
                # 保留检索信息并按 RRF 顺序重建
                info_by_id = {r["chunk_id"]: r for r in retrieved}
                bm25_score = {cid: s for cid, s in sparse_hits}
                retrieved = []
                for cid in merged_ids:
                    if cid in info_by_id:
                        item = dict(info_by_id[cid])
                        # 保留向量分数，混合模式下额外标注 bm25 命中
                        item["bm25_rank"] = sparse_ids.index(cid) + 1 if cid in bm25_score else None
                        retrieved.append(item)
                    elif cid in bm25_score:
                        retrieved.append({
                            "chunk_id": cid,
                            "score": bm25_score[cid],
                            "title": "(BM25)",
                            "text_preview": "",
                            "bm25_rank": sparse_ids.index(cid) + 1,
                        })
                retrieved_ids = [r["chunk_id"] for r in retrieved]

        # 计算 Recall@K
        query_recall = {}
        for k in recall_at_k:
            hits = set(retrieved_ids[:k])
            recall = len(hits & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0
            query_recall[f"recall@{k}"] = recall
            recall_at_k[k] += recall

        # 计算 MRR
        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                mrr = 1.0 / rank
                break
        mrr_total += mrr

        # 计算 NDCG@K（简化版：相关=1，不相关=0）
        dcg = 0.0
        idcg = 0.0
        import math
        for i, rid in enumerate(retrieved_ids[:top_k], start=1):
            rel = 1.0 if rid in relevant_ids else 0.0
            dcg += rel / math.log2(i + 1)
        for i in range(1, min(len(relevant_ids), top_k) + 1):
            idcg += 1.0 / math.log2(i + 1)
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_total += ndcg

        evaluated += 1

        details.append({
            "id": query_id,
            "query": query_text,
            "category": q.get("category", ""),
            "difficulty": q.get("difficulty", ""),
            "relevant_ids": list(relevant_ids),
            "retrieved": retrieved,
            "metrics": {
                **query_recall,
                "mrr": round(mrr, 4),
                "ndcg": round(ndcg, 4),
            },
            "error": None,
        })

        # 打印单条结果
        hits_str = "[HIT]" if retrieved_ids[0] in relevant_ids else "[MISS]"
        print(f"\n  [{query_id}] {hits_str} {q.get('difficulty','?')} | {query_text[:50]}...")
        print(f"    相关: {relevant_ids}")
        for r in retrieved[:3]:
            match = "*" if r["chunk_id"] in relevant_ids else " "
            print(f"    {match} Top-{retrieved.index(r)+1}: {r['chunk_id']} ({r['score']:.4f}) {r['title'][:40]}")

    mc.close()

    # 汇总
    if evaluated > 0:
        for k in recall_at_k:
            recall_at_k[k] = round(recall_at_k[k] / evaluated, 4)
        mrr = round(mrr_total / evaluated, 4)
        ndcg = round(ndcg_total / evaluated, 4)

    print(f"\n{'='*60}")
    print(f"评估汇总 ({db_name}, {evaluated} 条有标注的查询)")
    print(f"{'='*60}")
    print(f"  Recall@1:  {recall_at_k[1]:.4f}")
    print(f"  Recall@3:  {recall_at_k[3]:.4f}")
    print(f"  Recall@5:  {recall_at_k[5]:.4f}")
    print(f"  Recall@10: {recall_at_k[10]:.4f}")
    print(f"  MRR:       {mrr:.4f}")
    print(f"  NDCG@10:   {ndcg:.4f}")
    print(f"{'='*60}")

    # 保存详细结果
    output_file = OUTPUT_DIR / f"eval_results_{db_name}.json"
    summary = {
        "db": db_name,
        "top_k": top_k,
        "total_docs": total_docs,
        "evaluated": evaluated,
        "total_queries": len(queries),
        "metrics": {
            "recall@1": recall_at_k[1],
            "recall@3": recall_at_k[3],
            "recall@5": recall_at_k[5],
            "recall@10": recall_at_k[10],
            "mrr": mrr,
            "ndcg@10": ndcg,
        },
        "details": details,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_file}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="检索效果评估")
    parser.add_argument("--db", choices=["laws", "cases", "all"], default="laws",
                        help="评估目标数据库 (default: laws)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Top-K 检索数量 (default: {DEFAULT_TOP_K})")
    parser.add_argument("--mode", choices=["vector", "hybrid"], default="vector",
                        help="评估模式: vector=纯向量, hybrid=向量+BM25+RRF (default: vector)")
    parser.add_argument("--no-model", action="store_true",
                        help="跳过模型加载，仅统计测试集信息")
    args = parser.parse_args()
    hybrid_mode = (args.mode == "hybrid")

    print("=" * 60)
    print("检索效果评估工具")
    print(f"  检索模式: {args.mode}")
    print("=" * 60)

    # 加载测试集
    queries = load_test_queries(args.db)
    print(f"\n[测试集] 加载 {len(queries)} 条查询")
    by_category = {}
    for q in queries:
        cat = q.get("category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")
    has_gt = sum(1 for q in queries if q.get("relevant_chunk_ids"))
    print(f"  有标注: {has_gt}")
    print(f"  无标注(仅检索): {len(queries) - has_gt}")

    if args.no_model:
        print("\n跳过模型评估。")
        return

    if args.db == "all":
        for db_name in ["laws", "cases"]:
            if os.path.exists(DB_CONFIGS[db_name]["milvus_db"]):
                evaluate(db_name, queries, args.top_k, hybrid_mode=hybrid_mode)
            else:
                print(f"\n[跳过] 数据库不存在: {db_name}")
    else:
        if not os.path.exists(DB_CONFIGS[args.db]["milvus_db"]):
            print(f"\n[错误] 数据库不存在: {DB_CONFIGS[args.db]['milvus_db']}")
            print("请先运行 build_vector_db.py 或 build_case_db.py 构建向量库")
            sys.exit(1)
        evaluate(args.db, queries, args.top_k, hybrid_mode=hybrid_mode)


if __name__ == "__main__":
    main()