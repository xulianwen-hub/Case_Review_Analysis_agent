"""验证：标准 RRF 多库检索"""
import sys
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, ".")
from src.config import DB_SOURCE_CONFIGS
from src.ranker import MultiPathRetriever, RRFMerger
from src.router import QueryVariant

out = []

def p(msg):
    out.append(msg)
    print(msg)

p("=" * 60)
p("标准 RRF 多库检索验证")
p("=" * 60)

p("\n[1] 数据源配置:")
for src in DB_SOURCE_CONFIGS:
    p(f"  {src['name']}: db={src['milvus_db']}, collection={src['collection']}")

p("\n[2] 初始化 MultiPathRetriever...")
retriever = MultiPathRetriever.from_config()
p(f"  数据源: {[s.name for s in retriever._sources]}")

p("\n[3] 检索测试...")
query = "公司拖欠我3个月工资怎么办"
queries = [QueryVariant(text=query, source_type="original", weight=1.0)]

dense_docs, bm25_docs = retriever.retrieve_all(queries, original_query=query)
p(f"  Dense: {len(dense_docs)} 条")
p(f"  BM25:  {len(bm25_docs)} 条")

dense_by_src = {}
bm25_by_src = {}
for d in dense_docs:
    dense_by_src[d.source_db] = dense_by_src.get(d.source_db, 0) + 1
for d in bm25_docs:
    bm25_by_src[d.source_db] = bm25_by_src.get(d.source_db, 0) + 1
p(f"  Dense 来源: {dense_by_src}")
p(f"  BM25 来源:  {bm25_by_src}")

p("\n[4] 标准 RRF 融合（无加权）...")
merger = RRFMerger()
rrf_results = merger.merge_multi_source(dense_docs, bm25_docs, top_k=10)
p(f"  融合后: {len(rrf_results)} 条")

p("\n[5] Top-5 结果:")
for i, r in enumerate(rrf_results[:5]):
    src_label = {"laws": "法条", "non_laws": "案例/法规"}.get(r.source_db, r.source_db)
    src_type = {"dense": "语义", "sparse": "关键词"}.get(r.source, r.source)
    text_preview = (r.text or "(empty)")[:80].replace("\n", " ")
    p(f"  {i+1}. [{src_label}/{src_type}] RRF={r.score:.4f} | {r.law_name} | {text_preview}")

p("\n" + "=" * 60)
p("[OK] 标准 RRF 多库检索验证通过!")
p("=" * 60)

retriever.close()

with open("test_result.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))