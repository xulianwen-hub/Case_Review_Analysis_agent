"""10 行诊断脚本：定位 MultiPathRetriever.from_config() 哪一步挂"""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("Step 0: import config...")
from src.core.config import DB_SOURCE_CONFIGS, MODELS_DIR
print(f"  DB_SOURCE_CONFIGS = {DB_SOURCE_CONFIGS}")
for c in DB_SOURCE_CONFIGS:
    for k in ["milvus_db","sqlite_db"]:
        print(f"  → {c['name']}/{k} exists? {os.path.exists(c[k])} (path={c[k]})")

print("\nStep 1: import ranker classes...")
from src.rag.ranker import MultiPathRetriever, DBSource, RRFMerger, CrossEncoderRanker
print("  OK")

print("\nStep 2: create retriever obj via from_config()  [最可能崩，一步一步拆]")
ret = MultiPathRetriever(db_sources=[])
print("  2a: obj created, empty sources OK")

print("  2b: _init_model() 加载 bge-base-zh-v1.5...")
try:
    ret._init_model()
    print(f"  OK! model={ret._model}")
except Exception as e:
    print(f"  崩！ model加载异常: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("  2c: 逐个 DBSource() 实例化 & 连接...")
OK = 0
for i, src_cfg in enumerate(DB_SOURCE_CONFIGS):
    try:
        source = DBSource(
            name=src_cfg["name"],
            milvus_db=src_cfg["milvus_db"],
            sqlite_db=src_cfg["sqlite_db"],
            collection=src_cfg["collection"],
            db_type=src_cfg["db_type"],
            model=ret._model,
        )
        ret._sources.append(source)
        print(f"  [{i}] {src_cfg['name']} 连接 OK")
        OK += 1
    except Exception as e:
        print(f"  [{i}] {src_cfg['name']} 崩！ {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
if OK == 0:
    print("\n💥 所有数据源都连接失败！以上就是具体崩的原因")
    sys.exit(1)
print(f"  Total {OK}/{len(DB_SOURCE_CONFIGS)} 数据源 OK")

print("\nStep 3: test query 检索...")
from src.rag.ranker import QueryVariant
qv = QueryVariant(text="违法解除劳动合同怎么赔偿", variant_type="original")
dense_docs, bm25_docs = ret.retrieve_all([qv], original_query="违法解除怎么赔偿", top_k_per_source=3)
print(f"  Dense {len(dense_docs)} 条, BM25 {len(bm25_docs)} 条")
for d in dense_docs[:3]:
    print(f"   · Dense rank={d.rank} score={d.score:.3f} chunk={d.chunk_id} law={d.law_name} art={d.article}")

print("\nStep 4: RRF merge + CE init...")
merger = RRFMerger()
top15 = merger.merge_multi_source(dense_docs, bm25_docs, top_k=15)
print(f"  RRF after top15 = {len(top15)}")
ce = CrossEncoderRanker()
print("  CE init -> _init_model()")
ce._init_model()
print(f"  CE model loaded? initialized={ce._initialized}, model_obj={'OK' if ce._ce_model else 'None'}")

print("\nStep 5: CE rerank...")
out = ce.rerank("违法解除劳动合同怎么赔偿", top15, top_k=10, input_top_n=15)
print(f"  CE after top10 = {len(out)}")
for r in out[:5]:
    print(f"   · rank={r.rank} score={r.score:.4f} law={r.law_name} art={r.article}")

print("\n✅ 所有诊断步骤通过！")
ret.close()