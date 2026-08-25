"""直接测试 DBSource.bm25_search()"""
import sys
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, ".")
from src.ranker import DBSource

source_laws = DBSource(
    name="laws",
    milvus_db="data/processed/vectors/laws_milvus.db",
    sqlite_db="data/processed/chunks.db",
    collection="labor_laws",
    db_type="laws",
    model=None,
)

source_cases = DBSource(
    name="non_laws",
    milvus_db="data/processed/vectors/cases_milvus.db",
    sqlite_db="data/processed/chunks_cases.db",
    collection="labor_cases",
    db_type="cases",
    model=None,
)

query = "公司拖欠我3个月工资怎么办"
print(f"Query: {query}\n")

for source in [source_laws, source_cases]:
    print(f"[{source.name}] Testing BM25...")
    results = source.bm25_search(query, top_k=15)
    print(f"  Results: {len(results)}")
    for r in results[:3]:
        print(f"  - id={r.chunk_id}, score={r.score:.4f}, rank={r.rank}")
    print()

source_laws.close()
source_cases.close()
print("Done.")