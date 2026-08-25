"""快速诊断：检查数据库是否可访问"""
import sys
import os
import sqlite3
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.insert(0, ".")

# 检查 SQLite
laws_db = "data/processed/chunks.db"
cases_db = "data/processed/chunks_cases.db"

for db_path, label in [(laws_db, "laws"), (cases_db, "cases")]:
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        count = cur.fetchone()[0]
        print(f"[{label}] SQLite: {count} chunks in {db_path}")
        conn.close()
    else:
        print(f"[{label}] SQLite: NOT FOUND: {db_path}")

# 检查 Milvus
from pymilvus import MilvusClient

laws_milvus = "data/processed/vectors/laws_milvus.db"
cases_milvus = "data/processed/vectors/cases_milvus.db"

for db_path, collection, label in [(laws_milvus, "labor_laws", "laws"), (cases_milvus, "labor_cases", "cases")]:
    if os.path.exists(db_path):
        try:
            mc = MilvusClient(uri=db_path)
            stats = mc.get_collection_stats(collection)
            print(f"[{label}] Milvus: {stats['row_count']} vectors in {collection}")
            mc.close()
        except Exception as e:
            print(f"[{label}] Milvus: ERROR: {e}")
    else:
        print(f"[{label}] Milvus: NOT FOUND: {db_path}")

print("\nDone.")