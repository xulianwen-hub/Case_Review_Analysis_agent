"""验证案例向量库最终状态"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

result = {}

# 1. SQLite 状态
sqlite_path = 'data/processed/chunks_cases.db'
result['sqlite'] = {'exists': os.path.exists(sqlite_path),
                    'size': os.path.getsize(sqlite_path) if os.path.exists(sqlite_path) else 0}
print(f"SQLite: exists={result['sqlite']['exists']}, size={result['sqlite']['size']}")

# 2. Milvus 状态
milvus_path = 'data/processed/vectors/cases_milvus.db'
result['milvus_dir_exists'] = os.path.isdir(milvus_path)
print(f"Milvus dir: exists={result['milvus_dir_exists']}")

if os.path.isdir(milvus_path):
    try:
        from pymilvus import MilvusClient
        mc = MilvusClient(uri=milvus_path)
        cols = mc.list_collections()
        result['collections'] = cols
        print(f"collections: {cols}")
        if 'labor_cases' in cols:
            try:
                stats = mc.get_collection_stats('labor_cases')
                result['labor_cases_rows'] = stats.get('row_count', -1)
                print(f"labor_cases rows: {stats.get('row_count', -1)}")
            except Exception as e:
                result['stats_err'] = str(e)[:200]
                print(f"stats error: {e}")
        mc.close()
    except Exception as e:
        result['milvus_err'] = str(e)[:300]
        print(f"Milvus 错误: {type(e).__name__}: {str(e)[:200]}")

with open('_verify_result.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("DONE")