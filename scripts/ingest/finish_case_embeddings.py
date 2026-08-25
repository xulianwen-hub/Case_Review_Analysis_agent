"""
单进程连续补完案例向量库嵌入（避免每批重载 BGE 模型）

原理：
- 模型只加载一次，循环处理未嵌入的子块
- 每批 300 条，批次间写入 Milvus 并记录进度
- 全部完成后写元信息文件
"""
import json
import os
import sys
import time
import argparse

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CASE_DIRS = [
    "data/processed/cases",
    "data/processed/judgments",
    "data/processed/regulations",
]
MILVUS_DB = r"d:\agent_develp\data\processed\vectors\cases_milvus.db"
COLLECTION_NAME = "labor_cases"
DIMENSION = 768
MODELS_DIR = r"d:\agent_develp\models"
PROGRESS_FILE = r"data\processed\vectors\cases_progress.json"


def load_all_childs():
    childs = []
    for dir_path in CASE_DIRS:
        f = os.path.join(dir_path, "child_chunks.json")
        if os.path.exists(f):
            with open(f, "r", encoding="utf-8") as fh:
                childs.extend(json.load(fh))
    return childs


def format_data_time(dt):
    if isinstance(dt, (int, float)) and dt > 1e12:
        from datetime import datetime
        try:
            return datetime.fromtimestamp(dt / 1000).strftime("%Y-%m-%d")
        except Exception:
            return ""
    return str(dt)[:30]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=300, help="每批嵌入数量")
    args = parser.parse_args()

    childs = load_all_childs()
    total = len(childs)
    print(f"[数据] 共 {total} 个子块")

    # 加载 BGE 模型（只加载一次）
    from sentence_transformers import SentenceTransformer
    model_path = os.path.join(MODELS_DIR, "models", "BAAI--bge-base-zh-v1.5", "snapshots", "master")
    if not os.path.exists(model_path):
        from modelscope import snapshot_download
        model_path = snapshot_download("BAAI/bge-base-zh-v1.5", cache_dir=MODELS_DIR)
    print(f"[模型] 加载: {model_path}")
    model = SentenceTransformer(model_path)

    # 连接 Milvus（延长超时避免 too_many_pings）
    from pymilvus import MilvusClient, DataType
    mc = MilvusClient(uri=MILVUS_DB, timeout=30)

    # 确保 collection
    if not mc.has_collection(COLLECTION_NAME):
        schema = mc.create_schema(auto_id=False, description="劳动纠纷案例向量库")
        schema.add_field("id", DataType.VARCHAR, max_length=32, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIMENSION)
        schema.add_field("text", DataType.VARCHAR, max_length=32768)
        schema.add_field("title", DataType.VARCHAR, max_length=256)
        schema.add_field("publish_source", DataType.VARCHAR, max_length=128)
        schema.add_field("data_time", DataType.VARCHAR, max_length=32)
        schema.add_field("parent_key", DataType.VARCHAR, max_length=64)
        index_params = mc.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="IVF_FLAT",
                               metric_type="IP", params={"nlist": 128})
        mc.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)
        print(f"[Milvus] 创建 collection: {COLLECTION_NAME}")

    # 获取已嵌入的 id
    embedded_ids = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            embedded_ids = set(json.load(f))
    # 尝试从 Milvus 查询补充
    mc.load_collection(COLLECTION_NAME)
    try:
        page_size = 1000
        offset = 0
        while True:
            res = mc.query(collection_name=COLLECTION_NAME, filter="id != ''",
                           output_fields=["id"], limit=page_size, offset=offset)
            if not res:
                break
            for row in res:
                embedded_ids.add(row.get("id", ""))
            offset += page_size
            if len(res) < page_size:
                break
    except Exception as e:
        print(f"[提示] 从 Milvus 查询已有 id 失败: {e}")

    print(f"[进度] 已嵌入: {len(embedded_ids)}/{total}")

    # 循环嵌入剩余
    pending = [c for c in childs if c["chunk_id"] not in embedded_ids]
    print(f"[待嵌入] {len(pending)} 条")

    round_no = 0
    while pending:
        round_no += 1
        batch = pending[:args.batch]
        texts = [c["content"] for c in batch]

        print(f"\n[第{round_no}轮] 嵌入 {len(batch)} 条...")
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        rows = []
        for j, c in enumerate(batch):
            text = c["content"]
            if len(text) > 32700:
                text = text[:32700]
            rows.append({
                "id": c["chunk_id"],
                "vector": embeddings[j].tolist(),
                "text": text,
                "title": c.get("title", "")[:250],
                "publish_source": c.get("publishSource", "")[:120],
                "data_time": format_data_time(c.get("dataTime", "")),
                "parent_key": c.get("parent_key", "")[:60],
            })

        try:
            mc.insert(collection_name=COLLECTION_NAME, data=rows)
            ok = len(rows)
        except Exception as e:
            err_msg = str(e)
            print(f"  [警告] 批量插入失败: {err_msg[:120]}")
            if "GOAWAY" in err_msg or "UNAVAILABLE" in err_msg:
                print("  [重连] 重建 Milvus 连接...")
                time.sleep(2)
                mc.close()
                mc = MilvusClient(uri=MILVUS_DB, timeout=30)
                mc.load_collection(COLLECTION_NAME)
            # 逐条重试
            ok = 0
            for row in rows:
                try:
                    mc.insert(collection_name=COLLECTION_NAME, data=[row])
                    ok += 1
                except Exception as e2:
                    print(f"  [跳过] {row['id']}: {str(e2)[:80]}")
        print(f"  [Milvus] 插入成功: {ok}")

        # 批次间短暂休息，避免 gRPC too_many_pings
        time.sleep(1)

        # 更新进度
        for c in batch:
            embedded_ids.add(c["chunk_id"])
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(embedded_ids), f, ensure_ascii=False)

        pending = [c for c in childs if c["chunk_id"] not in embedded_ids]
        print(f"  [进度] 剩余: {len(pending)}")

        if len(pending) == 0:
            break

    mc.load_collection(COLLECTION_NAME)
    stats = mc.get_collection_stats(COLLECTION_NAME)
    print(f"\n{'='*40}")
    print(f"[完成] 总向量数: {stats['row_count']}/{total}")
    print(f"{'='*40}")

    # 元信息
    meta = {
        "collection": COLLECTION_NAME,
        "total_vectors": stats["row_count"],
        "dimension": DIMENSION,
        "model": "BAAI/bge-base-zh-v1.5",
        "completed": stats["row_count"] >= total,
    }
    with open("data/processed/vectors/cases_vector_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    mc.close()


if __name__ == "__main__":
    main()