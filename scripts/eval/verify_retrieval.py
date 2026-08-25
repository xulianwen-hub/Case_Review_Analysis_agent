"""独立验证脚本——测试已构建的向量库检索效果"""
import json
import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

CHILD_CHUNKS_FILE = r"d:\agent_develp\data\processed\laws\child_chunks.json"
PARENT_CHUNKS_FILE = r"d:\agent_develp\data\processed\laws\parent_chunks.json"
MILVUS_DB_FILE = r"d:\agent_develp\data\processed\vectors\laws_milvus.db"
MODEL_CACHE_DIR = r"d:\agent_develp\models"

# BGE 模型本地路径（自动查找 base 或 small）
def _find_model_dir():
    base_path = os.path.join(MODEL_CACHE_DIR, "models", "BAAI--bge-base-zh-v1.5", "snapshots", "master")
    if os.path.exists(base_path):
        return base_path
    small_path = os.path.join(MODEL_CACHE_DIR, "models", "BAAI--bge-small-zh-v1.5", "snapshots", "master")
    if os.path.exists(small_path):
        return small_path
    return base_path  # fallback

MODEL_DIR = _find_model_dir()

def main():
    print("=" * 60)
    print("向量库检索验证")
    print("=" * 60)

    # 加载模型
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_DIR)

    # 加载父块映射
    parent_map = {}
    if os.path.exists(PARENT_CHUNKS_FILE):
        with open(PARENT_CHUNKS_FILE, "r", encoding="utf-8") as f:
            for p in json.load(f):
                parent_map[p["parent_key"]] = p["content"]

    # 连接向量库
    from pymilvus import MilvusClient
    mc = MilvusClient(uri=MILVUS_DB_FILE)
    mc.load_collection("labor_laws")

    stats = mc.get_collection_stats("labor_laws")
    print(f"\n  总向量数: {stats['row_count']}")
    print(f"  父块数: {len(parent_map)}")

    # 测试检索
    test_queries = [
        ("拖欠工资解除合同", "公司拖欠我3个月工资，我想辞职并要求赔偿，有什么法律依据？"),
        ("经济补偿计算", "我在公司工作了5年，月薪8000，公司要辞退我，应该补偿多少？"),
        ("工伤认定", "我在上班路上被车撞了，这算工伤吗？需要什么材料？"),
        ("试用期", "试用期最长可以多久？试用期工资怎么算？"),
        ("竞业限制", "离职后公司不让我去同行工作，合法吗？"),
    ]

    for label, query in test_queries:
        query_embedding = model.encode([query], normalize_embeddings=True)

        results = mc.search(
            collection_name="labor_laws",
            data=query_embedding.tolist(),
            anns_field="vector",
            search_params={"metric_type": "IP", "params": {"nprobe": 8}},
            limit=3,
            output_fields=["law_name", "chapter", "article", "text", "parent_key"],
        )

        print(f"\n{'─'*60}")
        print(f"[查询: {label}]")
        print(f"  Q: {query[:60]}...")
        for j, hits in enumerate(results):
            for k, hit in enumerate(hits):
                entity = hit.get('entity', {})
                law = entity.get('law_name', '')
                article = entity.get('article', '')
                score = hit['distance']
                text = entity.get('text', '')[:80]
                print(f"  Top{k+1}: [{law} | {article}] score={score:.4f}")
                print(f"          {text}...")
                pkey = entity.get("parent_key", "")
                if pkey in parent_map:
                    pp = parent_map[pkey][:120].replace("\n", " ")
                    print(f"          [父块] {pp}...")

    mc.close()
    print(f"\n{'='*60}")
    print("验证完成")

if __name__ == "__main__":
    main()