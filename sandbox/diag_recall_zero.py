"""诊断 RAG_001：违法解除 2N。定位 Recall 全 0% 的根本原因。"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = ROOT / "sandbox" / "diag_recall_zero.log"
_f = open(LOG, "w", encoding="utf-8")
_orig_out, _orig_err = sys.stdout, sys.stderr
sys.stdout = _f
sys.stderr = _f
def _close():
    try:
        sys.stdout.flush(); sys.stderr.flush(); _f.close()
        sys.stdout, sys.stderr = _orig_out, _orig_err
    except Exception:
        pass
import atexit
atexit.register(_close)

from src.rag.ranker import (
    MultiPathRetriever, RRFMerger, CrossEncoderRanker,
    RankResult, QueryVariant,
)

retriever = MultiPathRetriever.from_config()
merger = RRFMerger()
ce = CrossEncoderRanker()

# 1. RAG_001 真实跑
query = "公司违法解除劳动合同怎么赔偿？"
qv = QueryVariant(text=query, source_type="original")
dense_docs, bm25_docs = retriever.retrieve_all([qv], original_query=query)

# 2. 打印 Dense Top-5
print("\n=== A 管线（纯 Dense）Top 6 ===")
uniq, seen = [], set()
for d in sorted(dense_docs, key=lambda x: x.score, reverse=True):
    if d.chunk_id in seen: continue
    seen.add(d.chunk_id)
    uniq.append(d)
for i, d in enumerate(uniq[:6], 1):
    print(f"  [{i}] score={d.score:.4f} chunk={d.chunk_id[:20]}...")
    print(f"       law_name={d.law_name!r}  article={d.article!r}  chapter={d.chapter!r}")
    print(f"       title={d.title!r}")
    print(f"       text[:80]={(d.text or '')[:80]!r}")

# 3. 打印 RRF 粗排 Top 6（B 管线）
print("\n=== B 管线（Dense+BM25+RRF 粗排）Top 6 ===")
rrf_top = merger.merge_multi_source(dense_docs, bm25_docs, top_k=15)
for r in rrf_top[:6]:
    print(f"  [{r.rank}] RRF_score={r.score:.4f} source_db={r.source_db}")
    print(f"       law_name={r.law_name!r}  article={r.article!r}  chapter={r.chapter!r}")
    print(f"       title={r.title!r}")
    print(f"       text[:80]={(r.text or '')[:80]!r}")

# 4. 打印匹配器的抽取出结果
print("\n=== 测试法条标识符匹配器 ===")
import re
def _extract(r: RankResult):
    combined = f"{r.law_name or ''} {r.article or ''} {r.chapter or ''} {r.title or ''} {r.text or ''[:200]}"
    print(f"  combined 输入 = {combined[:150]!r}")
    m = re.search(r"第([一二三四五六七八九十百千两0-9]+)条", combined)
    if not m: return ""
    cn_num = m.group(1)
    mapping = {"零":0,"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"百":100,"千":1000,"两":2}
    try:
        arabic = int(cn_num)
    except ValueError:
        total, current = 0, 0
        for ch in cn_num:
            if ch == "十":
                current = current if current else 1
                total += current * 10; current = 0
            elif ch == "百":
                current = current if current else 1
                total += current * 100; current = 0
            elif ch == "千":
                current = current if current else 1
                total += current * 1000; current = 0
            elif ch in mapping:
                current = mapping[ch]
        total += current
        arabic = total
    return f"第{arabic}条"

print(f"  RAG_001 的 must_hit = ['第四十七条','第四十八条','第八十七条']")
print(f"  标准化后 = ['第47条','第48条','第87条']\n")
for r in rrf_top[:6]:
    ident = _extract(r)
    print(f"  第{r.rank}名 → 抽取标识符={ident!r}  → 命中 must_hit? {ident in ['第47条','第48条','第87条']}")
    print()

retriever.close()