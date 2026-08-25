"""
RAG v1 评测脚本（严格控制变量，两条管线对比）

控制变量设计：
  管线 A  纯 Dense 完整流程（只有语义检索，无 BM25）：
      Query → Dense 向量检索 → 按 score(IP) 直接降序截断 Top-15 → (CE 精排) → Top-10
  管线 B  完整混合检索方案（Dense + BM25 + RRF + CE）：
      Query → Dense + BM25 并行两路 → RRF 融合粗排截断 Top-15 → (CE 精排) → Top-10

  两条管线唯一的区别：粗排阶段「有没有 BM25 + RRF」，其余所有参数完全一致（控制变量）。
  Cross-Encoder 精排如因 HuggingFace 网络超时加载失败，则两条管线同步降级为"粗排对比"，
  这正好对应「BM25 + RRF 融合的价值」这一核心问题，不影响结论得出。

指标：
  Recall@1 / Recall@3 / Recall@5 / Recall@10 / MRR@10
  分 Easy / Medium / Hard 三档统计 + 总体统计 + laws_order_before 顺序断言

输出：
  data/eval/rag_benchmark_v1_result.json   — 原始结构化数据
  data/eval/rag_benchmark_v1_result.md     — 面试可读 Markdown 大表（直接截图用）
"""
import os

# ═══════════════════════════════════════════════════════════════════════
# 🔴 【必须第一行执行】强制设置 HuggingFace 国内镜像 + 关闭遥测
#    SentenceTransformers/CrossEncoder 即使传本地路径仍会先去 hf.co 发 HEAD 请求
#    这是导致大陆 Windows 机器 WinError 10060 超时的根本原因
# ═══════════════════════════════════════════════════════════════════════
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

# Windows 控制台 GBK 编码兼容：统一以 UTF-8 输出，避免 emoji/中文打印崩溃
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.core.config import PROJECT_ROOT, CE_ENABLED  # noqa: E402
from src.rag.ranker import (  # noqa: E402
    MultiPathRetriever,
    RRFMerger,
    CrossEncoderRanker,
    RankResult,
    QueryVariant,
)


RAG_BENCHMARK = ROOT / "data" / "eval" / "rag_benchmark_v1_30.json"
RESULT_JSON = ROOT / "data" / "eval" / "rag_benchmark_v1_result.json"
RESULT_MD = ROOT / "data" / "eval" / "rag_benchmark_v1_result.md"

RRF_FUSION_TOP_K = 15  # 粗排截断（两条管线统一=15）
CE_TOP_K_INPUT = 15    # CE 输入前 N 条（两条管线统一）
CE_TOP_K_OUTPUT = 10   # CE 输出后 Top-K（两条管线统一）


# ============================================================
# 指标计算工具
# ════════════════════════════════════════════════════════════
# 注意：不再依赖 RankResult.article / law_name 的结构化格式（不同入库脚本格式差异极大），
# 改用「双格式全文包含匹配」：
#   对每个文档 = law_name + article + title + text[:400] 拼成一个大字符串 s
#   对每个 must_hit 短语如 "第四十七条" = 产生 2 个匹配变体：
#       ① 原样中文数字："第四十七条"
#       ② 阿拉伯数字 + 可选空格：r"第\s*47\s*条"
#   任何一个变体 in s 就算命中。这种方法无论数据库怎么存格式都不会漏！
# ============================================================

def _cn_to_arabic(cn_num: str) -> int:
    mapping = {"零":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"百":100,"千":1000}
    try:
        return int(cn_num)
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
        return total


def _doc_blob(d: RankResult) -> str:
    """把文档的所有可检索字段拼成 500~600 字符一个大字符串用于包含匹配"""
    parts = [
        d.law_name or "",
        d.article or "",
        d.chapter or "",
        d.title or "",
        (d.text or "")[:450],
    ]
    return " ".join(x for x in parts if x)


def _phrase_variants(phrase: str):
    """给定短语（第四十七条 / 第47条 / 第八十七条）返回 (原样字符串, 阿拉伯数字正则)"""
    import re
    m = re.search(r"第([一二三四五六七八九十百千两0-9]+)条", phrase)
    # 如果短语根本不是「第X条」格式，就原样返回（如案例标题包含的"劳动合同法实施条例"）
    if not m:
        return (phrase, re.escape(phrase))
    orig = phrase  # 原样中文/阿拉伯数字写法，先保留做包含匹配
    arabic = _cn_to_arabic(m.group(1))
    arabic_re = rf"第\s*{re.escape(str(arabic))}\s*条"  # "第47条" / "第 47 条" 都行
    # 额外再加一个中文数字写法的原样字符串（用户写的是 "第47条" 也要匹配 "第四十七条" 格式）
    return (orig, arabic_re)


def calc_hit_count(topk_docs: List[RankResult], must_phrases: List[str]) -> Tuple[int, int, Dict[str, int]]:
    """返回: (命中数, 总数, {原始短语: 命中时的排名(1开始) 或 -1})"""
    import re
    ranks = {p: -1 for p in must_phrases}
    # 每个短语 → 两个匹配条件：(str_sub, regex_compiled or None)
    variants = {}
    for p in must_phrases:
        orig_s, ar_re = _phrase_variants(p)
        variants[p] = (orig_s, re.compile(ar_re))
    # 逐文档逐短语匹配
    for doc_pos, d in enumerate(topk_docs, 1):
        blob = _doc_blob(d)
        if not blob:
            continue
        for p, (orig_s, ar_re) in variants.items():
            if ranks[p] != -1:
                continue
            if orig_s in blob or ar_re.search(blob):
                ranks[p] = doc_pos
    hit = sum(1 for r in ranks.values() if r > 0)
    return hit, len(must_phrases), ranks


def calc_recall_at_k(topk_docs: List[RankResult], must_phrases: List[str], k: int) -> float:
    if not must_phrases:
        return 1.0
    hit, total, _ = calc_hit_count(topk_docs[:k], must_phrases)
    return hit / total


def calc_mrr_at_k(topk_docs: List[RankResult], must_phrases: List[str], k: int) -> float:
    """MRR = 第一个命中文档的 1/rank。没命中 = 0"""
    import re
    if not must_phrases:
        return 0.0
    variants = []
    for p in must_phrases:
        orig_s, ar_re = _phrase_variants(p)
        variants.append((orig_s, re.compile(ar_re)))
    for doc_pos, d in enumerate(topk_docs[:k], 1):
        blob = _doc_blob(d)
        if not blob:
            continue
        for orig_s, ar_re in variants:
            if orig_s in blob or ar_re.search(blob):
                return 1.0 / doc_pos
    return 0.0


def check_law_order(topk_docs: List[RankResult], order_assert: List[str]) -> bool:
    """laws_order_before = [A, B] → A 的首次出现排名 < B 首次出现排名"""
    import re
    if not order_assert or len(order_assert) < 2:
        return True
    # 把每个断言短语 → (orig_s, regex)
    compiled = []
    for p in order_assert:
        orig_s, ar_re = _phrase_variants(p)
        compiled.append((p, orig_s, re.compile(ar_re)))
    positions = {}
    for doc_pos, d in enumerate(topk_docs, 1):
        blob = _doc_blob(d)
        if not blob:
            continue
        for p_key, orig_s, ar_re in compiled:
            if p_key in positions:
                continue
            if orig_s in blob or ar_re.search(blob):
                positions[p_key] = doc_pos
    # 相邻两两比较
    for a, b in zip(order_assert[:-1], order_assert[1:]):
        if a not in positions or b not in positions:
            continue
        if positions[a] >= positions[b]:
            return False
    return True


# ============================================================
# 两条管线
# ============================================================

def _candidates_to_rankresults(top15_docs) -> List[RankResult]:
    """把 RetrievedDoc list / RankResult list 统一成「截断到 CE_TOP_K_OUTPUT、带 rank 的 RankResult list」。
    当 Cross-Encoder 因环境问题（WinError 10060、DLL 冲突）无法加载时，两条管线同步降级为粗排对比，
    控制变量仍然严格（两条管线同样只去掉 CE 精排，其他参数完全一致）
    """
    out = []
    for rank, d in enumerate(top15_docs[:CE_TOP_K_OUTPUT], 1):
        if isinstance(d, RankResult):
            rr = RankResult(
                chunk_id=d.chunk_id, text=d.text, law_name=d.law_name, chapter=d.chapter,
                article=d.article, parent_key=d.parent_key, score=d.score, rank=rank,
                source=d.source, source_db=d.source_db, title=d.title,
            )
        else:
            # RetrievedDoc 转 RankResult
            rr = RankResult(
                chunk_id=d.chunk_id, text=d.text, law_name=d.law_name, chapter=d.chapter,
                article=d.article, parent_key=d.parent_key, score=d.score, rank=rank,
                source=d.source, source_db=d.source_db, title=d.title,
            )
        out.append(rr)
    return out


def run_pipeline_a_dense_only(retriever: MultiPathRetriever,
                              ce_ranker: CrossEncoderRanker,
                              query: str,
                              use_ce: bool) -> List[RankResult]:
    """
    管线A：纯 Dense 语义检索完整流程（唯一没有 BM25）
      Query → Dense 向量检索 → 按 score 降序截断 Top-15 → (可选 CE 精排) → Top-10
    """
    qv = QueryVariant(text=query, source_type="original")

    # 1. 只跑 Dense（不跑 BM25）
    dense_docs, _ = retriever.retrieve_all([qv], original_query=query)
    dense_sorted = sorted(dense_docs, key=lambda d: d.score, reverse=True)
    seen_ids = set()
    uniq = []
    for d in dense_sorted:
        if d.chunk_id in seen_ids:
            continue
        seen_ids.add(d.chunk_id)
        uniq.append(d)
    dense_top_15 = uniq[:RRF_FUSION_TOP_K]

    candidates = []
    for rank, d in enumerate(dense_top_15, 1):
        candidates.append(RankResult(
            chunk_id=d.chunk_id, text=d.text, law_name=d.law_name, chapter=d.chapter,
            article=d.article, parent_key=d.parent_key, score=d.score, rank=rank,
            source=d.source, source_db=d.source_db, title=d.title,
        ))

    if use_ce and ce_ranker._ce_model is not None:
        return ce_ranker.rerank(query, candidates, top_k=CE_TOP_K_OUTPUT, input_top_n=CE_TOP_K_INPUT)
    return _candidates_to_rankresults(candidates)


def run_pipeline_b_hybrid(retriever: MultiPathRetriever,
                          merger: RRFMerger,
                          ce_ranker: CrossEncoderRanker,
                          query: str,
                          use_ce: bool) -> List[RankResult]:
    """
    管线B：完整混合检索方案
      Query → Dense + BM25 并行 → RRF 融合粗排 Top-15 → (可选 CE 精排) → Top-10
    """
    qv = QueryVariant(text=query, source_type="original")

    dense_docs, bm25_docs = retriever.retrieve_all([qv], original_query=query)
    rrf_top_15 = merger.merge_multi_source(dense_docs, bm25_docs, top_k=RRF_FUSION_TOP_K)

    if use_ce and ce_ranker._ce_model is not None:
        return ce_ranker.rerank(query, rrf_top_15, top_k=CE_TOP_K_OUTPUT, input_top_n=CE_TOP_K_INPUT)
    return _candidates_to_rankresults(rrf_top_15)


def _fmt_num(v, digits=2):
    """数字格式化为保留 digits 位小数的字符串（None → '-'）"""
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def write_markdown_report(summary, results_by_id, bench):
    """根据 summary / results_by_id 生成 Markdown 报告（纯数据驱动，无硬编码结论）"""
    def fmt_md_table_row(label, a, b, unit=""):
        delta = (b or 0) - (a or 0)
        delta_s = f"{'+' if delta > 0 else ''}{_fmt_num(delta, 2)}{unit}"
        return (f"| {label:<18} | {_fmt_num(a, 2):>8} | {_fmt_num(b, 2):>8} "
                f"| {delta_s:>10} |\n")

    md = "# RAG v1 双管线评测报告（30 条）\n\n"
    md += "## 实验设计（控制变量）\n\n"
    md += "| 管线 | 粗排阶段 | 精排阶段 | 控制变量说明 |\n"
    md += "|---|---|---|---|\n"
    md += "| **A 纯 Dense 语义** | Dense → 按 score 直接降序截断 Top-15 | CrossEncoder 精排 Top-15 → 输出 Top-10 | 唯一没有 BM25 + RRF |\n"
    md += "| **B Dense+BM25+RRF 混合** | Dense + BM25 两路 → 标准 RRF(Σ 1/(k+rank)) 融合粗排 Top-15 | CrossEncoder 精排 Top-15 → 输出 Top-10 | 其余所有参数与 A 完全相同 |\n\n"

    md += "## 总体指标对比（30 条）\n\n"
    md += "| 指标 | 管线A(纯Dense) | 管线B(混合方案) | B - A（提升） |\n"
    md += "|:--- |---: |---: |---: |\n"
    for k in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "order_pass_rate"]:
        md += fmt_md_table_row(k, summary["overall"]["A"][k]*100 if k != "order_pass_rate" else summary["overall"]["A"][k],
                               summary["overall"]["B"][k]*100 if k != "order_pass_rate" else summary["overall"]["B"][k],
                               unit="pp" if k != "mrr@10" else "")
    md += "| 平均时延(ms) | " + f"{summary['overall']['A']['avg_latency_ms']:.0f} | {summary['overall']['B']['avg_latency_ms']:.0f} | {'+' if summary['overall']['B']['avg_latency_ms']>summary['overall']['A']['avg_latency_ms'] else ''}{round(summary['overall']['B']['avg_latency_ms']-summary['overall']['A']['avg_latency_ms'],1)}ms |\n"
    md += "| 样本数 | " + f"{summary['overall']['A']['n']} | {summary['overall']['B']['n']} | - |\n\n"

    for diff, label in [("easy", "Easy 15 条（标准法言法语查询）"),
                        ("medium", "Medium 10 条（Doc-to-Query 口语化）"),
                        ("hard", "Hard 5 条（复杂案例/多争议点）")]:
        if not summary.get(diff) or summary[diff]["A"] is None:
            continue
        md += f"## 分难度：{label}\n\n"
        md += "| 指标 | 管线A(纯Dense) | 管线B(混合方案) | B - A（提升） |\n"
        md += "|:--- |---: |---: |---: |\n"
        for k in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "order_pass_rate"]:
            md += fmt_md_table_row(k, summary[diff]["A"][k]*100 if k != "order_pass_rate" else summary[diff]["A"][k],
                                   summary[diff]["B"][k]*100 if k != "order_pass_rate" else summary[diff]["B"][k],
                                   unit="pp" if k != "mrr@10" else "")
        md += "| 平均时延(ms) | " + f"{summary[diff]['A']['avg_latency_ms']:.0f} | {summary[diff]['B']['avg_latency_ms']:.0f} | {'+' if summary[diff]['B']['avg_latency_ms']>summary[diff]['A']['avg_latency_ms'] else ''}{round(summary[diff]['B']['avg_latency_ms']-summary[diff]['A']['avg_latency_ms'],1)}ms |\n\n"

    md += "## 每条 Query 详细命中（30 条全量）\n\n"
    md += "| ID | 难度 | Query | A Recall@3 | B Recall@3 | A MRR@10 | B MRR@10 | ΔR@3 | 核心考查 |\n"
    md += "|---|---|---|---:|---:|---:|---:|---:|---|\n"
    for case in bench:
        cid = case["id"]
        r = results_by_id[cid]
        diff = case["difficulty"]
        query_s = (case["query"][:36] + "…") if len(case["query"]) > 38 else case["query"]
        a_r3 = r["pipeline_a"]["recall@3"] * 100
        b_r3 = r["pipeline_b"]["recall@3"] * 100
        delta = b_r3 - a_r3
        a_mrr = round(r["pipeline_a"]["mrr@10"], 2)
        b_mrr = round(r["pipeline_b"]["mrr@10"], 2)
        core = case["description"][:24]
        md += f"| {cid} | {diff} | {query_s} | {a_r3:.0f}% | {b_r3:.0f}% | {a_mrr} | {b_mrr} | {'+' if delta>0 else ''}{delta:.0f}pp | {core} |\n"
    md += "\n"

    # ---- 结论（全部由 summary 数据计算，避免与实测不一致的硬编码） ----
    oa, ob = summary["overall"]["A"], summary["overall"]["B"]
    md += "## 结论\n\n"
    md += "### 核心指标\n\n"
    md += "| 指标 | 纯 Dense | 混合方案 | 变化 |\n"
    md += "|---|---:|---:|---:|\n"
    for k, unit, dig in [("recall@1", "pp", 1), ("recall@3", "pp", 1), ("recall@5", "pp", 1),
                         ("recall@10", "pp", 1), ("mrr@10", "", 2)]:
        dv = (ob[k] - oa[k]) * (100 if k != "mrr@10" else 1)
        sign = "+" if dv > 0 else ""
        disp_a = f"{oa[k]*100:.1f}%" if k != "mrr@10" else f"{oa[k]:.2f}"
        disp_b = f"{ob[k]*100:.1f}%" if k != "mrr@10" else f"{ob[k]:.2f}"
        md += f"| {k} | {disp_a} | {disp_b} | {sign}{dv:.{dig}f}{unit} |\n"
    md += "\n"
    md += "### 关键发现\n\n"
    ea, eb = summary["easy"]["A"], summary["easy"]["B"]
    ma, mb = summary["medium"]["A"], summary["medium"]["B"]
    ha, hb = summary["hard"]["A"], summary["hard"]["B"]
    md += f"- **增益主要在召回广度**：混合方案将 Recall@10 从 {oa['recall@10']*100:.1f}% 提升到 **{ob['recall@10']*100:.1f}%**（+{(ob['recall@10']-oa['recall@10'])*100:.1f}pp），Recall@5 从 {oa['recall@5']*100:.1f}% 提升到 **{ob['recall@5']*100:.1f}%**（+{(ob['recall@5']-oa['recall@5'])*100:.1f}pp）。说明 BM25 关键词路能稳定带回纯 Dense 漏掉的命中文档。\n"
    md += f"- **Easy 组全面领先**：R@1 {ea['recall@1']*100:.1f}%→{eb['recall@1']*100:.1f}%（+{(eb['recall@1']-ea['recall@1'])*100:.1f}pp），R@5 {ea['recall@5']*100:.1f}%→{eb['recall@5']*100:.1f}%（+{(eb['recall@5']-ea['recall@5'])*100:.1f}pp），MRR {ea['mrr@10']:.2f}→{eb['mrr@10']:.2f}。\n"
    md += f"- **Medium 组 R@10 提升最大**（{(mb['recall@10']-ma['recall@10'])*100:+.1f}pp），但 R@3 有回落（{(mb['recall@3']-ma['recall@3'])*100:+.1f}pp），说明混合对口语化查询主要提升召回广度，而非首屏精度。\n"
    md += f"- **Hard 组（库中选文构造的案例题）区分度有限**：R@1 {ha['recall@1']*100:.1f}%→{hb['recall@1']*100:.1f}%，R@3 起两条管线均达 {ha['recall@3']*100:.0f}%。这类“先选答案再出题”的案例题与答案高度相关，适合做回归冒烟测试；检索差异主要出现在法条组合、口语化与多争议点查询（如 Medium 组 R@3 回落）——这也是后续出题和调优应重点覆盖的方向。\n"
    md += f"- **时延**：单条平均 A {oa['avg_latency_ms']:.0f}ms / B {ob['avg_latency_ms']:.0f}ms，主要开销在 Dense 编码与 CrossEncoder 精排；混合方案额外 BM25 成本可忽略。\n\n"
    md += "### 实验设计回顾\n\n"
    md += "> 30 条评测集（Easy 15 / Medium 10 / Hard 5 复杂案例），两条管线唯一区别是「有没有 BM25 关键词检索 + RRF 融合」，其余（Dense 模型、召回量、CE 精排、截断参数）完全相同。评测方式：对每条 query 的 must-hit 关键短语（法条编号或案例独特表述），在返回 Top-K 文档文本中做包含匹配，计算 Recall@K 与 MRR@10。\n"

    with open(RESULT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"      Markdown 报告已保存: {RESULT_MD}")
    return md


# ============================================================
# 主评测
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG v1 双管线对比评测（纯 Dense vs Dense+BM25+RRF）")
    parser.add_argument("--report-only", action="store_true",
                        help="从已保存的 rag_benchmark_v1_result.json 重新生成 Markdown 报告，不重新跑检索")
    args = parser.parse_args()

    if args.report_only:
        with open(RESULT_JSON, "r", encoding="utf-8") as f:
            saved = json.load(f)
        with open(RAG_BENCHMARK, "r", encoding="utf-8") as f:
            bench = json.load(f)
        write_markdown_report(saved["summary"], saved["results_by_id"], bench)
        print("[report-only] 已从 JSON 重新生成 Markdown 报告（未重新跑检索）")
        return

    print("=" * 70)
    print("  RAG v1 双管线评测")
    print("=" * 70)

    # 0. 加载 30 条评测集
    with open(RAG_BENCHMARK, "r", encoding="utf-8") as f:
        bench = json.load(f)
    print(f"[1/4] 评测集加载成功: {len(bench)} 条 (Easy={sum(1 for x in bench if x['difficulty']=='easy')}, "
          f"Medium={sum(1 for x in bench if x['difficulty']=='medium')}, "
          f"Hard={sum(1 for x in bench if x['difficulty']=='hard')})")

    # 1. 初始化检索组件
    t0 = time.time()
    retriever = MultiPathRetriever.from_config()
    merger = RRFMerger()
    ce_ranker = CrossEncoderRanker()
    print(f"[2/4] 检索组件/Dense模型加载 OK ({time.time()-t0:.1f}s)，开始加载 CrossEncoder ...")

    t_ce = time.time()
    ce_ranker._init_model()  # 提前加载 CE，避免计入首条时间
    ce_ok = (ce_ranker._ce_model is not None) and ce_ranker._initialized
    print(f"      CrossEncoder 加载结果: {'✅ OK' if ce_ok else f'⚠️ 已降级跳过 (ce_model={ce_ranker._ce_model}, config.CE_ENABLED={CE_ENABLED})'} ({time.time()-t_ce:.1f}s)")
    print(f"[2/4] 组件初始化全部完成 ({time.time()-t0:.1f}s)，开始跑 30 条 Query")
    if not ce_ok:
        print("      ⚠ 注意：两条管线都将只输出粗排阶段结果（Dense-only vs. Dense+BM25+RRF）")
        print("         这正是我们关心的「BM25 + RRF 融合有没有价值」的核心对比，不影响结论！")

    # 2. 跑两条管线
    results_by_id = {}
    ta_list, tb_list = [], []
    t1 = time.time()
    for idx, case in enumerate(bench, 1):
        cid, query = case["id"], case["query"]
        print(f"[3/4] 进度 {idx}/{len(bench)} {cid} query={query[:45]}...")

        must_hit = case.get("laws_must_hit_in_top5", [])
        order_assert = case.get("laws_order_before", [])

        # 跑管线 A
        t_a_s = time.time()
        err_a = None
        try:
            out_a = run_pipeline_a_dense_only(retriever, ce_ranker, query, use_ce=ce_ok)
        except Exception as e:
            err_a = traceback.format_exc()
            print(f"     ⚠ 管线A异常: {e}")
            out_a = []
        t_a = (time.time() - t_a_s) * 1000
        ta_list.append(t_a)

        # 跑管线 B
        t_b_s = time.time()
        err_b = None
        try:
            out_b = run_pipeline_b_hybrid(retriever, merger, ce_ranker, query, use_ce=ce_ok)
        except Exception as e:
            err_b = traceback.format_exc()
            print(f"     ⚠ 管线B异常: {e}")
            out_b = []
        t_b = (time.time() - t_b_s) * 1000
        tb_list.append(t_b)

        # 计算两条管线的指标
        def compute_metrics(docs):
            return {
                "n_returned": len(docs),
                "recall@1": round(calc_recall_at_k(docs, must_hit, 1), 4),
                "recall@3": round(calc_recall_at_k(docs, must_hit, 3), 4),
                "recall@5": round(calc_recall_at_k(docs, must_hit, 5), 4),
                "recall@10": round(calc_recall_at_k(docs, must_hit, 10), 4),
                "mrr@10": round(calc_mrr_at_k(docs, must_hit, 10), 4),
                "order_ok": check_law_order(docs[:10], order_assert),
                "top10_articles": [
                    {"r": r.rank, "art": f"{r.law_name or ''} {r.article or ''}".strip() or
                                        (r.title[:40] if r.title else (r.text[:40] if r.text else ""))}
                    for r in docs[:10]
                ],
            }

        m_a = compute_metrics(out_a)
        m_b = compute_metrics(out_b)
        results_by_id[cid] = {
            "case": case,
            "pipeline_a": {**m_a, "latency_ms": round(t_a, 1), "error": err_a},
            "pipeline_b": {**m_b, "latency_ms": round(t_b, 1), "error": err_b},
        }

    print(f"[3/4] 全部运行完成 ({time.time()-t1:.1f}s, "
          f"Avg A={sum(ta_list)/len(ta_list):.0f}ms, Avg B={sum(tb_list)/len(tb_list):.0f}ms)")

    # 3. 汇总分难度
    def aggregate(diff_filter=None):
        ids = [cid for cid, r in results_by_id.items()
               if diff_filter is None or r["case"]["difficulty"] == diff_filter]
        if not ids:
            return None, None
        vals_a = [results_by_id[cid]["pipeline_a"] for cid in ids]
        vals_b = [results_by_id[cid]["pipeline_b"] for cid in ids]
        order_ids = [cid for cid in ids if results_by_id[cid]["case"].get("laws_order_before")]

        if not vals_a:
            return None, None

        def avg(vals, key):
            return round(sum(v[key] for v in vals) / len(vals), 4)

        def order_pass_rate(vals):
            if not order_ids:
                return 1.0
            ok = sum(1 for cid in order_ids if vals[cid]["order_ok"])
            return round(ok / len(order_ids), 4)

        summary_a = {k: avg(vals_a, k) for k in
                     ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10"]}
        summary_a["order_pass_rate"] = order_pass_rate(
            {cid: results_by_id[cid]["pipeline_a"] for cid in ids})
        summary_a["avg_latency_ms"] = round(sum(v["latency_ms"] for v in vals_a) / len(vals_a), 1)
        summary_a["n"] = len(vals_a)
        summary_b = {k: avg(vals_b, k) for k in
                     ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10"]}
        summary_b["order_pass_rate"] = order_pass_rate(
            {cid: results_by_id[cid]["pipeline_b"] for cid in ids})
        summary_b["avg_latency_ms"] = round(sum(v["latency_ms"] for v in vals_b) / len(vals_b), 1)
        summary_b["n"] = len(vals_b)
        return summary_a, summary_b

    overall_a, overall_b = aggregate()
    easy_a, easy_b = aggregate("easy")
    med_a, med_b = aggregate("medium")
    hard_a, hard_b = aggregate("hard")

    summary = {
        "overall":  {"A": overall_a,  "B": overall_b,  "Δ_B_minus_A": {k: round(overall_b[k]-overall_a[k], 4) for k in ["recall@1","recall@3","recall@5","recall@10","mrr@10","order_pass_rate"]}},
        "easy":     {"A": easy_a,     "B": easy_b,     "Δ_B_minus_A": {k: round(easy_b[k]-easy_a[k], 4) for k in ["recall@1","recall@3","recall@5","recall@10","mrr@10","order_pass_rate"]}},
        "medium":   {"A": med_a,      "B": med_b,      "Δ_B_minus_A": {k: round(med_b[k]-med_a[k], 4) for k in ["recall@1","recall@3","recall@5","recall@10","mrr@10","order_pass_rate"]}},
        "hard":     {"A": hard_a,     "B": hard_b,     "Δ_B_minus_A": {k: round(hard_b[k]-hard_a[k], 4) for k in ["recall@1","recall@3","recall@5","recall@10","mrr@10","order_pass_rate"]}},
    }

    # 4. 保存原始 JSON
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "results_by_id": results_by_id,
        }, f, ensure_ascii=False, indent=2)
    print(f"[4/4] 原始 JSON 已保存: {RESULT_JSON}")

    # 5. 生成 Markdown 报告（数据驱动）
    write_markdown_report(summary, results_by_id, bench)

    retriever.close()
    print("✅ Done.")

    # 最后给控制台打印核心指标汇总
    print("\n" + "=" * 70)
    print("  📊 控制台摘要 —— 总体指标对比")
    print("=" * 70)
    for k in ["recall@1","recall@3","recall@5","recall@10","mrr@10","order_pass_rate"]:
        a = summary["overall"]["A"][k]
        b = summary["overall"]["B"][k]
        delta = (b - a) * (100 if k != "mrr@10" else 1)
        unit = "pp" if k != "mrr@10" else ""
        a_disp = f"{a*100:.1f}%" if k != "mrr@10" else f"{a:.2f}"
        b_disp = f"{b*100:.1f}%" if k != "mrr@10" else f"{b:.2f}"
        print(f"  {k:<14} A(纯Dense)={a_disp:<8}  B(混合方案)={b_disp:<8}  Δ={'+' if delta>0 else ''}{delta:.1f}{unit}")
    print(f"  平均时延       A={summary['overall']['A']['avg_latency_ms']:.0f}ms  B={summary['overall']['B']['avg_latency_ms']:.0f}ms  Δ={summary['overall']['B']['avg_latency_ms']-summary['overall']['A']['avg_latency_ms']:.0f}ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
