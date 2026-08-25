"""
阶段一：前置路由与意图解析 + 阶段二：查询改写策略选择

阶段一（三层路由）：
  Layer 1: 规则引擎（<1ms）—— 问候语/法律关键词正则
  Layer 2: （预留）轻量分类器 —— BERT-tiny 二分类
  Layer 3: LLM 兜底（<5% 请求触发）—— 规则未命中时

阶段二（查询改写）：
  Simple → 不改写
  Medium → Multi-Query（2个同义变体）
  Complex → 子问题分解
  HyDE → 独立闭环（不在策略矩阵中）

输出：
  RoutingDecision = {go_to_rag, target_collections, complexity, strategy}
  QuerySet = [{text, source_type}]
"""
import re
import hashlib
import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from ..core.config import (
    LEGAL_KEYWORDS_PATTERN,
    GREETING_PATTERNS, GREETING_REPLIES,
    QUERY_REWRITE_SIMPLE_THRESHOLD, QUERY_REWRITE_MEDIUM_THRESHOLD,
    BANNED_PATTERNS,
)
from ..core.logger import logger


# ============================================================
# 数据类
# ============================================================

@dataclass
class RoutingDecision:
    """阶段一路由决策"""
    go_to_rag: bool = True
    target_collections: List[str] = field(default_factory=lambda: ["laws", "non_laws"])
    complexity: float = 0.5         # 0.0(Simple) ~ 1.0(Complex)
    strategy: str = "simple"        # simple / medium / complex
    direct_reply: Optional[str] = None  # 非 RAG 直接回复
    layer_used: str = "rule"        # rule / classifier / llm


@dataclass
class QueryVariant:
    """单个候选 Query"""
    text: str
    source_type: str  # original / synonym / sub_question
    weight: float = 1.0


# ============================================================
# 阶段一：前置路由
# ============================================================

class IntentRouter:
    """
    阶段一：前置路由与意图解析

    Layer 1: 规则引擎（<1ms）
        - 问候语正则匹配 → go_to_rag=false, 直接返回
        - 法律关键词命中 → go_to_rag=true, 复杂度初步判断
    Layer 2: （预留）轻量分类器 —— 待标注数据后实现
    Layer 3: LLM 兜底 —— 仅规则未命中时调用

    设计原则：
    - 规则引擎处理 90%+ 请求，延迟 < 1ms
    - 分类器是可插拔的（当前跳过，直接走 LLM 兜底）
    - LLM 兜底仅在规则未命中时触发
    """

    # 法律关键词正则（编译一次，全局复用）
    _legal_pattern = re.compile(LEGAL_KEYWORDS_PATTERN)
    _greeting_patterns = [re.compile(p) for p in GREETING_PATTERNS]

    @classmethod
    def route(cls, query: str, llm_client=None) -> RoutingDecision:
        """
        主路由入口

        Args:
            query: 用户原始输入
            llm_client: LLM 客户端（仅 layer 3 需要）

        Returns:
            RoutingDecision 路由决策
        """
        query = query.strip()

        # === Layer 1: 规则引擎 ===
        decision = cls._rule_engine(query)
        if decision:
            decision.layer_used = "rule"
            return decision

        # Layer 2: 轻量分类器 —— 暂未实现，直接进入 Layer 3
        # TODO: 积累标注数据后训练 BERT-tiny 分类器

        # === Layer 3: LLM 兜底 ===
        if llm_client:
            decision = cls._llm_fallback(query, llm_client)
            decision.layer_used = "llm"
            return decision

        # 无 LLM 可用时，默认走 RAG
        return RoutingDecision(
            go_to_rag=True,
            target_collections=["laws", "non_laws"],
            complexity=0.5,
            strategy="simple",
            layer_used="rule",
        )

    @classmethod
    def _rule_engine(cls, query: str) -> Optional[RoutingDecision]:
        """Layer 1: 规则引擎"""

        # 1.1 问候语检测
        for i, pat in enumerate(cls._greeting_patterns):
            if pat.match(query):
                type_key = ["greeting", "thanks", "bye", "presence"][i] if i < 4 else "greeting"
                return RoutingDecision(
                    go_to_rag=False,
                    direct_reply=GREETING_REPLIES.get(type_key, GREETING_REPLIES["greeting"]),
                )

        # 1.2 法律关键词检测
        if cls._legal_pattern.search(query):
            # 根据查询长度和结构判断复杂度
            complexity = cls._estimate_complexity(query)
            return RoutingDecision(
                go_to_rag=True,
                target_collections=["laws", "non_laws"],
                complexity=complexity,
                strategy=cls._complexity_to_strategy(complexity),
            )

        return None

    @classmethod
    def _estimate_complexity(cls, query: str) -> float:
        """
        快速估算查询复杂度（规则引擎内部）

        判定维度：
        - 长度（越长越复杂）
        - 问号数量（多问号 = 子问题）
        - 法律术语密度（高密度可能表明复杂问题）
        - 数字出现（金额/年限/天数 = 涉及具体计算）

        Returns:
            0.0(Simple) ~ 1.0(Complex)
        """
        score = 0.0

        # 长度因子（50字以下 → <0.3, 100字以上 → >0.6）
        length = len(query)
        if length > 100:
            score += 0.4
        elif length > 50:
            score += 0.2
        elif length < 20:
            score += 0.0
        else:
            score += 0.1

        # 问号数量（多问号 = 多子问题）
        question_count = query.count("？") + query.count("?")
        if question_count >= 3:
            score += 0.4
        elif question_count >= 2:
            score += 0.2
        elif question_count == 1:
            score += 0.0

        # 数字出现（金额/年限 = 计算型问题更复杂）
        digit_count = len(re.findall(r'\d+', query))
        if digit_count >= 3:
            score += 0.2
        elif digit_count >= 1:
            score += 0.1

        return min(max(score, 0.0), 1.0)

    @classmethod
    def _complexity_to_strategy(cls, complexity: float) -> str:
        """复杂度映射到改写策略"""
        if complexity < QUERY_REWRITE_SIMPLE_THRESHOLD:
            return "simple"
        elif complexity < QUERY_REWRITE_MEDIUM_THRESHOLD:
            return "medium"
        else:
            return "complex"

    @classmethod
    def _llm_fallback(cls, query: str, llm_client) -> RoutingDecision:
        """Layer 3: LLM 兜底路由"""
        prompt = f"""你是一个法律问题路由助手。判断用户输入是否需要检索法律条文。

用户输入："{query}"

输出 JSON 格式（只输出 JSON，不要其他内容）：
{{
    "go_to_rag": true/false,
    "target_collections": ["laws", "non_laws"],
    "complexity": "Simple" 或 "Complex"
}}

规则：
- 如果用户问的是劳动法相关问题（劳动合同、工资、加班、工伤、辞退等），go_to_rag=true
- 如果只是闲聊/问候，go_to_rag=false
- 如果问题需要多步骤推理或多条款引用，complexity="Complex"
- 否则 complexity="Simple"
"""

        try:
            response = llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            # 提取 JSON
            json_match = re.search(r'\{[^}]+\}', response.strip())
            if json_match:
                data = json.loads(json_match.group())
                complexity_str = data.get("complexity", "Simple")
                complexity = 0.8 if complexity_str == "Complex" else 0.3
                return RoutingDecision(
                    go_to_rag=data.get("go_to_rag", True),
                    target_collections=data.get("target_collections", ["laws", "non_laws"]),
                    complexity=complexity,
                    strategy=cls._complexity_to_strategy(complexity),
                )
        except Exception as e:
            logger.warning(f"[路由] LLM 兜底解析失败，使用默认路由: {e}")

        return RoutingDecision(
            go_to_rag=True,
            target_collections=["laws", "non_laws"],
            complexity=0.5,
            strategy="simple",
        )


# ============================================================
# 阶段二：查询改写
# ============================================================

class QueryRewriter:
    """
    阶段二：查询改写策略选择

    策略矩阵：
      Simple  → 不改写 [原始Query]
      Medium  → Multi-Query [原始, 同义1, 同义2]
      Complex → 子问题分解 [原始, 子问题1, 子问题2, ...]

    HyDE 不在策略矩阵中——它是独立的二次检索闭环
    只在阶段五 CE Top-1 < 0.4 时触发

    关键原则：
      1. 原始 Query 绝对保留，作为黄金兜底
      2. 每个改写 Query 标注来源类型
    """

    @staticmethod
    def rewrite(query: str, strategy: str, llm_client=None) -> Dict:
        """
        根据策略生成候选 Query 集

        Args:
            query: 用户原始输入
            strategy: simple / medium / complex
            llm_client: LLM 客户端（Multi-Query 或子问题分解时需要）

        Returns:
            {
                "queries": [QueryVariant, ...],
                "strategy": str,
                "original": str,
            }
        """
        if strategy == "simple":
            return QueryRewriter._simple_rewrite(query)

        if strategy == "medium" and llm_client:
            return QueryRewriter._multi_query_rewrite(query, llm_client)

        if strategy == "complex" and llm_client:
            return QueryRewriter._sub_question_rewrite(query, llm_client)

        # 降级：无法调用 LLM 时退回 simple
        return QueryRewriter._simple_rewrite(query)

    @staticmethod
    def _simple_rewrite(query: str) -> Dict:
        """Simple 策略：仅原始 Query"""
        return {
            "queries": [
                QueryVariant(text=query, source_type="original", weight=1.0),
            ],
            "strategy": "simple",
            "original": query,
            "sub_questions": [],
        }

    @staticmethod
    def _multi_query_rewrite(query: str, llm_client) -> Dict:
        """Medium 策略：生成 2 个同义变体"""
        prompt = f"""你是一个查询改写助手。用户查询涉及劳动法律，请生成 2 个语义相同但不同表述的同义查询。

原始查询："{query}"

输出 JSON 数组（只输出 JSON，不要其他内容）：
["同义查询1", "同义查询2"]

要求：
- 保持原始查询的法律含义不变
- 使用不同的措辞和表达方式
- 不要添加原始查询中没有的信息
"""

        try:
            response = llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300,
            )
            variants = json.loads(response.strip())
            if isinstance(variants, list) and 1 <= len(variants) <= 3:
                queries = [QueryVariant(text=query, source_type="original", weight=1.2)]
                for v in variants[:2]:
                    queries.append(QueryVariant(text=v, source_type="synonym", weight=0.9))
                return {
                    "queries": queries,
                    "strategy": "medium",
                    "original": query,
                    "synonyms": variants[:2],
                    "sub_questions": [],
                }
        except Exception as e:
            logger.warning(f"[改写] Multi-Query 失败，退回 simple: {e}")

        return QueryRewriter._simple_rewrite(query)

    @staticmethod
    def _sub_question_rewrite(query: str, llm_client) -> Dict:
        """Complex 策略：子问题分解"""
        prompt = f"""你是一个法律问题分析助手。用户提出了一个复杂的劳动法问题，请将其拆分为 2~5 个可以独立检索的子问题。

用户问题："{query}"

输出 JSON 数组（只输出 JSON，不要其他内容）：
["子问题1", "子问题2", "子问题3"]

要求：
- 每个子问题应该是独立的、可以直接检索的
- 子问题之间不要有依赖关系（可以并行处理）
- 覆盖用户问题的所有方面
- 用简洁的陈述句表达
"""

        try:
            response = llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            sub_questions = json.loads(response.strip())
            if isinstance(sub_questions, list) and 1 <= len(sub_questions) <= 5:
                queries = [QueryVariant(text=query, source_type="original", weight=1.5)]
                for sq in sub_questions:
                    queries.append(QueryVariant(text=sq, source_type="sub_question", weight=0.8))
                return {
                    "queries": queries,
                    "strategy": "complex",
                    "original": query,
                    "sub_questions": sub_questions,
                }
        except Exception as e:
            logger.warning(f"[改写] 子问题分解失败，退回 simple: {e}")

        return QueryRewriter._simple_rewrite(query)


# ============================================================
# HyDE 独立闭环（阶段五之后触发，不在此模块）
# ============================================================

class HyDERewriter:
    """
    HyDE（假设文档嵌入）—— 独立二次检索闭环

    触发条件：阶段五 CE 精排后 Top-1 < 0.4

    流程：
      Low CE score → LLM 生成假设文档 → BGE embedding →
      作为额外 Query 重新 ANN 检索 → 结果合并到候选池 → 重新走 RRF + CE

    幻觉控制：生成的假设文档如果包含具体法条号，需要标记为不确定
    """

    @staticmethod
    def generate_hypothetical_document(query: str, llm_client) -> Tuple[Optional[str], bool]:
        """
        生成假设文档

        Args:
            query: 原始查询
            llm_client: LLM 客户端

        Returns:
            (假设文档, has_uncertain_articles)
            has_uncertain_articles=True 表示文档中包含具体法条号，需要验证
        """
        prompt = f"""你是一个劳动法专家。请根据以下用户问题，写一段约 200 字的"假设性法律分析"，模拟法律条文应该如何回答这个问题。

用户问题："{query}"

要求：
- 用学术性语言写，像法律条文一样结构化
- 内容基于你对劳动法的了解
- 不要引用具体的法条号（如"第X条"），用"相关法律规定"代替
- 最后加上一句："以上为假设性分析，具体法条需从知识库检索确认"

直接输出假设文档文本，不要 JSON。"""

        try:
            response = llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=400,
            )

            # 幻觉检测：检查是否包含具体法条号
            has_articles = bool(re.search(r'第[一二三四五六七八九十百千\d]+条', response))

            return response.strip(), has_articles

        except Exception as e:
            logger.error(f"[HyDE] 生成假设文档失败: {e}")
            return None, False


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 测试规则引擎
    print("=== 测试阶段一：路由 ===")
    test_queries = [
        "你好",
        "谢谢",
        "公司拖欠我3个月工资，我可以辞职并要求赔偿吗",
        "今天天气怎么样",
    ]
    for q in test_queries:
        decision = IntentRouter.route(q)
        print(f"\n  Query: {q}")
        print(f"    go_to_rag: {decision.go_to_rag}")
        print(f"    complexity: {decision.complexity:.2f}")
        print(f"    strategy: {decision.strategy}")
        print(f"    direct_reply: {decision.direct_reply}")

    print("\n=== 测试阶段二：改写 ===")
    result = QueryRewriter.rewrite("公司拖欠工资，我想辞职并要求赔偿", strategy="simple")
    print(f"  Simple: {len(result['queries'])} 个 Query")
    for qv in result['queries']:
        print(f"    [{qv.source_type}] {qv.text[:60]}...")