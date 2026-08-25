"""
src.state.extractor —— 真实 LLM 案情证据槽抽取器

把用户自然语言描述 → 结构化证据槽（EvidenceSlotMemory）：
    1. 优先调用真实 LLM，要求只输出 JSON（槽 key -> 值/None）
    2. 解析失败 / LLM 不可用时，降级到正则抽取（离线可跑）

设计要点：
    · 只接受 STANDARD_SLOTS 中已知的 key，防止 LLM 幻觉出无关字段
    · 值做类型归一化（月薪"1.2万" → 12000）
    · 抽取结果统一走 evidence_slots.batch_update()，state 可追溯
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from ..core.logger import logger
from ..memory import EvidenceSlotMemory


# ============================================================
# 工具函数：从 LLM 回复中稳健地解析 JSON
# ============================================================
def parse_json_response(text: str) -> Optional[dict]:
    """从 LLM 回复中提取第一个合法 JSON 对象（容忍 markdown 代码块/前后废话）"""
    if not text:
        return None
    raw = text.strip()
    # 去掉 ```json ... ``` 代码块
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    # 定位第一个 '{'，用括号配对找结尾
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


# ============================================================
# 值归一化辅助
# ============================================================
def _to_money(value: Any) -> Optional[int]:
    """把 '1.2万' / '12000元' / 12000 归一化为整数元"""
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        return None
    s = value.strip().replace(",", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(万|w|千|k|元|块)?", s, re.IGNORECASE)
    if not m:
        return None
    try:
        num = float(m.group(1))
    except ValueError:
        return None
    unit = (m.group(2) or "").lower()
    if unit in ("万", "w"):
        num *= 10000
    elif unit in ("千", "k"):
        num *= 1000
    if not (0 < num < 10_000_000):
        return None
    return int(num)


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s in ("是", "有", "签了", "签过", "true", "True", "yes", "1"):
            return True
        if s in ("否", "没", "没有", "未签", "没签", "false", "False", "no", "0"):
            return False
    return None


def normalize_extracted(updates: dict) -> dict:
    """按槽位类型做归一化，丢掉无法识别的值"""
    result = {}
    money_keys = {"monthly_salary", "already_compensation"}
    bool_keys = {"signed_contract", "salary_proof"}
    for k, v in updates.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if k in money_keys:
            nv = _to_money(v)
            if nv is not None:
                result[k] = nv
        elif k in bool_keys:
            nv = _to_bool(v)
            if nv is not None:
                result[k] = nv
        else:
            result[k] = v
    return result


# ============================================================
# 正则兜底抽取（离线 / LLM 不可用时的降级路径）
# ============================================================
def regex_extract_slots(user_text: str) -> dict:
    """
    从自由文本中抽取常见证据槽（规则版，源自 V1 mock）。
    覆盖：入职/离职日期、月薪、合同、城市、辞退形式与原因、已付补偿、工资证明。
    """
    text = user_text
    result: Dict[str, Any] = {}

    # --- 日期 ---
    dates = list(re.finditer(r"(20\d{2})[年\-/.](\d{1,2})月?", text))
    if dates:
        result["employment_start"] = f"{dates[0].group(1)}-{int(dates[0].group(2)):02d}"
        if len(dates) >= 2:
            m2 = dates[-1]
            result["employment_end"] = f"{m2.group(1)}-{int(m2.group(2)):02d}"

    # --- 月薪 ---
    has_salary_context = any(k in text for k in ("工资", "月薪", "薪资", "税前", "税后", "应发", "实发", "收入", "到手"))
    if has_salary_context:
        p = re.compile(r"(到手|税前|应发|实发)?\s*(?:工资|月薪|薪资)?\s*"
                       r"([0-9]{3,6})\s*(?:元|块)?")
        salary_keyword_indices = [text.find(k) for k in ("工资", "税前", "税后", "月薪", "薪资", "收入", "到手", "应发", "实发") if k in text]
        best_val, best_dist = None, 10 ** 9
        for m in p.finditer(text):
            try:
                v = int(m.group(2))
            except (ValueError, TypeError):
                continue
            if not (3000 <= v <= 150000):
                continue
            lo, hi = max(0, m.start(2) - 10), min(len(text), m.end(2) + 8)
            ctx = text[lo:hi]
            if any(k in ctx for k in ("给我", "给了我", "到账", "补偿", "遣散", "赔偿", "先付", "预付")):
                continue
            d = min(abs(m.start(2) - i) for i in salary_keyword_indices)
            if d < best_dist:
                best_dist, best_val = d, v
        if best_val is not None:
            result["monthly_salary"] = best_val

    # --- 合同 ---
    if ("没签" in text or "未签" in text or "没有签" in text) and "合同" in text:
        result["signed_contract"] = False
    elif "签了合同" in text or "有合同" in text or "签过合同" in text:
        result["signed_contract"] = True

    # --- 城市 ---
    for city in ("北京", "深圳", "广州", "杭州", "上海", "苏州", "成都", "南京", "武汉"):
        if city in text:
            result["company_city"] = city
            break

    # --- 解除形式 ---
    if ("口头" in text or "微信" in text or "聊天记录" in text or "群里" in text) and \
            any(k in text for k in ("辞", "开", "离职")):
        result["termination_form"] = "口头/聊天记录（建议截图+录屏）"
    elif ("书面" in text or "通知书" in text) and "辞" in text:
        result["termination_form"] = "书面解除通知书"

    # --- 解除原因 ---
    if "公司" in text and any(k in text for k in ("经营不善", "效益不好", "裁员")):
        result["termination_reason"] = "公司以经营不善/经济性裁员为由解除"
    elif "老板" in text and any(k in text for k in ("开了", "开除", "辞退")):
        result["termination_reason"] = "口头辞退（疑似违法解除，待进一步确认）"
    elif "主动" in text and any(k in text for k in ("离职", "辞职")):
        result["termination_reason"] = "劳动者主动提出辞职"

    # --- 已付补偿 ---
    m = re.search(r"给了我?\s*([0-9]+)\s*(千|万|元|块)?", text)
    comp_keywords = ("补偿", "遣散", "n+1", "n＋1", "赔偿", "到账", "已经付", "已经给", "打了", "打给我", "给了我")
    if m and any(k in text.lower() or k in text for k in comp_keywords):
        val = int(m.group(1))
        unit = m.group(2) or "元"
        if unit in ("万", "w"):
            val *= 10000
        elif unit == "千":
            val *= 1000
        result["already_compensation"] = val

    # --- 工资证明 ---
    if any(k in text for k in ("流水", "工资条", "转账记录", "工资卡")):
        result["salary_proof"] = True

    return result


# ============================================================
# LLM 证据抽取器（主实现）
# ============================================================
class LLMEvidenceExtractor:
    """
    真实 LLM 结构化抽取：
        extract(user_text, evidence_slots) -> {key: value}

    - llm: src.core.llm_client 的客户端（chat 方法返回文本）
    - 失败时自动降级为 regex_extract_slots
    """

    def __init__(self, llm: Any = None, enabled: bool = True):
        self.llm = llm
        self.enabled = enabled

    # ---------- Prompt 构建 ----------
    @staticmethod
    def _slot_descriptor(slots: List[Any]) -> str:
        lines = []
        for s in slots:
            hint = f"（{s.ask_hint}）" if getattr(s, "ask_hint", "") else ""
            lines.append(f"- {s.key}: {s.label}{hint}")
        return "\n".join(lines)

    def _build_messages(self, user_text: str, slots: List[Any]) -> List[dict]:
        system = (
            "你是劳动纠纷案情信息抽取器。用户会用自然语言描述劳动纠纷，"
            "请从描述中抽取下列槽位的值，只输出 JSON 对象，不要输出任何其他文字。"
            "规则：\n"
            "1. 只输出列出的 key；描述中没提到的槽位值为 null。\n"
            "2. 月薪/已付补偿等金额统一转成数字（单位元），如 1.2万 → 12000。\n"
            "3. signed_contract / salary_proof 输出 true/false。\n"
            "4. 日期格式 YYYY-MM（如 2021-06）。\n"
            "5. termination_reason / termination_form 用一句简短中文概括。"
        )
        user = (
            f"可抽取的槽位：\n{self._slot_descriptor(slots)}\n\n"
            f"用户描述：\n{user_text}\n\n"
            "请输出 JSON："
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ---------- 主入口 ----------
    def extract(self, user_text: str, evidence_slots: EvidenceSlotMemory) -> dict:
        if not self.enabled or self.llm is None:
            return self._fallback(user_text)

        slots = evidence_slots.list_slots()
        try:
            resp = self.llm.chat(
                messages=self._build_messages(user_text, slots),
                temperature=0.1,
                max_tokens=1000,
            )
            parsed = parse_json_response(resp)
            if not isinstance(parsed, dict):
                logger.warning("[证据抽取] LLM 返回无法解析，降级正则抽取")
                return self._fallback(user_text)

            # 只保留已知槽位 + 归一化
            known = {s.key for s in slots}
            cleaned = {k: v for k, v in parsed.items() if k in known}
            normalized = normalize_extracted(cleaned)
            if normalized:
                logger.info(f"[证据抽取] LLM 抽取 {len(normalized)} 个字段: {list(normalized)}")
                return normalized
            logger.info("[证据抽取] LLM 未抽到有效字段，尝试正则兜底")
            return self._fallback(user_text)
        except Exception as e:
            logger.warning(f"[证据抽取] LLM 调用失败，降级正则: {type(e).__name__}: {str(e)[:120]}")
            return self._fallback(user_text)

    # 兼容别名：外部可以注入自定义抽取函数（测试用）
    def set_fallback(self, fn: Callable[[str], dict]) -> None:
        self._custom_fallback = fn

    def _fallback(self, user_text: str) -> dict:
        fn = getattr(self, "_custom_fallback", None)
        return fn(user_text) if fn else regex_extract_slots(user_text)
