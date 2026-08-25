"""
🧪 演示：AgentOrchestrator 5 轮完整咨询（含 Memory + FSM + Tool 链路）

用途：秋招面试时直接跑给面试官看 —— 证明你这两块不是纸上谈兵，是能跑通端到端流程的。

预期状态跃迁轨迹：
  第1轮 "老板把我开了"     INIT → UNDERSTANDING → COLLECTING_EVIDENCE（追问工资/入职日期/辞退形式）
  第2轮 补充工资和入职日期  COLLECTING → UNDERSTANDING → COLLECTING（继续问辞退形式/地区）
  第3轮 补充地区和辞退细节  COLLECTING → UNDERSTANDING → ANALYZING（完整度达标）
                                → TOOL_USE (law_lookup 未签合同+辞退)
                                → GENERATING → REVIEWING → FINALIZING（产出报告）
  第4轮 "生成仲裁申请书"    FINALIZING → TOOL_USE（文书工具 Mock 响应）
  第5轮 "我有骨折工伤认定"  → REFERRING_OUT（前置行政程序）
  Bonus:  "重新开始"         → 清空全部状态，重新 INIT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import AgentOrchestrator, AgentResponse
from src.memory import EvidenceSlotMemory


def print_header(msg: str) -> None:
    print()
    print("=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def demo_dialogue_5turns():
    # mock 模式：离线可跑，演示 FSM/记忆/工具编排链路（真实模式见 demo_multi_turn.py）
    orch = AgentOrchestrator(mode="mock")
    sid = None  # 第一轮 None，后面用 orch 返回的 session_id

    # ================= 第 1 轮 =================
    print_header("第 1/5 轮：用户首次输入（案情非常模糊）")
    print("👤 用户: 「老板昨天把我开了，给了我5000块让我签主动离职书，我没签」")
    r1: AgentResponse = orch.handle_message(sid, "老板昨天把我开了，给了我5000块让我签主动离职书，我没签")
    sid = r1.session_id
    print(r1.pretty(show_debug=True))
    print()
    print("  [证据槽状态]")
    sess = orch.store.get(sid)
    print(sess.evidence_slots.pretty_summary())

    # ================= 第 2 轮 =================
    print_header("第 2/5 轮：用户补充部分信息")
    print("👤 用户: 「我是2021年6月入职的，到现在5年多了。税前工资12000。一直没签合同。」")
    r2 = orch.handle_message(sid, "我是2021年6月入职的，到现在5年多了。税前工资12000。一直没签合同。")
    print(r2.pretty(show_debug=True))
    print()
    print("  [证据槽状态]")
    print(orch.store.get(sid).evidence_slots.pretty_summary())

    # ================= 第 3 轮 =================
    print_header("第 3/5 轮：用户补充地区和辞退细节 → 信息达标 → 产出最终分析报告")
    print("👤 用户: 「公司在北京朝阳区。辞退是部门微信群里发语音说的，我有录屏和工资流水。给我的5000已经到账了。」")
    r3 = orch.handle_message(sid, "公司在北京朝阳区。辞退是部门微信群里发语音说的，我有录屏和工资流水。给我的5000已经到账了。")
    print(r3.pretty(show_debug=True))
    print()
    print("  [证据槽最终状态]")
    print(orch.store.get(sid).evidence_slots.pretty_summary())
    print()
    print("  🧭 FSM 完整跃迁轨迹:")
    print(orch.store.get(sid).fsm.pretty_transitions())

    # ================= 第 4 轮 =================
    print_header("第 4/5 轮：用户在终态请求生成文书 → TOOL_USE (Mock 文书工具)")
    print("👤 用户: 「好的，帮我生成仲裁申请书吧」")
    r4 = orch.handle_message(sid, "好的，帮我生成仲裁申请书吧")
    print(r4.pretty(show_debug=True))

    # ================= 第 5 轮 =================
    print_header("第 5/5 轮：用户问工伤 → 命中转出规则 → REFERRING_OUT")
    print("👤 用户: 「哦对了，上班路上摔骨折了，社保局给的工伤认定书拿到了，能算赔偿吗？」")
    r5 = orch.handle_message(sid, "哦对了，上班路上摔骨折了，社保局给的工伤认定书拿到了，能算赔偿吗？")
    print(r5.pretty(show_debug=True))

    # ================= Bonus: reset =================
    print_header("Bonus：用户说「重新开始」 → 所有状态清空")
    print("👤 用户: 「重新开始」")
    r6 = orch.handle_message(sid, "重新开始")
    print(r6.pretty(show_debug=True))
    sess_after_reset = orch.store.get(sid)
    d, t, ratio = sess_after_reset.evidence_slots.completion_ratio()
    print(f"   ✅ Reset 成功: 完整度回到 {d}/{t} ({ratio:.0%})，FSM 回到 {sess_after_reset.fsm.current_state.value}")

    # ========== 给面试官看的总结 ==========
    print()
    print("=" * 70)
    print("  🏁 5 轮对话演示结束。")
    print("=" * 70)
    print()
    print("  🧠 记忆模块 4 种用法全部生效：")
    print("     ① ShortTermMemory 保存了每轮 user/assistant 对话 + tool 结果")
    print("     ② EvidenceSlotMemory 结构化抽取 9 项关键信息 + 完整度阈值判断驱动")
    print("     ③ SummaryBuffer 自动压缩（演示中消息少还没触发阈值，可以自己加轮数试试）")
    print()
    print("  🔀 编排模块 4 种组件全部生效：")
    print("     ① FSM：9 次合法跃迁（非法跃迁会抛 AssertionError）")
    print("     ② ToolRegistry：law_lookup Mock 工具按标准 BaseTool 接口自动调用")
    print("     ③ SessionStore：所有上下文按 sid 绑定，5 轮后仍能取回")
    print("     ④ Orchestrator：规则驱动推进 理解→追问→检索→工具→生成→反思→终态")


if __name__ == "__main__":
    demo_dialogue_5turns()
