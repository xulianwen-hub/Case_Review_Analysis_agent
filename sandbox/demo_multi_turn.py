"""
🧪 演示：多轮对话编排（真实组件版）

流程：
    第1轮 用户模糊描述 → UNDERSTANDING → COLLECTING（LLM/正则抽取证据槽，追问缺失项）
    第2轮 补充工资/入职/合同 → 继续追问
    第3轮 补充地区/辞退形式 → 完整度达标 → ANALYZING → TOOL_USE（真实 RAG）
                                   → GENERATING（8 模块报告）→ REVIEWING → FINALIZING
    第4轮 说「重新开始」 → 全部重置

用法：
    python sandbox/demo_multi_turn.py            # 真实模式（LLM 走 DeepSeek，失败自动降级）
    python sandbox/demo_multi_turn.py --mock     # 纯离线演示（不调 LLM / 不加载模型）
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.state import AgentOrchestrator, AgentResponse


def print_header(msg: str) -> None:
    print()
    print("=" * 70)
    print(f"  {msg}")
    print("=" * 70)


def show(orch, r: AgentResponse) -> None:
    print(r.pretty(show_debug=True))
    sess = orch.store.get(r.session_id)
    if sess is not None:
        print()
        print(sess.evidence_slots.pretty_summary())


def main():
    parser = argparse.ArgumentParser(description="多轮对话编排 demo")
    parser.add_argument("--mock", action="store_true", help="使用 mock 组件（离线）")
    args = parser.parse_args()

    mode = "mock" if args.mock else "auto"
    print_header(f"启动 AgentOrchestrator（mode={mode}）—— 真实模式首次分析需加载 RAG 模型（10-20s）")
    orch = AgentOrchestrator(mode=mode)
    sid = None

    # ============ 第 1 轮 ============
    print_header("第 1/4 轮：用户首次输入（案情模糊）")
    msg = "老板昨天把我开了，给了我5000块让我签主动离职书，我没签"
    print("👤 用户:", msg)
    r1 = orch.handle_message(sid, msg)
    sid = r1.session_id
    show(orch, r1)

    # ============ 第 2 轮 ============
    print_header("第 2/4 轮：用户补充部分信息")
    msg = "我是2021年6月入职的，到现在5年多了。税前工资12000。一直没签合同。有工资流水。"
    print("👤 用户:", msg)
    r2 = orch.handle_message(sid, msg)
    show(orch, r2)

    # ============ 第 3 轮 ============
    print_header("第 3/4 轮：用户补充地区与辞退细节 → 触发真实分析")
    msg = "公司在北京朝阳区。辞退是部门微信群里发语音说的，我有录屏。"
    print("👤 用户:", msg)
    r3 = orch.handle_message(sid, msg)
    show(orch, r3)
    sess = orch.store.get(sid)
    print("\n  🧭 FSM 完整跃迁轨迹:")
    print(sess.fsm.pretty_transitions())

    # ============ 第 4 轮 ============
    print_header("第 4/4 轮：重新开始")
    r4 = orch.handle_message(sid, "重新开始")
    print(r4.pretty(show_debug=True))

    orch.close()


if __name__ == "__main__":
    main()
