"""
循环运行案例向量库嵌入，直到全部完成。
用法: python scripts/run_all_case_embeddings.py
"""
import subprocess
import sys
import os

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BATCH_SIZE = 500
MAX_ROUNDS = 20  # 最多 20 轮，防止死循环


def main():
    for round_idx in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"第 {round_idx} 轮嵌入 (batch={BATCH_SIZE})")
        print(f"{'='*60}")

        # 运行增量嵌入脚本
        result = subprocess.run(
            [sys.executable, "scripts/build_case_milvus_incremental.py",
             "--batch", str(BATCH_SIZE)],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        print(result.stdout)
        if result.stderr:
            print(f"[stderr] {result.stderr[-500:]}")

        # 检查是否完成
        if "[完成]" in result.stdout:
            print("\n✅ 全部嵌入完成！")
            break
        if "[进度]" not in result.stdout:
            print("\n⚠️ 未检测到进度输出，尝试继续...")
            # 如果已经全部嵌入，脚本会直接输出 [完成]

        # 避免过于频繁
        import time
        time.sleep(1)


if __name__ == "__main__":
    main()