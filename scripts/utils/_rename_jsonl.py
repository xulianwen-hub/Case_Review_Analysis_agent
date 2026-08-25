"""将 JSONL 格式的 .json 文件重命名为 .jsonl，解决 VS Code 红色波浪线问题"""
import os

BASE_DIR = r"d:\agent_develp\data\raw"
DIRS = ["cases", "judgments", "regulations"]

for d in DIRS:
    dir_path = os.path.join(BASE_DIR, d)
    if not os.path.isdir(dir_path):
        continue
    for fname in os.listdir(dir_path):
        if fname.endswith(".json"):
            old = os.path.join(dir_path, fname)
            new = os.path.join(dir_path, fname[:-5] + ".jsonl")
            os.rename(old, new)
            print(f"Renamed: {os.path.basename(old)} -> {os.path.basename(new)}")

print("Done")