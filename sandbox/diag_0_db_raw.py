"""纯原生 SQLite 直接查表，完全不碰 Milvus/RAG 封装，避免 Python 解释器崩"""
import os, sys, sqlite3, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = str(ROOT / "sandbox" / "diag_direct_db.txt")
# 手动写日志，绝对不依赖 stdout
def log(s):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(s) + "\n")

# 清空上次
try: open(LOG, "w").close()
except: pass

log("Step 1: 打开 chunks.db")
try:
    conn = sqlite3.connect(str(ROOT / "data" / "processed" / "chunks.db"))
    cur = conn.cursor()
except Exception as e:
    log(f"FAIL: {e}")
    sys.exit(1)
log("OK")

log("\nStep 2: 查所有表名")
try:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cur.fetchall()
    log(f"tables = {tables}")
except Exception as e:
    log(f"FAIL: {e}")

log("\nStep 3: 查 chunks 表 schema")
try:
    cur.execute("PRAGMA table_info(chunks)")
    cols = cur.fetchall()
    for c in cols:
        log(f"  col: {c}")
except Exception as e:
    log(f"FAIL chunks schema: {e}")

log("\nStep 4: 从 chunks 表随机抽 10 条，看 law_name/article/chapter 列真实内容")
try:
    cur.execute("SELECT chunk_id, law_name, article, chapter, title, substr(text,1,120) FROM chunks WHERE law_name IS NOT NULL LIMIT 10")
    rows = cur.fetchall()
    log(f"共抽到 {len(rows)} 条")
    for i, r in enumerate(rows, 1):
        log(f"\n  [{i}] chunk_id = {r[0]}")
        log(f"       law_name = {r[1]!r}")
        log(f"       article  = {r[2]!r}")
        log(f"       chapter  = {r[3]!r}")
        log(f"       title    = {r[4]!r}")
        log(f"       text[:120]= {r[5]!r}")
except Exception as e:
    log(f"FAIL select 10: {e}")
    import traceback
    log(traceback.format_exc())

conn.close()
log("\nDONE")
print("END, 请查看:", LOG)