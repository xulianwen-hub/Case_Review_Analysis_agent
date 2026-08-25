"""
增量法条录入脚本（兼容入口，委托给 src.ingest）
- 用法: python scripts/ingest.py <新法条DOCX路径>
- 自动解析 → 切分 → 写入 SQLite + Milvus
- 不重建已有数据库，仅增量新增
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import ingest_law


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/ingest.py <新法条DOCX文件路径>")
        print("     python scripts/ingest.py data/raw/laws/新法条.docx")
        sys.exit(1)

    docx_path = sys.argv[1]
    result = ingest_law(docx_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()