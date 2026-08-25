"""
数据管道编排器（支持断点续跑）

用法：
    python -m src.pipeline --full          # 全量构建
    python -m src.pipeline --ingest <docx> # 增量录入
    python -m src.pipeline --resume        # 从上次断点继续
"""
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field

from ..core.config import PROJECT_ROOT, PROCESSED_DIR, LAWS_DIR, VECTORS_DIR
from ..core.logger import logger

CHECKPOINT_FILE = str(PROCESSED_DIR / ".pipeline_checkpoint.json")


@dataclass
class PipelineStep:
    """管道步骤定义"""
    name: str
    fn: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    description: str = ""


class PipelineRunner:
    """
    数据管道编排器

    功能：
    - 按顺序执行多个步骤
    - 每个步骤完成后保存检查点
    - 失败时记录当前进度，支持断点续跑
    - 支持 --resume 从上次断点继续
    - 支持 --force 强制重新执行所有步骤
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.steps: List[PipelineStep] = []
        self._checkpoint_data = self._load_checkpoint()
        self._step_results: Dict[str, any] = {}

    def add_step(self, name: str, fn: Callable, *args,
                 description: str = "", **kwargs) -> "PipelineRunner":
        self.steps.append(PipelineStep(
            name=name, fn=fn, args=args, kwargs=kwargs,
            description=description or name,
        ))
        return self

    def _load_checkpoint(self) -> Dict:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"pipeline": self.name, "completed_steps": [], "failed_step": None,
                "step_results": {}, "started_at": None, "updated_at": None}

    def _save_checkpoint(self):
        self._checkpoint_data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._checkpoint_data["step_results"] = {
            k: str(v)[:200] if not isinstance(v, (str, int, float, bool, list, dict)) else v
            for k, v in self._step_results.items()
        }
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(self._checkpoint_data, f, ensure_ascii=False, indent=2)

    def _clear_checkpoint(self):
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        self._checkpoint_data = {"pipeline": self.name, "completed_steps": [],
                                 "failed_step": None, "step_results": {},
                                 "started_at": None, "updated_at": None}

    def run(self, resume: bool = False, force: bool = False) -> Dict:
        """
        执行管道

        Args:
            resume: 是否从断点继续
            force: 是否强制重新执行所有步骤

        Returns:
            {"status": "ok"|"partial"|"failed", "completed": [...], "failed": ...}
        """
        if force:
            self._clear_checkpoint()
            resume = False

        completed_steps = set(self._checkpoint_data["completed_steps"])
        failed_step = self._checkpoint_data.get("failed_step")

        if resume and failed_step:
            logger.info(f"[断点续跑] 上次失败于: {failed_step}，从该步骤重试")
            completed_steps.discard(failed_step)
            self._checkpoint_data["failed_step"] = None
            self._checkpoint_data["completed_steps"] = list(completed_steps)
            self._save_checkpoint()

        if not self._checkpoint_data.get("started_at"):
            self._checkpoint_data["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        self._checkpoint_data["pipeline"] = self.name
        self._save_checkpoint()

        total = len(self.steps)
        executed = []
        skipped = []
        failed = None

        logger.info("=" * 60)
        logger.info(f"管道执行: {self.name} (共 {total} 步)")
        if resume:
            logger.info(f"模式: 断点续跑 (已完成 {len(completed_steps)}/{total})")
        logger.info("=" * 60)

        for i, step in enumerate(self.steps, 1):
            if step.name in completed_steps:
                logger.info(f"[{i}/{total}] ⏭ 跳过: {step.description} (已完成)")
                skipped.append(step.name)
                continue

            logger.info(f"[{i}/{total}] ▶ 执行: {step.description}")
            step_start = time.time()

            try:
                result = step.fn(*step.args, **step.kwargs)
                elapsed = time.time() - step_start

                self._step_results[step.name] = result
                completed_steps.add(step.name)
                self._checkpoint_data["completed_steps"] = list(completed_steps)
                self._checkpoint_data["failed_step"] = None
                self._save_checkpoint()

                executed.append(step.name)
                logger.info(f"[{i}/{total}] ✅ 完成: {step.description} (耗时 {elapsed:.1f}s)")

            except Exception as e:
                elapsed = time.time() - step_start
                self._checkpoint_data["failed_step"] = step.name
                self._save_checkpoint()

                failed = {"step": step.name, "error": str(e)}
                logger.error(f"[{i}/{total}] ❌ 失败: {step.description} (耗时 {elapsed:.1f}s)")
                logger.error(f"  错误: {type(e).__name__}: {e}")

                import traceback
                traceback.print_exc()

                return {
                    "status": "partial" if executed else "failed",
                    "completed": executed,
                    "skipped": skipped,
                    "failed": failed,
                    "remaining": [s.name for s in self.steps if s.name not in completed_steps],
                    "checkpoint": CHECKPOINT_FILE,
                }

        self._clear_checkpoint()

        logger.info("=" * 60)
        logger.info(f"管道执行完成: {len(executed)} 执行, {len(skipped)} 跳过")
        logger.info("=" * 60)

        return {
            "status": "ok",
            "completed": executed,
            "skipped": skipped,
            "failed": None,
            "step_results": self._step_results,
        }

    @staticmethod
    def get_checkpoint_status() -> Optional[Dict]:
        if not os.path.exists(CHECKPOINT_FILE):
            return None
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)


def build_full_pipeline(law_docx_files: List[str] = None) -> PipelineRunner:
    """
    构建全量数据管道

    步骤：
    1. 解析所有 DOCX 法条
    2. 切分 chunks
    3. 构建 SQLite 文本库
    4. 构建 Milvus 向量库
    """
    from .ingest import IngestManager, _parse_docx, _group_articles_by_section, _split_section_into_chunks

    runner = PipelineRunner(name="full_build")
    ingest_mgr = IngestManager()

    if law_docx_files is None:
        raw_laws_dir = PROJECT_ROOT / "data" / "raw" / "laws"
        if raw_laws_dir.exists():
            law_docx_files = [str(p) for p in raw_laws_dir.glob("*.docx")]
        else:
            law_docx_files = []

    def step_parse_all():
        all_articles = []
        for f in law_docx_files:
            articles = _parse_docx(f)
            all_articles.extend(articles)
            logger.info(f"  解析: {os.path.basename(f)} → {len(articles)} 条")
        return {"count": len(all_articles), "articles": all_articles}

    def step_chunk(articles_data):
        articles = articles_data["articles"]
        sections = _group_articles_by_section(articles)

        child_chunks = []
        parent_chunks = []
        for key, sec in sections.items():
            chunks = _split_section_into_chunks(
                sec["articles"], sec["law_name"], sec["law_short"], sec["chapter"]
            )
            parent_key = f"{sec['law_name']}||{sec['chapter']}"
            parent_id = hashlib.md5(parent_key.encode()).hexdigest()[:12]

            for ci, chunk in enumerate(chunks):
                chunk_id = hashlib.md5(f"{parent_key}|chunk{ci}".encode()).hexdigest()[:12]
                child_chunks.append({
                    "chunk_id": chunk_id, "chunk_type": "child",
                    "law_name": sec["law_name"], "law_short": sec["law_short"],
                    "chapter": sec["chapter"], "article_number": chunk["article_range"],
                    "content": chunk["text"], "content_length": len(chunk["text"]),
                    "parent_key": parent_key, "chunk_index": ci,
                })

            from .ingest import _build_prefix
            prefix = _build_prefix(sec["law_short"], sec["chapter"])
            parent_text = prefix + "\n\n" + "\n\n".join(
                f"{a['article_number']}" +
                (f" {a['article_title']}" if a.get('article_title') else "") +
                f"\n{a['content']}"
                for a in sec["articles"]
            )
            parent_chunks.append({
                "chunk_id": parent_id, "chunk_type": "parent",
                "parent_key": parent_key, "law_name": sec["law_name"],
                "law_short": sec["law_short"], "chapter": sec["chapter"],
                "article_count": len(sec["articles"]),
                "content": parent_text, "content_length": len(parent_text),
                "sub_chunk_count": len(chunks),
            })

        ingest_mgr._save_json(ingest_mgr.all_articles_file, articles)
        ingest_mgr._save_json(ingest_mgr.child_chunks_file, child_chunks)
        ingest_mgr._save_json(ingest_mgr.parent_chunks_file, parent_chunks)

        mapping = {c["chunk_id"]: c["parent_key"] for c in child_chunks}
        ingest_mgr._save_json(ingest_mgr.child_parent_map_file, mapping)

        return {"child_count": len(child_chunks), "parent_count": len(parent_chunks)}

    def step_build_sqlite(data):
        import sqlite3
        childs = ingest_mgr._load_json(ingest_mgr.child_chunks_file)
        parents = ingest_mgr._load_json(ingest_mgr.parent_chunks_file)

        from .config import SQLITE_DB
        conn = sqlite3.connect(SQLITE_DB)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, chunk_type TEXT NOT NULL, text TEXT NOT NULL,
            law_name TEXT, chapter TEXT, article TEXT, parent_key TEXT, content_length INTEGER)""")
        cur.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, law_name, chapter, article, content='chunks', content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 1')""")
        conn.commit()

        for trigger in ["chunks_ai", "chunks_ad", "chunks_au"]:
            cur.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        cur.execute("""CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text, law_name, chapter, article)
            VALUES (new.rowid, new.text, new.law_name, new.chapter, new.article); END""")
        cur.execute("""CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, law_name, chapter, article)
            VALUES ('delete', old.rowid, old.text, old.law_name, old.chapter, old.article); END""")
        cur.execute("""CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, law_name, chapter, article)
            VALUES ('delete', old.rowid, old.text, old.law_name, old.chapter, old.article);
            INSERT INTO chunks_fts(rowid, text, law_name, chapter, article)
            VALUES (new.rowid, new.text, new.law_name, new.chapter, new.article); END""")

        count = 0
        for c in childs:
            cur.execute("""INSERT OR REPLACE INTO chunks
                (id, chunk_type, text, law_name, chapter, article, parent_key, content_length)
                VALUES (?, 'child', ?, ?, ?, ?, ?, ?)""",
                (c["chunk_id"], c["content"], c.get("law_short", c.get("law_name", "")),
                 c.get("chapter", ""), c.get("article_number", ""),
                 c.get("parent_key", ""), c.get("content_length", len(c.get("content", "")))))
            count += 1
        for p in parents:
            cur.execute("""INSERT OR REPLACE INTO chunks
                (id, chunk_type, text, law_name, chapter, article, parent_key, content_length)
                VALUES (?, 'parent', ?, ?, ?, '', ?, ?)""",
                (p["chunk_id"], p["content"], p.get("law_short", p.get("law_name", "")),
                 p.get("chapter", ""), p.get("parent_key", ""),
                 p.get("content_length", len(p.get("content", "")))))
            count += 1

        conn.commit()
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        conn.commit()
        conn.close()
        return {"sqlite_count": count}

    def step_build_milvus(data):
        from pymilvus import MilvusClient, DataType
        from .config import MILVUS_DB, COLLECTION_NAME, MODELS_DIR

        childs = ingest_mgr._load_json(ingest_mgr.child_chunks_file)

        model_path = os.path.join(MODELS_DIR, "models", "BAAI--bge-base-zh-v1.5", "snapshots", "master")
        if not os.path.exists(model_path):
            from modelscope import snapshot_download
            model_path = snapshot_download("BAAI/bge-base-zh-v1.5", cache_dir=MODELS_DIR)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path)

        mc = MilvusClient(uri=MILVUS_DB)
        if mc.has_collection(COLLECTION_NAME):
            mc.drop_collection(COLLECTION_NAME)

        schema = mc.create_schema(auto_id=False, description="劳动法律法规向量库")
        schema.add_field("id", DataType.VARCHAR, max_length=32, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=768)
        schema.add_field("text", DataType.VARCHAR, max_length=32768)
        schema.add_field("law_name", DataType.VARCHAR, max_length=128)
        schema.add_field("chapter", DataType.VARCHAR, max_length=256)
        schema.add_field("article", DataType.VARCHAR, max_length=128)
        schema.add_field("parent_key", DataType.VARCHAR, max_length=128)

        index_params = mc.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="IVF_FLAT",
                               metric_type="IP", params={"nlist": 128})
        mc.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)

        total = len(childs)
        for i in range(0, total, 50):
            batch = childs[i:i + 50]
            batch_texts = [c["content"] for c in batch]
            embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
            rows = []
            for j, c in enumerate(batch):
                rows.append({
                    "id": c["chunk_id"], "vector": embeddings[j].tolist(),
                    "text": c["content"],
                    "law_name": c.get("law_short", c.get("law_name", "")),
                    "chapter": c.get("chapter", ""),
                    "article": c.get("article_number", ""),
                    "parent_key": c.get("parent_key", ""),
                })
            mc.insert(collection_name=COLLECTION_NAME, data=rows)
            logger.info(f"  [Milvus] {min(i+50, total)}/{total}")

        mc.load_collection(COLLECTION_NAME)
        mc.close()
        return {"milvus_count": total}

    runner.add_step("parse", step_parse_all, description="解析所有 DOCX 法条")
    runner.add_step("chunk", step_chunk, description="切分法条为 chunks")
    runner.add_step("sqlite", step_build_sqlite, description="构建 SQLite 文本库")
    runner.add_step("milvus", step_build_milvus, description="构建 Milvus 向量库")

    return runner


def main():
    import argparse

    parser = argparse.ArgumentParser(description="数据管道编排器")
    parser.add_argument("--full", action="store_true", help="全量构建")
    parser.add_argument("--ingest", type=str, metavar="DOCX", help="增量录入法条")
    parser.add_argument("--resume", action="store_true", help="从上次断点继续")
    parser.add_argument("--force", action="store_true", help="强制重新执行所有步骤")
    parser.add_argument("--status", action="store_true", help="查看检查点状态")
    args = parser.parse_args()

    if args.status:
        status = PipelineRunner.get_checkpoint_status()
        if status:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print("无检查点记录")
        return

    if args.ingest:
        from .ingest import ingest_law
        result = ingest_law(args.ingest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.full or args.resume:
        runner = build_full_pipeline()
        result = runner.run(resume=args.resume, force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()