"""
增量法条录入模块
- 解析 DOCX → 切分 → 增量写入 SQLite + Milvus
- 不重建全库，只更新受影响的章节
- 入库后自动清除检索缓存
"""
import json
import os
import re
import hashlib
import sqlite3
import time
from collections import OrderedDict
from typing import List, Dict, Set, Tuple, Optional

from ..core.config import (
    PROJECT_ROOT, DATA_DIR, PROCESSED_DIR, LAWS_DIR, VECTORS_DIR,
    MILVUS_DB, SQLITE_DB, MODELS_DIR, COLLECTION_NAME,
)
from ..core.logger import logger

CHUNK_TARGET = 600
MIN_LAST_CHUNK = 200
OVERLAP_ARTICLES = 1

LAW_NAME_MAP = {
    "中华人民共和国劳动争议调解仲裁法_20071229": "劳动争议调解仲裁法",
    "中华人民共和国劳动合同法_20121228": "劳动合同法",
    "中华人民共和国劳动法_20181229": "劳动法",
    "中华人民共和国社会保险法_20181229": "社会保险法",
    "工伤保险条例_20101220": "工伤保险条例",
}


def _parse_docx(filepath: str) -> List[Dict]:
    """解析单个 DOCX 文件，返回法条列表"""
    import docx as python_docx

    doc = python_docx.Document(filepath)
    law_name = os.path.splitext(os.path.basename(filepath))[0]

    articles = []
    current_chapter = ""
    current_article = None
    current_content_lines = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        chapter_match = re.match(r"^第[一二三四五六七八九十百]+章\s*(.*)", text)
        if chapter_match:
            current_chapter = text
            continue

        article_match = re.match(r"^第[一二三四五六七八九十百]+条\s*(.*)", text)
        if article_match:
            if current_article:
                current_article["content"] = "".join(current_content_lines)
                articles.append(current_article)

            article_number = re.match(r"^第[一二三四五六七八九十百]+条", text).group()
            current_article = {
                "law_name": law_name,
                "chapter": current_chapter,
                "article_number": article_number,
                "article_title": "",
            }
            current_content_lines = []

            rest = article_match.group(1).strip()
            if rest:
                colon_pos = rest.find("：")
                if colon_pos == -1:
                    colon_pos = rest.find(":")
                if colon_pos != -1:
                    current_article["article_title"] = rest[:colon_pos].strip()
                    current_content_lines.append(rest[colon_pos + 1:].strip())
                else:
                    if len(rest) <= 10:
                        current_article["article_title"] = rest
                    else:
                        current_content_lines.append(rest)
        else:
            if current_article:
                current_content_lines.append(text)

    if current_article:
        current_article["content"] = "".join(current_content_lines)
        articles.append(current_article)

    return articles


def _build_prefix(law_short: str, chapter: str) -> str:
    parts = [law_short]
    chapter_clean = chapter.replace("\u3000", " ").replace("\u00a0", " ").strip()
    chapter_clean = re.sub(r'\s{2,}', '', chapter_clean)
    parts.append(chapter_clean)
    return "【" + "·".join(parts) + "】"


def _find_sentence_boundary(text: str, target_pos: int) -> int:
    candidates = []
    for m in re.finditer(r'(?:^|\n)\s*(?:第[一二三四五六七八九十百千\d]+条|第[一二三四五六七八九十百千\d]+款)', text):
        candidates.append((m.start(), 1))
    for m in re.finditer(r'[。．]', text):
        candidates.append((m.end(), 2))
    for m in re.finditer(r'[；;]\s*\n', text):
        candidates.append((m.end(), 3))
    for m in re.finditer(r'[；;]', text):
        candidates.append((m.end(), 4))

    best, best_dist = None, float('inf')
    for pos, pri in candidates:
        dist = abs(pos - target_pos)
        if dist > 200:
            continue
        if best is None or pri < best[1] or (pri == best[1] and dist < best_dist):
            best, best_dist = (pos, pri), dist

    return best[0] if best else target_pos


def _split_section_into_chunks(articles: List[Dict], law_name: str,
                                law_short: str, chapter: str) -> List[Dict]:
    if not articles:
        return []

    article_texts = []
    for a in articles:
        line = a['article_number']
        if a.get('article_title'):
            line += f" {a['article_title']}"
        line += f"\n{a['content']}"
        article_texts.append(line)

    total_chars = sum(len(t) for t in article_texts)
    prefix = _build_prefix(law_short, chapter)

    if total_chars <= CHUNK_TARGET:
        full_text = prefix + "\n\n" + "\n\n".join(article_texts)
        return [{"text": full_text, "article_count": len(articles),
                 "article_range": f"{articles[0]['article_number']}~{articles[-1]['article_number']}"}]

    full_plain = "\n\n".join(article_texts)
    article_positions = []
    pos = 0
    for i, at in enumerate(article_texts):
        start = pos
        end = pos + len(at)
        article_positions.append((start, end, at, articles[i]))
        pos = end + 2

    chunks = []
    chunk_start = 0

    while chunk_start < len(full_plain):
        target_end = min(chunk_start + CHUNK_TARGET, len(full_plain))
        if target_end >= len(full_plain):
            chunk_text = full_plain[chunk_start:].strip()
            if chunk_text:
                included = [a for s, e, at, a in article_positions
                            if (s >= chunk_start and e <= len(full_plain)) or
                               (s < chunk_start and e > chunk_start)]
                article_range = f"{included[0]['article_number']}~{included[-1]['article_number']}" if included else ""
                chunks.append({"text": prefix + "\n\n" + chunk_text,
                               "article_count": len(included), "article_range": article_range})
            break

        boundary = _find_sentence_boundary(full_plain, target_end)
        if boundary <= chunk_start:
            boundary = target_end

        chunk_text = full_plain[chunk_start:boundary].strip()
        if not chunk_text:
            break

        included = [a for s, e, at, a in article_positions
                    if s >= chunk_start and e <= boundary]
        article_range = f"{included[0]['article_number']}~{included[-1]['article_number']}" if included else ""
        chunks.append({"text": prefix + "\n\n" + chunk_text,
                       "article_count": len(included), "article_range": article_range})

        if included and OVERLAP_ARTICLES > 0:
            last_article_start = next((s for s, e, at, a in article_positions
                                       if a['article_number'] == included[-1]['article_number']), None)
            chunk_start = last_article_start if (last_article_start and last_article_start > chunk_start) else boundary
        else:
            chunk_start = boundary

        while chunk_start < len(full_plain) and full_plain[chunk_start] in '\n\r ':
            chunk_start += 1

    if len(chunks) >= 2:
        last_text = chunks[-1]["text"]
        last_body = last_text.replace(prefix + "\n\n", "", 1) if prefix + "\n\n" in last_text else last_text
        if len(last_body) < MIN_LAST_CHUNK:
            prev = chunks[-2]
            prev_body = prev["text"].replace(prefix + "\n\n", "", 1) if prefix + "\n\n" in prev["text"] else prev["text"]
            merged_body = prev_body + "\n\n" + last_body
            prev["text"] = prefix + "\n\n" + merged_body
            prev["article_count"] += chunks[-1]["article_count"]
            if prev["article_range"] and chunks[-1]["article_range"]:
                pa = prev["article_range"].split("~")
                la = chunks[-1]["article_range"].split("~")
                if len(pa) == 2 and len(la) == 2:
                    prev["article_range"] = f"{pa[0]}~{la[1]}"
            chunks.pop()

    return chunks


def _group_articles_by_section(articles: List[Dict]) -> OrderedDict:
    sections = OrderedDict()
    for art in articles:
        key = f"{art['law_name']}||{art['chapter']}"
        if key not in sections:
            law_short = LAW_NAME_MAP.get(art['law_name'], art['law_name'])
            sections[key] = {"law_name": art['law_name'], "law_short": law_short,
                             "chapter": art['chapter'], "articles": []}
        sections[key]["articles"].append({
            "article_number": art["article_number"],
            "article_title": art.get("article_title", ""),
            "content": art["content"],
        })
    return sections


class IngestManager:
    """增量录入管理器"""

    def __init__(self):
        LAWS_DIR.mkdir(parents=True, exist_ok=True)
        VECTORS_DIR.mkdir(parents=True, exist_ok=True)

        self.all_articles_file = LAWS_DIR / "all_articles.json"
        self.child_chunks_file = LAWS_DIR / "child_chunks.json"
        self.parent_chunks_file = LAWS_DIR / "parent_chunks.json"
        self.child_parent_map_file = LAWS_DIR / "child_parent_map.json"

    def _load_json(self, path, default=None):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default if default is not None else []

    def _save_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def ingest_law(self, docx_path: str) -> Dict:
        """
        增量录入一部新法条

        Args:
            docx_path: DOCX 文件路径

        Returns:
            {
                "status": "ok" | "error",
                "added_articles": int,
                "affected_sections": int,
                "new_child_chunks": int,
                "updated_parent_chunks": int,
                "total_child_chunks": int,
                "error": str | None,
            }
        """
        start_time = time.time()

        if not os.path.exists(docx_path):
            return {"status": "error", "error": f"文件不存在: {docx_path}"}

        try:
            logger.info("=" * 60)
            logger.info(f"增量法条录入: {os.path.basename(docx_path)}")
            logger.info("=" * 60)

            # 1. 解析新 DOCX
            logger.info("[1/6] 解析法条...")
            new_articles = _parse_docx(docx_path)
            logger.info(f"  提取 {len(new_articles)} 条新法条")

            # 2. 合并到 all_articles.json
            logger.info(f"[2/6] 合并到 {self.all_articles_file}...")
            all_articles = self._load_json(self.all_articles_file)
            existing_keys = set()
            for a in all_articles:
                existing_keys.add((a["law_name"], a["article_number"]))

            added = 0
            for a in new_articles:
                key = (a["law_name"], a["article_number"])
                if key not in existing_keys:
                    all_articles.append(a)
                    existing_keys.add(key)
                    added += 1
                else:
                    for i, old in enumerate(all_articles):
                        if old["law_name"] == a["law_name"] and old["article_number"] == a["article_number"]:
                            all_articles[i] = a
                            break
                    logger.info(f"  [更新] {a['article_number']} 已存在，内容已覆盖")

            self._save_json(self.all_articles_file, all_articles)
            logger.info(f"  新增 {added} 条，总共 {len(all_articles)} 条")

            # 3. 确定受影响的章节
            logger.info("[3/6] 确定受影响的章节...")
            affected_keys = set()
            for a in new_articles:
                affected_keys.add(f"{a['law_name']}||{a['chapter']}")
            logger.info(f"  受影响的章节: {len(affected_keys)} 个")

            # 4. 加载旧数据，获取旧 child_id
            old_child_ids = {}
            old_child_chunks = self._load_json(self.child_chunks_file)
            for c in old_child_chunks:
                pk = c.get("parent_key", "")
                if pk in affected_keys:
                    if pk not in old_child_ids:
                        old_child_ids[pk] = []
                    old_child_ids[pk].append(c["chunk_id"])

            # 5. 重新切分受影响的章节
            logger.info("[4/6] 重新切分受影响的章节...")
            sections = _group_articles_by_section(all_articles)
            affected_sections = {k: v for k, v in sections.items() if k in affected_keys}

            child_chunks_delta = []
            parent_chunks_delta = []
            old_parent_chunks = self._load_json(self.parent_chunks_file)

            for key, sec in affected_sections.items():
                law_name = sec["law_name"]
                law_short = sec["law_short"]
                chapter = sec["chapter"]
                arts = sec["articles"]

                chunks = _split_section_into_chunks(arts, law_name, law_short, chapter)
                parent_key = f"{law_name}||{chapter}"
                parent_id = hashlib.md5(parent_key.encode()).hexdigest()[:12]

                for ci, chunk in enumerate(chunks):
                    chunk_id = hashlib.md5(f"{parent_key}|chunk{ci}".encode()).hexdigest()[:12]
                    child_chunks_delta.append({
                        "chunk_id": chunk_id, "chunk_type": "child",
                        "law_name": law_name, "law_short": law_short,
                        "chapter": chapter, "article_number": chunk["article_range"],
                        "content": chunk["text"], "content_length": len(chunk["text"]),
                        "parent_key": parent_key, "chunk_index": ci,
                    })

                prefix = _build_prefix(law_short, chapter)
                parent_text = prefix + "\n\n" + "\n\n".join(
                    f"{a['article_number']}" +
                    (f" {a['article_title']}" if a.get('article_title') else "") +
                    f"\n{a['content']}"
                    for a in arts
                )
                parent_chunks_delta.append({
                    "chunk_id": parent_id, "chunk_type": "parent",
                    "parent_key": parent_key, "law_name": law_name, "law_short": law_short,
                    "chapter": chapter, "article_count": len(arts),
                    "content": parent_text, "content_length": len(parent_text),
                    "sub_chunk_count": len(chunks),
                })

            logger.info(f"  生成新子块: {len(child_chunks_delta)} 个")
            logger.info(f"  更新父块: {len(parent_chunks_delta)} 个")

            # 6. 合并 JSON 文件
            logger.info("[5/6] 更新 JSON 文件 & 写入数据库...")
            unaffected_child = [c for c in old_child_chunks if c.get("parent_key") not in affected_keys]
            unaffected_parent = [p for p in old_parent_chunks if p.get("parent_key") not in affected_keys]

            new_child = unaffected_child + child_chunks_delta
            new_parent = unaffected_parent + parent_chunks_delta

            self._save_json(self.child_chunks_file, new_child)
            self._save_json(self.parent_chunks_file, new_parent)

            mapping = {c["chunk_id"]: c["parent_key"] for c in new_child}
            self._save_json(self.child_parent_map_file, mapping)

            # 7. 增量写入 SQLite
            sqlite_count = self._upsert_sqlite(child_chunks_delta, parent_chunks_delta)
            logger.info(f"  [SQLite] 写入 {sqlite_count} 条")

            # 8. 增量写入 Milvus
            milvus_count = self._upsert_milvus(child_chunks_delta, old_child_ids)
            logger.info(f"  [Milvus] 写入 {milvus_count} 条向量")

            total_time = time.time() - start_time
            result = {
                "status": "ok",
                "added_articles": added,
                "affected_sections": len(affected_keys),
                "new_child_chunks": len(child_chunks_delta),
                "updated_parent_chunks": len(parent_chunks_delta),
                "total_child_chunks": len(new_child),
                "elapsed_seconds": round(total_time, 2),
            }

            # 9. 清除检索缓存（新数据入库，旧缓存失效）
            try:
                from .ranker import _cache
                _cache.invalidate_all()
            except ImportError:
                pass

            logger.info(f"[OK] 增量录入完成! 耗时 {total_time:.1f}s")
            logger.info(f"  新增法条: {added} | 受影响章节: {len(affected_keys)}")
            logger.info(f"  新子块: {len(child_chunks_delta)} | 总子块: {len(new_child)}")

            return result

        except Exception as e:
            logger.error(f"[增量录入] 失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def _upsert_sqlite(self, child_chunks_delta: List[Dict],
                       parent_chunks_delta: List[Dict]) -> int:
        """增量写入 SQLite"""
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
        for c in child_chunks_delta:
            cur.execute("""INSERT OR REPLACE INTO chunks
                (id, chunk_type, text, law_name, chapter, article, parent_key, content_length)
                VALUES (?, 'child', ?, ?, ?, ?, ?, ?)""",
                (c["chunk_id"], c["content"], c.get("law_short", c.get("law_name", "")),
                 c.get("chapter", ""), c.get("article_number", ""),
                 c.get("parent_key", ""), c.get("content_length", len(c.get("content", "")))))
            count += 1

        for p in parent_chunks_delta:
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
        return count

    def _upsert_milvus(self, child_chunks_delta: List[Dict],
                       old_child_ids_for_parents: Dict[str, List[str]]) -> int:
        """增量写入 Milvus"""
        from pymilvus import MilvusClient

        mc = MilvusClient(uri=MILVUS_DB)

        model_path = os.path.join(MODELS_DIR, "models", "BAAI--bge-base-zh-v1.5", "snapshots", "master")
        if not os.path.exists(model_path):
            from modelscope import snapshot_download
            model_path = snapshot_download("BAAI/bge-base-zh-v1.5", cache_dir=MODELS_DIR)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path)

        if old_child_ids_for_parents:
            ids_to_delete = set()
            for parent_key in old_child_ids_for_parents:
                for cid in old_child_ids_for_parents[parent_key]:
                    ids_to_delete.add(cid)
            if ids_to_delete:
                try:
                    mc.delete(collection_name=COLLECTION_NAME, ids=list(ids_to_delete))
                    logger.info(f"  [Milvus] 删除旧子块: {len(ids_to_delete)} 个")
                except Exception as e:
                    logger.warning(f"  [Milvus] 删除旧子块失败（可能不存在）: {e}")

        if not child_chunks_delta:
            logger.info("  [Milvus] 无新子块需要写入")
            mc.close()
            return 0

        total = len(child_chunks_delta)
        for i in range(0, total, 50):
            batch = child_chunks_delta[i:i + 50]
            batch_texts = [c["content"] for c in batch]
            embeddings = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)

            rows = []
            for j, c in enumerate(batch):
                rows.append({
                    "id": c["chunk_id"],
                    "vector": embeddings[j].tolist(),
                    "text": c["content"],
                    "law_name": c.get("law_short", c.get("law_name", "")),
                    "chapter": c.get("chapter", ""),
                    "article": c.get("article_number", ""),
                    "parent_key": c.get("parent_key", ""),
                })

            mc.insert(collection_name=COLLECTION_NAME, data=rows)

        mc.load_collection(COLLECTION_NAME)
        mc.close()
        return total


_ingest_manager: Optional[IngestManager] = None


def get_ingest_manager() -> IngestManager:
    global _ingest_manager
    if _ingest_manager is None:
        _ingest_manager = IngestManager()
    return _ingest_manager


def ingest_law(docx_path: str) -> Dict:
    """便捷函数：增量录入一部法条"""
    return get_ingest_manager().ingest_law(docx_path)