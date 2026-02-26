"""Semantic memory: hybrid BM25 + vector search over agent memories.

Stores memories in SQLite with FTS5 (full-text) and sqlite-vec (vector)
tables. Embeddings via fastembed (ONNX, ~50MB RAM, no PyTorch).
Search uses Reciprocal Rank Fusion to combine both retrieval signals.

DB lives at {drive_root}/memory/semantic.db — one file, no external processes.
"""

from __future__ import annotations

import logging
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from prometheus.tools.registry import ToolContext, ToolEntry

log = logging.getLogger(__name__)

# Lazy-loaded embedding model (fastembed, ONNX)
_embed_model = None
_VEC_DIM = 384  # BAAI/bge-small-en-v1.5 dimension

# Module-level DB connection cache
_db_conn: Optional[sqlite3.Connection] = None
_db_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def _get_db(ctx: ToolContext) -> sqlite3.Connection:
    """Get or create SQLite connection with FTS5 + sqlite-vec."""
    global _db_conn, _db_path

    db_dir = ctx.drive_path("memory")
    db_dir.mkdir(parents=True, exist_ok=True)
    path = db_dir / "semantic.db"

    if _db_conn is not None and _db_path == path:
        try:
            _db_conn.execute("SELECT 1")
            return _db_conn
        except Exception:
            _db_conn = None

    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Load sqlite-vec extension
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception as e:
        log.warning("sqlite-vec not available, vector search disabled: %s", e)

    # Create tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT    NOT NULL,
            tags    TEXT    DEFAULT '',
            source  TEXT    DEFAULT 'agent',
            created REAL    NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, tags, content=memories, content_rowid=id);

        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, tags)
            VALUES (new.id, new.content, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags)
            VALUES ('delete', old.id, old.content, old.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, tags)
            VALUES ('delete', old.id, old.content, old.tags);
            INSERT INTO memories_fts(rowid, content, tags)
            VALUES (new.id, new.content, new.tags);
        END;
    """)

    # Create vector table (may fail if sqlite-vec not loaded)
    try:
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
                USING vec0(embedding float[{_VEC_DIM}])
        """)
    except Exception as e:
        log.debug("Could not create vec table: %s", e)

    conn.commit()
    _db_conn = conn
    _db_path = path
    return conn


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _get_embedder():
    """Lazy-load fastembed model (ONNX, ~50MB RAM)."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir="/tmp/fastembed")
        log.info("fastembed model loaded: BAAI/bge-small-en-v1.5 (%d dims)", _VEC_DIM)
        return _embed_model
    except ImportError:
        log.warning("fastembed not installed — vector search unavailable. pip install fastembed")
        return None
    except Exception as e:
        log.warning("fastembed load failed: %s", e)
        return None


def _embed(text: str) -> Optional[bytes]:
    """Embed text, return raw bytes for sqlite-vec. None if unavailable."""
    model = _get_embedder()
    if model is None:
        return None
    try:
        embeddings = list(model.embed([text]))
        vec = embeddings[0]
        return struct.pack(f"{_VEC_DIM}f", *vec)
    except Exception as e:
        log.warning("Embedding failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _memory_store(ctx: ToolContext, content: str, tags: str = "",
                  source: str = "agent") -> str:
    """Store a memory with optional tags and source."""
    if not content or not content.strip():
        return "Error: content is required and must be non-empty."

    content = content.strip()
    tags = tags.strip() if tags else ""
    source = source.strip() if source else "agent"

    db = _get_db(ctx)
    now = time.time()

    cursor = db.execute(
        "INSERT INTO memories (content, tags, source, created) VALUES (?, ?, ?, ?)",
        (content, tags, source, now),
    )
    mem_id = cursor.lastrowid

    # Store embedding vector
    vec_stored = False
    vec_bytes = _embed(content)
    if vec_bytes is not None:
        try:
            db.execute(
                "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                (mem_id, vec_bytes),
            )
            vec_stored = True
        except Exception as e:
            log.debug("Vec insert failed (non-fatal): %s", e)

    db.commit()

    tag_info = f" tags=[{tags}]" if tags else ""
    vec_info = " +vector" if vec_stored else " (text-only)"
    return f"Stored memory #{mem_id}{tag_info}{vec_info}."


def _memory_recall(ctx: ToolContext, query: str, k: int = 5) -> str:
    """Hybrid search: BM25 + vector similarity, fused via RRF."""
    if not query or not query.strip():
        return "Error: query is required."

    query = query.strip()
    k = min(max(int(k), 1), 50)
    db = _get_db(ctx)

    # --- BM25 results (FTS5) ---
    bm25_results: Dict[int, float] = {}
    try:
        rows = db.execute("""
            SELECT rowid, rank
            FROM memories_fts
            WHERE memories_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, k * 3)).fetchall()
        for rank_pos, row in enumerate(rows):
            bm25_results[row["rowid"]] = rank_pos + 1  # 1-indexed rank
    except Exception as e:
        log.debug("FTS5 search error (non-fatal): %s", e)

    # --- Vector results (sqlite-vec) ---
    vec_results: Dict[int, float] = {}
    vec_bytes = _embed(query)
    if vec_bytes is not None:
        try:
            rows = db.execute("""
                SELECT rowid, distance
                FROM memories_vec
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """, (vec_bytes, k * 3)).fetchall()
            for rank_pos, row in enumerate(rows):
                vec_results[row["rowid"]] = rank_pos + 1
        except Exception as e:
            log.debug("Vector search error (non-fatal): %s", e)

    # --- Reciprocal Rank Fusion ---
    RRF_K = 60  # standard RRF constant
    all_ids = set(bm25_results.keys()) | set(vec_results.keys())

    if not all_ids:
        return "No memories found. Store some with memory_store first."

    scored: List[tuple] = []
    for mem_id in all_ids:
        score = 0.0
        if mem_id in bm25_results:
            score += 1.0 / (RRF_K + bm25_results[mem_id])
        if mem_id in vec_results:
            score += 1.0 / (RRF_K + vec_results[mem_id])
        scored.append((mem_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [s[0] for s in scored[:k]]

    # --- Fetch full memories ---
    if not top_ids:
        return "No memories found."

    placeholders = ",".join("?" * len(top_ids))
    rows = db.execute(
        f"SELECT id, content, tags, source, created FROM memories WHERE id IN ({placeholders})",
        top_ids,
    ).fetchall()

    # Reorder by RRF score
    row_map = {r["id"]: r for r in rows}
    results = []
    for i, mem_id in enumerate(top_ids, 1):
        row = row_map.get(mem_id)
        if not row:
            continue
        # Determine search signal
        in_bm25 = mem_id in bm25_results
        in_vec = mem_id in vec_results
        signal = "bm25+vec" if (in_bm25 and in_vec) else ("bm25" if in_bm25 else "vec")

        age_days = (time.time() - row["created"]) / 86400
        age_str = f"{age_days:.0f}d ago" if age_days >= 1 else "today"

        tag_str = f" [{row['tags']}]" if row["tags"] else ""
        results.append(
            f"#{row['id']} ({signal}, {age_str}){tag_str}:\n{row['content']}"
        )

    header = f"Found {len(results)} memories (query: \"{query}\"):\n"
    return header + "\n---\n".join(results)


def _memory_list(ctx: ToolContext, tag: str = "", limit: int = 20) -> str:
    """List recent memories, optionally filtered by tag."""
    limit = min(max(int(limit), 1), 100)
    db = _get_db(ctx)

    if tag and tag.strip():
        # Filter by tag (substring match in comma-separated tags field)
        rows = db.execute(
            "SELECT id, content, tags, source, created FROM memories "
            "WHERE tags LIKE ? ORDER BY created DESC LIMIT ?",
            (f"%{tag.strip()}%", limit),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, content, tags, source, created FROM memories "
            "ORDER BY created DESC LIMIT ?",
            (limit,),
        ).fetchall()

    if not rows:
        return "No memories stored yet."

    total = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    lines = [f"Memories ({len(rows)} of {total} total):\n"]

    for row in rows:
        age_days = (time.time() - row["created"]) / 86400
        age_str = f"{age_days:.0f}d ago" if age_days >= 1 else "today"
        tag_str = f" [{row['tags']}]" if row["tags"] else ""
        preview = row["content"][:200] + ("..." if len(row["content"]) > 200 else "")
        lines.append(f"#{row['id']} ({row['source']}, {age_str}){tag_str}: {preview}")

    return "\n".join(lines)


def _memory_delete(ctx: ToolContext, memory_id: int) -> str:
    """Delete a memory by ID."""
    memory_id = int(memory_id)
    db = _get_db(ctx)

    row = db.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        return f"Memory #{memory_id} not found."

    db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    # Also remove from vec table
    try:
        db.execute("DELETE FROM memories_vec WHERE rowid = ?", (memory_id,))
    except Exception:
        pass  # vec table might not exist

    db.commit()
    return f"Deleted memory #{memory_id}."


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("memory_store", {
            "name": "memory_store",
            "description": (
                "Store a memory in semantic memory. Use for learnings, decisions, "
                "solutions, patterns, and anything worth recalling later. "
                "Indexed for both keyword and semantic (meaning-based) search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store (free text, markdown OK).",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags for categorization. E.g. 'fix,minimax,auth'",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["agent", "knowledge", "chat", "review"],
                        "description": "Source category (default: agent).",
                    },
                },
                "required": ["content"],
            },
        }, _memory_store),
        ToolEntry("memory_recall", {
            "name": "memory_recall",
            "description": (
                "Search semantic memory by meaning and keywords. Uses hybrid "
                "BM25 + vector search with rank fusion. Ask questions like "
                "'how did I fix auth errors' or 'MiniMax API gotchas'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 50).",
                    },
                },
                "required": ["query"],
            },
        }, _memory_recall),
        ToolEntry("memory_list", {
            "name": "memory_list",
            "description": (
                "List recent memories, optionally filtered by tag. "
                "Shows preview of each memory with ID, age, and tags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag (optional).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 20, max: 100).",
                    },
                },
                "required": [],
            },
        }, _memory_list),
        ToolEntry("memory_delete", {
            "name": "memory_delete",
            "description": "Delete a memory by its ID number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "integer",
                        "description": "ID of the memory to delete.",
                    },
                },
                "required": ["memory_id"],
            },
        }, _memory_delete),
    ]
