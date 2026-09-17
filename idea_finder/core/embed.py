"""Embedding stage helpers: e5-small vectors for pain bodies.

Uses ``intfloat/multilingual-e5-small`` on CPU (project constraint: local
single-user tool, no GPU requirement; canonical HF repo is
intfloat/multilingual-e5-small — the ``sentence-transformers`` org
namespace has no such model). Vectors are
L2-normalized (``normalize_embeddings=True``) so cosine similarity reduces
to a dot product; the clustering stage (D2) applies the ~0.82 cosine
threshold from context/SYSTEM_DESIGN.md on top of these vectors.

e5 convention: input texts must carry a task prefix. Corpus documents
(pain bodies) use ``"passage: "``; short ad-hoc search strings would use
``"query: "``. Pain bodies are documents being indexed, so they get the
passage prefix (see https://arxiv.org/abs/2212.03533, section 2.2).

Storage: vectors land in ``pain.embedding`` (``vector(384)``, migration
001) via :func:`idea_finder.db.repo.update_pain_embedding`; the ivfflat
cosine index ``idx_pain_embedding_ivfflat`` (lists=100, same migration)
serves ANN search at clustering time — nothing to do here, the index ships
with the schema. The model downloads to the local HF cache (~120 MB) on
first use and is reused from cache afterwards; the model object itself is
loaded once per process.

Typing note: sentence-transformers/torch ship no type information usable
under ``ty`` strict, so the model type is imported only under
``TYPE_CHECKING`` (stub-shaped) and the ``encode`` call site is narrowed
with an explicit ``Callable`` cast at the single place that touches it.

Stage contract: :func:`embed_pains` is idempotent — it selects only pains
with a NULL embedding, so a rerun embeds nothing and reports them as
skipped.

Smoke (acceptance)::

    PGDATA_DIR=<tmp> python -m idea_finder.core.embed
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from psycopg import Connection

from idea_finder.db.repo import update_pain_embedding

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "MODEL_ID",
    "EmbedError",
    "Embedder",
    "embed_pains",
    "embed_texts",
    "get_embedder",
]

#: Hugging Face id of the embedding model. Canonical repo (the
#: ``sentence-transformers`` org namespace has no multilingual-e5-small).
MODEL_ID = "intfloat/multilingual-e5-small"

#: e5 task prefixes; body texts are corpus documents (passage), ad-hoc
#: search strings would be queries.
_PASSAGE_PREFIX = "passage: "
_QUERY_PREFIX = "query: "

#: Pain-body batches sent to the encoder at once.
DEFAULT_BATCH_SIZE = 32


class EmbedError(Exception):
    """Raised when the model fails to load or the input is unusable."""


class _LazyModel:
    """Process-wide lazy model holder (singleton by module attribute)."""

    __slots__ = ("_model",)

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def get(self) -> SentenceTransformer:
        """Load the model on first call; reuse the cached instance after."""
        if self._model is None:
            try:
                # Local import: torch/sentence-transformers load in ~2s and
                # only the embed stage needs them.
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                msg = "sentence-transformers is not installed"
                raise EmbedError(msg) from e
            try:
                self._model = SentenceTransformer(MODEL_ID, device="cpu")
            except Exception as e:
                msg = f"failed to load embedding model {MODEL_ID}"
                raise EmbedError(msg) from e
            logger.info("loaded embedding model %s", MODEL_ID)
        return self._model


_model = _LazyModel()


def get_embedder() -> Embedder:
    """Return the process-wide :class:`Embedder` (module-level singleton)."""
    return _EMBEDDER


class Embedder:
    """Thin typed wrapper around the lazily loaded e5-small model."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with the e5 passage prefix; L2-normalized vectors.

        Raises:
            EmbedError: If ``texts`` is empty or the encoder fails.
        """
        if not texts:
            msg = "embed_texts requires a non-empty list of texts"
            raise EmbedError(msg)
        # sentence-transformers ships no type info; encode() is narrowed to
        # its documented signature here (single call site).
        encode = cast(
            "Callable[..., Sequence[Sequence[float]]]", _model.get().encode
        )
        try:
            vectors = encode(
                [_PASSAGE_PREFIX + text for text in texts],
                batch_size=DEFAULT_BATCH_SIZE,
                normalize_embeddings=True,
            )
        except EmbedError:
            raise
        except Exception as e:
            msg = "failed to encode texts"
            raise EmbedError(msg) from e
        return [[float(value) for value in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a short search string with the e5 query prefix."""
        return self.embed_texts([_QUERY_PREFIX + text])[0]


_EMBEDDER = Embedder()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Module-level shortcut: embed texts via the shared :class:`Embedder`."""
    return _EMBEDDER.embed_texts(texts)


def embed_pains(conn: Connection, batch_size: int = DEFAULT_BATCH_SIZE) -> dict[str, int]:
    """Backfill ``pain.embedding`` for pains that do not have one yet.

    Selects every pain whose embedding is NULL, encodes ``pain.body`` in
    batches of ``batch_size``, and writes vectors back through
    :func:`idea_finder.db.repo.update_pain_embedding`. Idempotent by
    selection: a rerun finds no NULL embeddings left.

    Returns:
        ``{"embedded": N, "skipped": M}`` where ``M`` is the number of
        pains that already had an embedding before this call.
    """
    with conn.transaction():
        rows = conn.execute(
            "SELECT id, body FROM pain WHERE embedding IS NULL ORDER BY created_at, id"
        ).fetchall()
        total_row = conn.execute("SELECT count(*) FROM pain").fetchone()
        total = int(total_row[0]) if total_row is not None else 0
    # Snapshot the pending set up front: rows are updated one by one below,
    # so a paginated "WHERE embedding IS NULL" scan would paginate over a
    # shrinking set and silently skip rows.
    skipped = total - len(rows)
    pending_ids = [str(row[0]) for row in rows]
    if not pending_ids:
        logger.info("embed_pains: nothing to embed, %d already embedded", skipped)
        return {"embedded": 0, "skipped": skipped}

    embedded = 0
    for chunk_start in range(0, len(pending_ids), batch_size):
        chunk = rows[chunk_start : chunk_start + batch_size]
        # The id set is fixed, so encode (pure compute, no DB transaction
        # held) and the per-row UPDATEs below cannot skip or double-count.
        ids: list[str] = []
        bodies: list[str] = []
        for pain_id, body in chunk:
            ids.append(str(pain_id))
            bodies.append(str(body))
        vectors = _EMBEDDER.embed_texts(bodies)
        for pain_id, vector in zip(ids, vectors, strict=True):
            update_pain_embedding(conn, pain_id, vector)
            embedded += 1
    logger.info(
        "embed_pains: embedded=%d skipped=%d (batch_size=%d)", embedded, skipped, batch_size
    )
    return {"embedded": embedded, "skipped": skipped}


def _main() -> int:
    """Smoke: tmp cluster, migrations, 10 synthetic pains, embed twice."""
    from datetime import UTC, datetime

    from idea_finder.core.canonical import canonical_url
    from idea_finder.core.models import Pain, RawPost
    from idea_finder.db.bootstrap_pgserver import ensure_pgserver
    from idea_finder.db.migrate import apply_migrations
    from idea_finder.db.repo import (
        ensure_source,
        insert_pain,
        insert_raw_post,
        upsert_prompt_version,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    data_dir = Path(os.environ.get("PGDATA_DIR", "data/pg")).resolve()
    handle = ensure_pgserver(data_dir)
    try:
        with handle.get_conn() as conn:
            applied = apply_migrations(conn)
            print("migrations applied:", applied)
            source_id = ensure_source(conn, "smoke-embed")
            pv_id = upsert_prompt_version(conn, "extract_pains", 1, "PROMPT BODY", "file")
            for i in range(1, 11):
                url = f"https://example.com/smoke/{i}"
                post_id = insert_raw_post(
                    conn,
                    RawPost(
                        source_id=source_id,
                        url_canon=canonical_url(url),
                        url=url,
                        title=f"Smoke {i}",
                        text=f"Smoke body {i}: nobody can find a cheap plumber on weekends",
                        published_at=datetime.now(tz=UTC),
                        kind="demand",
                    ),
                )
                if post_id is None:
                    # Rerun on the same cluster: raw_post already exists
                    # (canonical-URL dedup); reuse its id for the pain.
                    existing = conn.execute(
                        "SELECT id FROM raw_post WHERE url_canon = %s",
                        (canonical_url(url),),
                    ).fetchone()
                    if existing is None:
                        msg = f"smoke: raw_post {url} vanished unexpectedly"
                        raise EmbedError(msg)
                    post_id = str(existing[0])
                insert_pain(
                    conn,
                    Pain(
                        source_post_id=post_id,
                        body=f"Smoke pain {i}: no affordable weekend plumbing exists",
                        audience="flat owners",
                        quote="cheap plumber on weekends",
                    ),
                    None,
                    pv_id,
                )
            stats = embed_pains(conn)
            print(f"first run: embedded: {stats['embedded']}, skipped: {stats['skipped']}")
            row = conn.execute(
                "SELECT count(*) FROM pain WHERE embedding IS NULL"
            ).fetchone()
            left = int(row[0]) if row is not None else -1
            print("pains without embedding after first run:", left)
            stats2 = embed_pains(conn)
            print(
                f"second run: embedded: {stats2['embedded']}, "
                f"skipped: {stats2['skipped']}"
            )
        return 0
    except Exception as e:
        logger.error("embed smoke failed: %s", e)
        raise
    finally:
        handle.stop()


if __name__ == "__main__":
    sys.exit(_main())
