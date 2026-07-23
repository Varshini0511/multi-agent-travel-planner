"""
All Postgres (Neon) wiring lives here:

  - a connection pool shared by the LangGraph checkpointer AND the trips table
  - get_checkpointer() -> gives app/graph.py a PostgresSaver instead of MemorySaver,
    so trip state survives app restarts, not just the current process
  - a lightweight `trips` table for the UI's history sidebar - separate from
    LangGraph's own checkpoint tables, which store raw graph state and aren't
    meant to be queried directly for "list my past trips"
"""
from datetime import datetime, timezone

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

from app.config import get_secret

DATABASE_URL = get_secret("DATABASE_URL")

# Neon's free tier auto-suspends the database after inactivity (scale-to-zero)
# and closes idle server-side connections. A plain pool then hands out dead
# connections and psycopg raises "SSL connection has been closed unexpectedly".
# These settings make the pool resilient to that:
#   - check: ping each connection on checkout, transparently replacing dead ones
#   - min_size=0: don't keep idle connections open (nothing to go stale)
#   - max_idle / max_lifetime: proactively recycle before Neon kills them
_pool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=0,
    max_size=10,
    max_idle=30,
    max_lifetime=300,
    check=ConnectionPool.check_connection,
    kwargs={"autocommit": True, "prepare_threshold": 0},
)


def get_checkpointer() -> PostgresSaver:
    checkpointer = PostgresSaver(_pool)
    checkpointer.setup()  # creates LangGraph's checkpoint tables if they don't exist yet
    return checkpointer


def init_trips_table() -> None:
    with _pool.connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trips (
                thread_id TEXT PRIMARY KEY,
                destination TEXT NOT NULL,
                budget_usd INTEGER NOT NULL,
                nights INTEGER NOT NULL,
                itinerary TEXT NOT NULL,
                total_cost INTEGER NOT NULL,
                approved BOOLEAN NOT NULL,
                revision_count INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)


# ---------------------------------------------------------------------------
# Long-term semantic memory: preferences learned from past trips, stored as
# embedding vectors so a NEW trip can recall relevant ones by meaning (not
# keyword). Uses the pgvector extension; EMBEDDING_DIM must match the model
# used in app/memory.py (Gemini text-embedding-004 = 768 dims).
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 768


def init_memory_table() -> None:
    with _pool.connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)


def _vec_literal(embedding: list[float]) -> str:
    # pgvector accepts a bracketed string like '[0.1,0.2,...]' cast to ::vector
    return "[" + ",".join(str(x) for x in embedding) + "]"


def save_memory(content: str, embedding: list[float]) -> None:
    with _pool.connection() as conn:
        conn.execute(
            "INSERT INTO memories (content, embedding, created_at) VALUES (%s, %s::vector, %s)",
            (content, _vec_literal(embedding), datetime.now(timezone.utc)),
        )


def search_memories(embedding: list[float], k: int = 3) -> list[str]:
    with _pool.connection() as conn:
        rows = conn.execute(
            # <=> is pgvector's cosine-distance operator (smaller = more similar)
            "SELECT content FROM memories ORDER BY embedding <=> %s::vector LIMIT %s",
            (_vec_literal(embedding), k),
        ).fetchall()
        return [r[0] for r in rows]


def save_trip(thread_id: str, state: dict) -> None:
    with _pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO trips (thread_id, destination, budget_usd, nights, itinerary,
                                total_cost, approved, revision_count, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (thread_id) DO UPDATE SET
                itinerary = EXCLUDED.itinerary,
                total_cost = EXCLUDED.total_cost,
                approved = EXCLUDED.approved,
                revision_count = EXCLUDED.revision_count
            """,
            (
                thread_id,
                state["destination"],
                state["budget_usd"],
                state["nights"],
                state["itinerary"],
                state["total_cost"],
                state["approved"],
                state["revision_count"],
                datetime.now(timezone.utc),
            ),
        )


def get_trip(thread_id: str) -> dict | None:
    with _pool.connection() as conn:
        row = conn.execute(
            """
            SELECT thread_id, destination, budget_usd, nights, itinerary,
                   total_cost, approved, revision_count, created_at
            FROM trips WHERE thread_id = %s
            """,
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        columns = ["thread_id", "destination", "budget_usd", "nights", "itinerary",
                   "total_cost", "approved", "revision_count", "created_at"]
        return dict(zip(columns, row))


def list_trips() -> list[dict]:
    with _pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, destination, budget_usd, nights, itinerary,
                   total_cost, approved, revision_count, created_at
            FROM trips
            ORDER BY created_at DESC
            """
        ).fetchall()
        columns = ["thread_id", "destination", "budget_usd", "nights", "itinerary",
                   "total_cost", "approved", "revision_count", "created_at"]
        return [dict(zip(columns, row)) for row in rows]
