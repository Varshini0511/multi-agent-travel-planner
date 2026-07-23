"""
Long-term semantic memory across trips.

Unlike the checkpointer (which remembers ONE trip's state under a thread_id),
this remembers durable user PREFERENCES across separate trips/threads:
"prefers budget hotels", "dislikes long layovers", "vegetarian", etc.

Two nodes plug into the graph:
  - recall_memory (before planner): embed the current request, semantic-search
    past memories in pgvector, and inject the closest ones into the planner's
    context so old preferences shape the new plan.
  - remember_memory (after the critic approves): ask the LLM to extract any
    durable, reusable preference from this trip and store it (with its
    embedding) for the future.

Both degrade gracefully: if the embedding API is unavailable, memory is simply
skipped with a warning - it never breaks a trip.
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_secret
from app.db import save_memory, search_memories
from app.llm import structured
from app.state import TripState
from pydantic import BaseModel, Field

# gemini-embedding-001 is the embedding model this free-tier key has access to
# (text-embedding-004 returns 404 for it). Pinned to 768 dims via
# output_dimensionality so it matches db.EMBEDDING_DIM / the pgvector column.
_google_key = get_secret("GOOGLE_API_KEY", required=False)
_embeddings = (
    GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=_google_key,
        output_dimensionality=768,
    )
    if _google_key else None
)


class ExtractedPreference(BaseModel):
    has_preference: bool = Field(description="True if the trip reveals a durable, reusable traveler preference")
    preference: str = Field(description="The preference as a short standalone sentence, or empty if none")


_pref_extractor = structured(ExtractedPreference).with_retry(
    stop_after_attempt=3, wait_exponential_jitter=True
)


def recall_memory(state: TripState) -> dict:
    """Before planning: pull preferences relevant to this destination/request."""
    if _embeddings is None:
        return {"memory_context": ""}
    try:
        query = f"Trip to {state['destination']} for {state['nights']} nights, budget ${state['budget_usd']}. {state.get('refinement_request', '')}"
        vector = _embeddings.embed_query(query)
        memories = search_memories(vector, k=3)
        if not memories:
            print("[recall_memory] no relevant memories")
            return {"memory_context": ""}
        context = "\n".join(f"- {m}" for m in memories)
        print(f"[recall_memory] recalled {len(memories)} preference(s)")
        return {"memory_context": context}
    except Exception as exc:
        print(f"[recall_memory] unavailable ({exc!r}); skipping")
        return {"memory_context": "", "warnings": ["Long-term memory recall was unavailable."]}


def remember_memory(state: TripState) -> dict:
    """After approval: extract and store any durable preference from this trip."""
    if _embeddings is None:
        return {}
    try:
        extracted = _pref_extractor.invoke(
            f"A traveler planned a trip to {state['destination']} "
            f"({state['nights']} nights, budget ${state['budget_usd']}). "
            f"Their follow-up requests this session: {state.get('refinement_request', '') or '(none)'}\n"
            f"Final itinerary:\n{state['itinerary']}\n\n"
            f"Extract ONE durable, reusable travel preference this reveals about the "
            f"traveler (e.g. budget sensitivity, trip style), suitable for improving "
            f"FUTURE trips to other destinations. If nothing durable, say so."
        )
        if not extracted.has_preference or not extracted.preference.strip():
            print("[remember_memory] no durable preference to store")
            return {}
        vector = _embeddings.embed_query(extracted.preference)
        save_memory(extracted.preference, vector)
        print(f"[remember_memory] stored: {extracted.preference}")
        return {}
    except Exception as exc:
        print(f"[remember_memory] unavailable ({exc!r}); skipping")
        return {"warnings": ["Storing long-term memory was unavailable."]}
