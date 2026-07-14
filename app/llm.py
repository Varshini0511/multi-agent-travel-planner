"""
LLM access for the whole app - planner, critic, guardrail, and the tool loop
in tools.py all get their models from here.

Provider strategy: Groq (Llama 3.3) is PRIMARY, Google Gemini is a FALLBACK.
Groq's free tier is fast but has a tight daily token budget; when a Groq call
fails for ANY reason (rate limit, outage, a bad tool-call generation), the
same call is automatically retried against Gemini via LangChain's
`.with_fallbacks()`. Gemini is only wired in if GOOGLE_API_KEY is set, so the
app still runs Groq-only if you haven't added a Gemini key yet.

Everything is exposed through `structured(schema)` rather than a bare `llm`,
because the fallback has to be applied AFTER `.with_structured_output()` -
`.with_structured_output()` is a chat-model method and doesn't exist on the
combined fallback runnable, so we build the structured chain per-provider
first and only then fall back between them.
"""
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

_groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

_gemini = None
if os.environ.get("GOOGLE_API_KEY"):
    from langchain_google_genai import ChatGoogleGenerativeAI

    # gemini-flash-lite-latest is what this project's free-tier key actually
    # has access to - gemini-2.0-flash / 2.5-flash report a free-tier limit
    # of 0 for this key, so they can't be used without billing.
    _gemini = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0,
    )


def structured(schema):
    """Return a Runnable that yields an instance of `schema`, trying Groq
    first and falling back to Gemini (if configured) on any Groq failure.
    Callers still chain `.with_retry(...)` on the result as before."""
    primary = _groq.with_structured_output(schema)
    if _gemini is None:
        return primary
    return primary.with_fallbacks([_gemini.with_structured_output(schema)])
