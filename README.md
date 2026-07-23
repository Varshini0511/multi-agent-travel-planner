# Multi-Agent Travel Planner

A LangGraph multi-agent workflow that plans trips end to end: an input
**guardrail**, a **planner** that decomposes the request into research
domains, specialized **subagents** (flights / hotels / activities / visa)
that run in parallel and each choose their own tools, a **synthesizer** that
combines their findings, and a **critic** that reviews against budget and
loops back for revisions. State is persisted per trip in Postgres.

## Concepts implemented

- State / nodes / edges, conditional routing, parallel fan-out and fan-in
- Task decomposition and role-based agents (orchestrator / search / critic)
- Dynamic tool selection (each subagent picks web search vs currency convert)
- Reflection loop (critic sends the plan back to the planner)
- Persistence via a LangGraph Postgres checkpointer + a `trips` history table
- Context engineering (each agent sees only what it needs)
- Input guardrails (prompt-injection defense) and graceful degradation
- Multi-turn refinement against an existing trip
- Human-in-the-loop: the graph pauses (`interrupt`) after the planner so you
  can approve/edit the research plan before searches run
- Long-term semantic memory across trips (pgvector + Gemini embeddings): the
  planner recalls relevant preferences from past trips
- Evaluation harness (`eval.py`): golden cases scored by deterministic checks
  plus an LLM-as-judge rubric
- Provider fallback: Groq primary, Google Gemini fallback

## Stack

LangGraph · LangChain · Groq (Llama 3.3) + Google Gemini (chat + embeddings) ·
Tavily search · Neon Postgres (+ pgvector) · Streamlit

## Run locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
copy .env.example .env       # then fill in your keys
streamlit run streamlit_app.py
```

Required environment variables (see `.env.example`):
`GROQ_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `DATABASE_URL`.

There is also a CLI entry point (`python main.py`) and two standalone teaching
scripts (`01_hello_graph.py`, `02_single_agent.py`).

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (private is fine).
2. Go to https://share.streamlit.io, sign in with GitHub, and grant access to
   this repo.
3. Create a new app pointing at `streamlit_app.py` on the `main` branch.
4. In the app's **Settings -> Secrets**, paste your four secrets in TOML form
   (see `.streamlit/secrets.toml.example`). The app reads them via
   `app/config.py`, which checks `os.environ` (local `.env`) first and falls
   back to `st.secrets` (Streamlit Cloud).
5. Deploy. The Neon database is reached over the internet, so no extra setup is
   needed there.
