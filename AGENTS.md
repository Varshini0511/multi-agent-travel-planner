# AGENTS.md — context for AI coding agents

Handoff doc so any assistant (Claude Code, T3, Cursor, etc.) can pick up this
project cold. Human-facing setup/usage lives in `README.md`; this file is the
"how it's built and what to watch for" brief.

## What this is

A **LangGraph multi-agent travel planner**, built as a hands-on project to
learn agentic-AI concepts. It plans a trip end to end through a pipeline of
LLM-driven agents. It is a standalone Python project (its own venv), unrelated
to the Next.js shopping app that may share the workspace.

Status: **feature-complete** for its learning goals (all concepts below are
implemented and verified). Pushed to a **private** GitHub repo
`Varshini0511/multi-agent-travel-planner`. Deployed on Streamlit Community Cloud.

## Graph (the whole system)

```
START → guardrail ─reject→ END
          │ ok
       recall_memory → planner → human_approval ─(fan-out)→ {flights,hotels,activities,visa}
                                                                      │ (fan-in)
                                                                synthesizer → critic ─reject→ planner
                                                                                   │ approve
                                                                            remember_memory → END
```

Defined in `app/graph.py`. Nodes are in `app/nodes.py`, `app/guardrail.py`,
`app/memory.py`. The shared state schema is `app/state.py`.

## Layout

| Path | Role |
| --- | --- |
| `streamlit_app.py` | Web UI; deploy target. No agent logic — only renders + calls the graph. |
| `main.py` | CLI: one trip + a refinement. |
| `eval.py` | Eval harness: golden cases + deterministic checks + LLM-as-judge. |
| `01_hello_graph.py`, `02_single_agent.py` | Teaching scripts (state/nodes/edges; single-agent ReAct). |
| `app/state.py` | `TripState` TypedDict, `default_trip_state()`, `refinement_update()`, `merge_warnings` reducer. |
| `app/graph.py` | Wires all nodes + edges; attaches checkpointer; inits tables. |
| `app/nodes.py` | planner, human_approval, subagent factory, synthesizer, critic, routers. |
| `app/tools.py` | Tavily search + per-subagent dynamic tool loop + fallback estimates. |
| `app/guardrail.py` | Input validation + prompt-injection denylist + LLM check. |
| `app/memory.py` | Long-term memory: Gemini embeddings, recall/remember nodes. |
| `app/llm.py` | `structured(schema)` — Groq primary + Gemini fallback, structured output. |
| `app/db.py` | Neon Postgres: pool, PostgresSaver checkpointer, `trips` table, pgvector `memories`. |
| `app/config.py` | `get_secret()` — `.env` (os.environ) then `st.secrets`. |

## Concepts implemented (all done)

state/nodes/edges · conditional routing · parallel fan-out/fan-in · task
decomposition · role-based agents · context engineering · dynamic tool selection
· reflection loop · persistence (checkpointer + thread_id) · input guardrails ·
graceful degradation · multi-turn refinement · human-in-the-loop (interrupt) ·
long-term semantic memory (pgvector) · evaluation · provider fallback.

Not built (possible next steps): observability/tracing, agent-to-agent handoffs,
subgraphs, `Send`-based dynamic map-reduce, per-user memory scoping, token/cost
tracking.

## Conventions (follow these when changing code)

- **State is the single source of truth.** Every field a node reads/writes MUST
  be declared in `TripState` (`app/state.py`) — LangGraph silently drops
  undeclared fields. Seed every field in `default_trip_state()`; adding a field
  means updating that one function, not every caller.
- **Nodes return partial dicts** — only the fields they changed, never the whole
  state.
- **Concurrent writes need a reducer.** Multiple parallel subagents write
  `warnings`, so it uses a custom `Annotated[list, merge_warnings]` reducer
  (concat; reset on `None`). A plain field would raise "can receive only one
  value per step."
- **Every LLM call goes through `structured(schema)`** from `app/llm.py` (Groq→
  Gemini fallback) and is wrapped in `.with_retry(...)`. Don't instantiate chat
  models elsewhere.
- **Secrets via `get_secret(name)`** from `app/config.py` — never read
  `os.environ` directly in app code (breaks on Streamlit Cloud, which only has
  `st.secrets`).
- **Graceful degradation pattern:** LLM/search calls are wrapped in try/except;
  on failure they fall back to a deterministic estimate and append a `warnings`
  note instead of crashing.
- **Human-in-the-loop is gated** by `state["auto_approve"]`: True (CLI, eval)
  skips the `interrupt()`; Streamlit passes False to get the pause.

## Secrets / config

Env vars (local `.env`, git-ignored) or Streamlit Cloud secrets (TOML):
`GROQ_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `DATABASE_URL`.
`app/config.get_secret()` resolves either source. `.env.example` and
`.streamlit/secrets.toml.example` are the committed templates.

## Run / test / debug

```bash
streamlit run streamlit_app.py   # UI
python main.py                   # CLI full run
python eval.py                   # scorecard
```

Debug in VS Code via `.vscode/launch.json` (two configs). Breakpoints only fire
in a process the debugger launched (`F5`), not one started from a terminal. The
Streamlit config runs on port 8502.

## Gotchas (these cost real time if unknown)

- **Groq free tier: ~100k tokens/DAY.** Heavy testing exhausts it; it recovers
  on a rolling basis. A full trip ≈ 15–30 LLM calls. When Groq is out, calls
  fail over to Gemini automatically.
- **Gemini model names are pinned on purpose:** chat = `gemini-flash-lite-latest`,
  embeddings = `gemini-embedding-001` @ 768 dims. The `gemini-2.x-flash` and
  `text-embedding-004` names return `limit: 0` / 404 on the current free-tier key.
- **Groq/Llama-3.3 native tool-calling is unreliable** (emits malformed
  `<function=...>` that its own API rejects, deterministically). That's why the
  subagent tool loop is hand-rolled on `with_structured_output` instead of
  `create_agent`/`bind_tools`. Don't "simplify" it back to native tool-calling.
- **Neon autosuspends** (free tier) and drops idle connections. The pool in
  `app/db.py` uses `check=ConnectionPool.check_connection` + `min_size=0` to
  survive this; don't remove those.
- **Streamlit reruns the whole script every interaction**, and imports run all
  module-level code — so graph build + client setup happen on every page load.
  `@st.cache_resource` on `get_graph()` keeps the graph/pool alive across reruns;
  don't remove it.
- **`MAX_TOOL_STEPS` in `app/tools.py` is intentionally low (1)** to shrink the
  parallel-call burst under free-tier per-minute limits. Raising it increases
  fallback-estimate frequency.

## Git

Private repo `Varshini0511/multi-agent-travel-planner`, branch `main`.
`.env` and `.streamlit/secrets.toml` are git-ignored — never commit secrets.
Streamlit Cloud auto-redeploys on push to `main`.
