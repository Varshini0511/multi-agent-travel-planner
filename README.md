# 🧭 Multi-Agent Travel Planner

A LangGraph multi-agent system that plans a trip end to end. A request flows
through a **guardrail**, a **planner** that breaks the job into research
domains, specialized **subagents** (flights / hotels / activities / visa) that
run **in parallel** and each choose their own tools, a **synthesizer** that
combines their findings, and a **critic** that reviews against budget and loops
back for revisions. Trip state is persisted per-thread in Postgres, and the
planner remembers the traveler's preferences across separate trips.

Built as a hands-on way to learn agentic-AI concepts from the ground up — see
[Concepts](#concepts-implemented) for the full list.

## How it flows

```
START → guardrail ─(reject)─→ END
           │ ok
        recall_memory        recall past preferences (semantic search)
           │
        planner              decompose into research domains
           │
        human_approval       PAUSE for sign-off (UI only; interrupt/resume)
           │  (fan-out, parallel)
   ┌───────┼────────┬─────────┐
 flights  hotels  activities  visa     each: its own tool loop
   └───────┼────────┴─────────┘
        synthesizer          combine results + total cost (fan-in)
           │
        critic ─(over budget)─→ back to planner   (reflection loop)
           │ approved
        remember_memory      store durable preferences
           │
          END
```

## Project structure

```
multi-agent-travel-planner/
├── streamlit_app.py        # Web UI (the app you deploy)
├── main.py                 # CLI entry point — runs a trip + a refinement
├── eval.py                 # Evaluation harness (golden cases + LLM judge)
├── 01_hello_graph.py       # Teaching script: state / nodes / edges only
├── 02_single_agent.py      # Teaching script: the single-agent ReAct loop
├── app/
│   ├── state.py            # TripState schema + default_trip_state / reducers
│   ├── graph.py            # wires every node into the graph
│   ├── nodes.py            # planner, subagents, synthesizer, critic, approval
│   ├── tools.py            # Tavily search + the per-subagent tool loop
│   ├── guardrail.py        # input validation + prompt-injection defense
│   ├── memory.py           # long-term semantic memory (recall / remember)
│   ├── llm.py              # Groq primary + Gemini fallback, structured output
│   ├── db.py               # Neon Postgres: checkpointer, trips, pgvector memory
│   └── config.py           # secret resolution (.env locally, st.secrets on Cloud)
└── .vscode/launch.json     # debug configs for main.py and streamlit_app.py
```

## Concepts implemented

- State / nodes / edges, conditional routing, parallel fan-out and fan-in
- Task decomposition and role-based agents (orchestrator / search / critic)
- Context engineering (each agent sees only what it needs)
- Dynamic tool selection (each subagent picks web search vs currency convert)
- Reflection loop (critic sends the plan back to the planner)
- Persistence via a LangGraph Postgres checkpointer + a `trips` history table
- Input guardrails (prompt-injection defense) and graceful degradation
- Multi-turn refinement against an existing trip
- Human-in-the-loop: the graph pauses (`interrupt`) after the planner so you
  can approve/edit the research plan before searches run
- Long-term semantic memory across trips (pgvector + Gemini embeddings)
- Evaluation harness: golden cases scored by deterministic checks + an
  LLM-as-judge rubric
- Provider fallback: Groq primary, Google Gemini fallback

## Stack

LangGraph · LangChain · Groq (Llama 3.3) + Google Gemini (chat + embeddings) ·
Tavily search · Neon Postgres (+ pgvector) · Streamlit

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
copy .env.example .env         # cp on macOS/Linux — then fill in your keys
```

Required secrets (see `.env.example`) — all have free tiers:

| Variable          | Where to get it                          |
| ----------------- | ---------------------------------------- |
| `GROQ_API_KEY`    | console.groq.com                         |
| `GOOGLE_API_KEY`  | aistudio.google.com (chat + embeddings)  |
| `TAVILY_API_KEY`  | tavily.com                               |
| `DATABASE_URL`    | neon.tech (a Postgres connection string) |

## Running it — three ways

```bash
# 1. Web UI — plan trips, approve the plan, refine, browse history
streamlit run streamlit_app.py

# 2. CLI — one full run + a refinement, printed node by node
python main.py

# 3. Evaluation — score golden cases (deterministic checks + LLM judge)
python eval.py
```

## Debugging in VS Code (recommended for learning)

The repo ships two debug configs in `.vscode/launch.json`.

1. Open the **folder** `multi-agent-travel-planner` in VS Code (not a parent, not
   a single file — the dropdown only appears when this folder is the root).
2. Select the interpreter: `Ctrl+Shift+P → Python: Select Interpreter →` the one
   under `venv/`.
3. `Ctrl+Shift+D` → pick a config → `F5`:
   - **Python: main.py** — steps through a full run with no UI. Best for most
     concepts.
   - **Streamlit: streamlit_app.py** — runs the UI *under the debugger* on
     `http://localhost:8502`; use this to hit breakpoints on UI actions.

Tips that save pain:
- Put breakpoints on a **statement**, never a `def`/`class` line (those run once
  at import, not per call).
- Breakpoints only fire in a process the **debugger** launched. An app started
  from a plain terminal is not debuggable.
- Streamlit **reruns the whole script top-to-bottom on every interaction**, and
  imports run all module-level code — so setup/`build_graph` breakpoints hit at
  load, while node bodies (`planner`, `critic`, subagents) hit only when you
  click "Plan my trip".

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (private is fine — `.env` is git-ignored).
2. At https://share.streamlit.io, sign in with GitHub and grant access to the
   repo.
3. Create an app pointing at `streamlit_app.py` on `main`.
4. In **Settings → Secrets**, paste the four secrets in TOML form (see
   `.streamlit/secrets.toml.example`). `app/config.py` reads `os.environ` first
   (local `.env`) and falls back to `st.secrets` (Cloud).
5. Deploy. Neon is reached over the internet, so no extra DB setup is needed.

## Notes & gotchas

- **Free-tier rate limits.** A full trip fires many LLM calls; on free tiers a
  parallel burst can trip the per-minute limit. The app degrades gracefully
  (fallback estimates + a "Note:" banner) rather than crashing. `MAX_TOOL_STEPS`
  in `app/tools.py` caps how many calls each subagent makes.
- **Model names.** This project pins `gemini-flash-lite-latest` (chat) and
  `gemini-embedding-001` @ 768 dims — the `gemini-2.x-flash` / `text-embedding-004`
  names return errors on the current free-tier keys.
- **Never commit secrets.** `.env` and `.streamlit/secrets.toml` are git-ignored;
  only the `.example` templates are tracked.
