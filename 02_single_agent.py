"""
Concept 2: A single Executor agent, and why it strains on multi-domain tasks

The classic single-agent loop (ReAct pattern):
    agent decides an action -> tool executes -> agent observes result -> repeat
    until the agent decides it's done.

No API key yet, so `agent_brain` below is a hand-written stand-in for an LLM's
reasoning step. Later we'll swap it for a real ChatAnthropic call bound to
tools - the GRAPH SHAPE (agent <-> tool loop) does not change either way.

Run: python 02_single_agent.py
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class TripState(TypedDict):
    destination: str
    topics_needed: list[str]   # e.g. ["flights", "hotels", "activities"]
    findings: dict[str, str]   # topic -> result, filled in as we go
    done: bool
    next_topic: str            # scratch field: what the agent decided to research next


# --- Mocked "tool" -----------------------------------------------------
# Stands in for a real Tavily/SerpAPI call. Same shape you'd use for real:
# def search_web(query: str) -> str: return tavily_client.search(query)
MOCK_RESULTS = {
    "flights": "Round-trip Lisbon flights avg $480, best fares Tue/Wed departures.",
    "hotels": "Mid-range Lisbon hotels avg $110/night in Alfama & Baixa districts.",
    "activities": "Top picks: Belem Tower, Tram 28, day trip to Sintra.",
}


def search_web(topic: str, destination: str) -> str:
    return MOCK_RESULTS.get(topic, f"No data for {topic}")


# --- Agent node ----------------------------------------------------------
# Stand-in for an LLM reasoning step: "what's the single next thing to do?"
# A real LLM agent does this same job via tool-calling, one call at a time.
def agent_brain(state: TripState) -> dict:
    remaining = [t for t in state["topics_needed"] if t not in state["findings"]]
    if not remaining:
        print("[agent] all topics covered, wrapping up")
        return {"done": True}
    next_topic = remaining[0]
    print(f"[agent] deciding to research: '{next_topic}' (still queued: {remaining[1:]})")
    return {"next_topic": next_topic}


def tool_node(state: TripState) -> dict:
    topic = state["next_topic"]
    result = search_web(topic, state["destination"])
    print(f"[tool] searched '{topic}' -> {result}")
    return {"findings": {**state["findings"], topic: result}}


def route(state: TripState) -> str:
    return END if state.get("done") else "tool"


builder = StateGraph(TripState)
builder.add_node("agent", agent_brain)
builder.add_node("tool", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route, {"tool": "tool", END: END})
builder.add_edge("tool", "agent")  # loop back: observe -> decide again

graph = builder.compile()

if __name__ == "__main__":
    final = graph.invoke({
        "destination": "Lisbon",
        "topics_needed": ["flights", "hotels", "activities"],
        "findings": {},
        "done": False,
        "next_topic": "",
    })
    print("\nfinal findings:", final["findings"])

    # The limitation this sets up for Concept 3/4:
    # - every topic funnels through ONE agent, ONE context, in strict sequence
    # - the agent's "memory" (findings so far) keeps growing as more topics
    #   are added, bloating what it has to reason over each turn
    # - topics are independent (flights don't depend on hotels) but this
    #   agent still does them one-at-a-time - there's no parallelism
    # A planner that decomposes work + parallel subagents fixes both.
