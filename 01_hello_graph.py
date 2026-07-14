"""
Concept 1: State, Nodes, Edges

Every LangGraph app is three things:
  1. STATE  - a shared dict-like object that flows through the graph
  2. NODES  - plain functions that read state and return a partial update
  3. EDGES  - wires that say which node runs after which

No LLM, no API key needed here. Run: python 01_hello_graph.py
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1. STATE
# TypedDict describing every field that can flow through the graph.
# Every node receives the *whole* state and returns only the keys it changed.
class TripState(TypedDict):
    destination: str
    budget_usd: int
    notes: list[str]


# 2. NODES
# A node is just: (state) -> dict of updated fields.
def intake(state: TripState) -> dict:
    print(f"[intake] planning a trip to {state['destination']} on ${state['budget_usd']}")
    return {"notes": state["notes"] + [f"Trip requested to {state['destination']}"]}


def budget_check(state: TripState) -> dict:
    verdict = "tight" if state["budget_usd"] < 1500 else "comfortable"
    print(f"[budget_check] budget looks {verdict}")
    return {"notes": state["notes"] + [f"Budget assessed as {verdict}"]}


def summarize(state: TripState) -> dict:
    print("[summarize] final notes:")
    for n in state["notes"]:
        print("  -", n)
    return {}


# 3. EDGES
# START/END are sentinel nodes LangGraph provides. A linear chain here:
# START -> intake -> budget_check -> summarize -> END
builder = StateGraph(TripState)
builder.add_node("intake", intake)
builder.add_node("budget_check", budget_check)
builder.add_node("summarize", summarize)

builder.add_edge(START, "intake")
builder.add_edge("intake", "budget_check")
builder.add_edge("budget_check", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()

if __name__ == "__main__":
    graph.invoke({"destination": "Lisbon", "budget_usd": 1200, "notes": []})
