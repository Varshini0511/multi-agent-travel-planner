"""
Wires all the nodes into one graph:

    START -> guardrail --> recall_memory --> planner --> human_approval
                     \--> END (rejected)                       |
                                                        (fan-out, parallel)
                                          {flights, hotels, activities, visa}
                                                        |
                                                  synthesizer (fan-in)
                                                        |
                                                     critic --reject--> planner
                                                        | approve
                                                  remember_memory --> END

A checkpointer is attached so state persists per `thread_id` (persistence +
human-in-the-loop pause/resume). recall_memory / remember_memory add long-term
semantic memory across separate trips.
"""
from langgraph.graph import StateGraph, START, END

from app.state import TripState
from app.db import get_checkpointer, init_trips_table, init_memory_table
from app.guardrail import input_guardrail, route_after_guardrail
from app.memory import recall_memory, remember_memory
from app.nodes import (
    planner, human_approval, route_to_domains, make_subagent,
    synthesizer, critic, route_after_critic,
)

ALL_DOMAINS = ["flights", "hotels", "activities", "visa"]


def build_graph():
    builder = StateGraph(TripState)

    builder.add_node("guardrail", input_guardrail)
    builder.add_node("recall_memory", recall_memory)
    builder.add_node("planner", planner)
    builder.add_node("human_approval", human_approval)
    for domain in ALL_DOMAINS:
        builder.add_node(domain, make_subagent(domain))
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("critic", critic)
    builder.add_node("remember_memory", remember_memory)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail", route_after_guardrail,
        {"recall_memory": "recall_memory", "END": END},
    )

    # recall preferences from past trips, then plan.
    builder.add_edge("recall_memory", "planner")

    # planner proposes domains, then human_approval optionally pauses for sign-off.
    builder.add_edge("planner", "human_approval")

    # Dynamic parallel fan-out: route_to_domains returns a list of node
    # names based on state, and LangGraph runs all of them concurrently.
    builder.add_conditional_edges(
        "human_approval", route_to_domains,
        {d: d for d in ALL_DOMAINS},
    )

    # Fan-in: synthesizer only fires once, after every triggered domain
    # node for this step has finished.
    for domain in ALL_DOMAINS:
        builder.add_edge(domain, "synthesizer")

    builder.add_edge("synthesizer", "critic")

    # reject -> back to planner to revise; approve -> store memory, then end.
    builder.add_conditional_edges(
        "critic", route_after_critic,
        {"planner": "planner", "remember": "remember_memory"},
    )
    builder.add_edge("remember_memory", END)

    init_trips_table()
    init_memory_table()
    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)
