"""
Wires all the nodes from nodes.py into one graph:

    START -> guardrail --> planner --fan-out--> {flights, hotels, activities, visa} (parallel)
                     \--> END (rejected input)
                                        \\___ only nodes planner listed run ___/
                                              |
                                        synthesizer (fan-in)
                                              |
                                            critic --loop back to planner-->
                                              |
                                             END

A checkpointer is attached so state persists per `thread_id` (Concept:
persistence) - you can invoke the same thread again later and it resumes
from its last checkpoint instead of starting cold.
"""
from langgraph.graph import StateGraph, START, END

from app.state import TripState
from app.db import get_checkpointer, init_trips_table
from app.guardrail import input_guardrail, route_after_guardrail
from app.nodes import (
    planner, route_to_domains, make_subagent,
    synthesizer, critic, route_after_critic,
)

ALL_DOMAINS = ["flights", "hotels", "activities", "visa"]


def build_graph():
    builder = StateGraph(TripState)

    builder.add_node("guardrail", input_guardrail)
    builder.add_node("planner", planner)
    for domain in ALL_DOMAINS:
        builder.add_node(domain, make_subagent(domain))
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("critic", critic)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail", route_after_guardrail,
        {"planner": "planner", "END": END},
    )

    # Dynamic parallel fan-out: route_to_domains returns a list of node
    # names based on state, and LangGraph runs all of them concurrently.
    builder.add_conditional_edges(
        "planner", route_to_domains,
        {d: d for d in ALL_DOMAINS},
    )

    # Fan-in: synthesizer only fires once, after every triggered domain
    # node for this step has finished.
    for domain in ALL_DOMAINS:
        builder.add_edge(domain, "synthesizer")

    builder.add_edge("synthesizer", "critic")

    builder.add_conditional_edges(
        "critic", route_after_critic,
        {"planner": "planner", "END": END},
    )

    init_trips_table()
    checkpointer = get_checkpointer()
    return builder.compile(checkpointer=checkpointer)
