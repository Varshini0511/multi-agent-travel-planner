"""
The single shared schema for the whole graph. Every node in every concept
(planner, subagents, synthesizer, critic) reads and writes this same shape -
LangGraph merges each node's returned dict into it after every step.
"""
from typing import Annotated, TypedDict


def merge_warnings(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """Custom reducer for the `warnings` channel. Two jobs:

      1. CONCATENATE - multiple subagents can write warnings in the same
         parallel superstep, and LangGraph needs a reducer to combine
         concurrent writes to one key (plain operator.add did this before).
      2. RESET on None - passing warnings=None replaces the list with [].
         Needed for multi-turn: a refinement reuses the SAME thread, so the
         checkpoint still holds the previous run's warnings; without a way to
         clear them, operator.add would make warnings grow forever across
         turns. default_trip_state and refinement_update both pass None to
         start each turn's warnings fresh.
    """
    if new is None:
        return []
    return (existing or []) + new


class TripState(TypedDict):
    # -- input --
    destination: str
    budget_usd: int
    nights: int

    # -- multi-turn: a follow-up request against an EXISTING thread.
    # Empty on a fresh trip; set when a user asks to adjust an already-run
    # trip without starting a new thread_id.
    refinement_request: str

    # -- human-in-the-loop: when False, the graph pauses (interrupt) after the
    # planner so a human can approve/edit the domain list before subagents run.
    # True (CLI, eval) skips the pause so those runs don't block.
    auto_approve: bool

    # -- long-term memory: preferences recalled from past trips, injected into
    # the planner's context (populated by the recall_memory node).
    memory_context: str

    # -- guardrail output --
    rejected: bool
    rejection_reason: str

    # -- cross-cutting: any node can append a note about degraded behavior.
    # See merge_warnings above for why this needs a custom reducer (concat
    # concurrent writes, but reset on None for multi-turn).
    warnings: Annotated[list[str], merge_warnings]

    # -- planner output (decomposition) --
    domains: list[str]          # e.g. ["flights", "hotels", "activities", "visa"]
    revision_count: int

    # -- subagent outputs (one field per domain, no shared-key conflicts) --
    flights_result: str
    flights_cost: int
    hotels_result: str
    hotels_cost: int
    activities_result: str
    activities_cost: int
    visa_result: str
    visa_cost: int

    # -- synthesizer output --
    itinerary: str
    total_cost: int

    # -- critic output --
    approved: bool
    critic_feedback: str


def default_trip_state(destination: str, budget_usd: int, nights: int,
                       auto_approve: bool = True) -> "TripState":
    """The one place every field TripState declares gets seeded - used by
    every entry point (main.py, streamlit_app.py) so adding a new field to
    TripState only ever means updating it here, not hunting down every caller.
    auto_approve defaults True (CLI/eval never pause); Streamlit passes False."""
    return {
        "destination": destination,
        "budget_usd": budget_usd,
        "nights": nights,
        "refinement_request": "",
        "auto_approve": auto_approve,
        "memory_context": "",
        "rejected": False,
        "rejection_reason": "",
        "warnings": None,  # None => merge_warnings resets to [] (fresh trip)
        "domains": [],
        "revision_count": 0,
        "flights_result": "", "flights_cost": 0,
        "hotels_result": "", "hotels_cost": 0,
        "activities_result": "", "activities_cost": 0,
        "visa_result": "", "visa_cost": 0,
        "itinerary": "", "total_cost": 0,
        "approved": False, "critic_feedback": "",
    }


def refinement_update(message: str) -> dict:
    """A PARTIAL state update for continuing an existing thread, not a fresh
    trip. Unlike default_trip_state, this deliberately omits destination/
    budget_usd/nights/domains/etc - LangGraph's checkpointer already has
    those from the thread's last run, and invoking with a partial dict
    merges onto that persisted state rather than replacing it. Only the
    fields that need to change for a new refinement round are set here."""
    return {
        "refinement_request": message,
        "revision_count": 0,
        "approved": False,
        "warnings": None,  # None => merge_warnings resets, so this turn starts fresh
    }
