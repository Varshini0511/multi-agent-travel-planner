"""
Runs the full multi-agent travel planner twice to show two different things:

  1. Lisbon, tight budget -> triggers the critic->planner revision loop
     (no visa needed, so only flights/hotels/activities fan out)
  2. Hanoi, roomy budget -> triggers the dynamic visa branch (destination-
     dependent decomposition) and approves on the first pass

Each run uses its own `thread_id` so the checkpointer keeps their state
separate - then we inspect a saved thread's history to show persistence.
"""
from app.db import save_trip
from app.graph import build_graph
from app.state import default_trip_state, refinement_update

graph = build_graph()


def _print_result(thread_id: str, result: dict):
    if result["rejected"]:
        print(f"--- rejected: {result['rejection_reason']} ---")
        return
    print(f"\n--- itinerary (thread '{thread_id}') ---")
    print(result["itinerary"])
    print(f"total: ${result['total_cost']}  |  approved: {result['approved']}")
    if result["warnings"]:
        print("warnings:", result["warnings"])
    save_trip(thread_id, result)


def run_trip(thread_id: str, destination: str, budget_usd: int, nights: int):
    print(f"\n===== thread '{thread_id}': {destination}, ${budget_usd}, {nights} nights =====")
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(default_trip_state(destination, budget_usd, nights), config=config)
    _print_result(thread_id, result)
    return config


def refine_trip(config: dict, message: str):
    """Multi-turn: continues an EXISTING thread with a follow-up request,
    instead of starting a new one. Note this passes only a partial state
    update (refinement_update) - destination/budget/nights/etc all come
    from the thread's last checkpoint, not from this call."""
    thread_id = config["configurable"]["thread_id"]
    print(f"\n===== refining thread '{thread_id}': '{message}' =====")
    result = graph.invoke(refinement_update(message), config=config)
    _print_result(thread_id, result)
    return config


if __name__ == "__main__":
    lisbon_config = run_trip("trip-lisbon", "Lisbon", budget_usd=900, nights=3)

    # Multi-turn: same thread_id, no new trip started - the graph resumes
    # from Lisbon's last checkpoint and re-plans around this new ask.
    refine_trip(lisbon_config, "Can you add a day trip suggestion and keep it under budget?")

    # Persistence check: pull the checkpointed state back out for a thread
    # without re-running the graph. This is what lets a real app resume a
    # conversation/session days later just by reusing the same thread_id.
    print("\n===== inspecting persisted state for 'trip-lisbon' =====")
    saved = graph.get_state(lisbon_config)
    print("revision_count reached:", saved.values["revision_count"])
    print("domains used on final pass:", saved.values["domains"])
