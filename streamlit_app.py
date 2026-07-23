"""
Streamlit UI around the LangGraph app from app/graph.py - no agent logic
lives here. This file: takes trip requests, streams the graph's execution
as human-readable progress, saves the result to Postgres, and lists past
trips from the `trips` table so they can be revisited without rerunning
anything.

Run: streamlit run streamlit_app.py
"""
import uuid

import streamlit as st
from langgraph.types import Command

from app.db import get_trip, list_trips, save_trip
from app.graph import build_graph
from app.state import default_trip_state, refinement_update

st.set_page_config(page_title="Multi-Agent Travel Planner", page_icon="🧭", layout="wide")


@st.cache_resource
def get_graph():
    # Cached so the same checkpointer connection survives across Streamlit
    # reruns - otherwise every interaction would reopen the DB pool.
    return build_graph()


graph = get_graph()

if "selected_thread_id" not in st.session_state:
    st.session_state.selected_thread_id = None
if "last_run_state" not in st.session_state:
    st.session_state.last_run_state = None
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

NODE_LABELS = {
    "guardrail": "Checking your request",
    "planner": "Deciding what to research",
    "flights": "Searching flights",
    "hotels": "Searching hotels",
    "activities": "Searching activities",
    "visa": "Checking visa requirements",
    "synthesizer": "Putting the itinerary together",
    "critic": "Reviewing against your budget",
}


def _stream_and_save(thread_id: str, input_obj, status_label: str) -> None:
    """Stream one run to completion, a rejection, OR a human-in-the-loop pause.
    input_obj is either a full/partial state dict or a Command(resume=...).
    On an interrupt, stashes pending_approval and returns without saving; the
    main area then renders the approval UI, and _resume_after_approval()
    continues the same thread."""
    config = {"configurable": {"thread_id": thread_id}}
    steps_log = st.session_state.get("last_run_trace") or []
    rejected_reason = None
    interrupt_payload = None

    with st.status(status_label, expanded=True) as status:
        for step in graph.stream(input_obj, config=config, stream_mode="updates"):
            if "__interrupt__" in step:
                interrupt_payload = step["__interrupt__"][0].value
                break
            for node_name, update in step.items():
                steps_log.append((node_name, update))
                status.write(NODE_LABELS.get(node_name, node_name))
                if node_name == "guardrail" and update.get("rejected"):
                    rejected_reason = update.get("rejection_reason", "Request rejected.")
                if node_name == "critic" and not update.get("approved", True):
                    status.write(f"↳ {update.get('critic_feedback', '')}")

        if rejected_reason:
            status.update(label="Request rejected", state="error")
        elif interrupt_payload:
            status.update(label="Waiting for your approval", state="running")
        else:
            status.update(label="Trip planned", state="complete")

    st.session_state.last_run_trace = steps_log

    if rejected_reason:
        st.session_state.pending_approval = None
        st.error(f"Your request couldn't be planned: {rejected_reason}")
        return

    if interrupt_payload:
        st.session_state.pending_approval = {
            "thread_id": thread_id,
            "proposed_domains": interrupt_payload.get("proposed_domains", []),
        }
        return

    st.session_state.pending_approval = None
    final_state = graph.get_state(config).values
    save_trip(thread_id, final_state)
    st.session_state.selected_thread_id = thread_id
    st.session_state.last_run_state = final_state


def run_new_trip(destination: str, budget_usd: int, nights: int) -> None:
    thread_id = f"{destination.strip().lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
    st.session_state.selected_thread_id = None
    st.session_state.last_run_state = None
    st.session_state.last_run_trace = None
    st.session_state.pending_approval = None
    # auto_approve=False so the graph pauses after the planner for sign-off.
    _stream_and_save(thread_id, default_trip_state(destination, budget_usd, nights, auto_approve=False),
                      f"Planning your trip to {destination}...")


def _resume_after_approval(thread_id: str, approved_domains: list[str]) -> None:
    _stream_and_save(thread_id, Command(resume={"domains": approved_domains}),
                      "Finishing your trip...")


def refine_trip(thread_id: str, message: str) -> None:
    # Multi-turn: SAME thread_id, partial update only - destination/budget/
    # nights/etc are restored from the checkpoint, not re-sent here.
    st.session_state.pending_approval = None
    _stream_and_save(thread_id, refinement_update(message), "Updating your trip...")


def render_itinerary(destination: str, nights: int, budget_usd: int, itinerary: str,
                      total_cost: int, approved: bool, revision_count: int,
                      warnings: list[str] = ()) -> None:
    st.subheader(f"{destination} · {nights} nights")

    for warning in warnings:
        st.info(f"Note: {warning}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total cost", f"${total_cost}")
    col2.metric("Budget", f"${budget_usd}", delta=f"{budget_usd - total_cost:+d}",
                delta_color="normal" if total_cost <= budget_usd else "inverse")
    col3.metric("Revisions", revision_count)

    if total_cost <= budget_usd:
        st.success("Within budget.")
    elif approved:
        st.warning(f"Best effort after {revision_count} revision(s) — still over budget.")
    else:
        st.error("Over budget.")

    st.divider()

    for line in itinerary.split("\n"):
        if not line.startswith("- "):
            continue
        domain, _, text = line[2:].partition(": ")
        with st.container(border=True):
            st.markdown(f"**{domain}**")
            st.write(text)


with st.sidebar:
    st.header("Plan a new trip")
    destination = st.text_input("Destination", value="")
    budget_usd = st.number_input("Budget (USD)", min_value=100, value=1500, step=50)
    nights = st.number_input("Nights", min_value=1, value=3, step=1)

    if st.button("Plan my trip", type="primary"):
        if not destination.strip():
            st.error("Enter a destination.")
        else:
            run_new_trip(destination.strip(), int(budget_usd), int(nights))
            st.rerun()

    st.divider()
    st.subheader("Trip history")
    trips = list_trips()
    if not trips:
        st.caption("No trips yet — plan one above.")
    for trip in trips:
        status_word = "Within budget" if trip["total_cost"] <= trip["budget_usd"] else "Over budget"
        label = f"{trip['destination']} · ${trip['total_cost']} · {status_word}"
        if st.button(label, key=trip["thread_id"], use_container_width=True):
            st.session_state.selected_thread_id = trip["thread_id"]
            st.session_state.last_run_state = None
            st.session_state.last_run_trace = None
            st.session_state.pending_approval = None
            st.rerun()

pending = st.session_state.get("pending_approval")

if pending:
    st.title("🧭 Multi-Agent Travel Planner")
    st.subheader("Approve the research plan")
    st.write(
        "The planner proposes researching these areas. Adjust the selection if "
        "you like, then approve to run the searches."
    )
    ALL = ["flights", "hotels", "activities", "visa"]
    chosen = st.multiselect(
        "Research areas", options=ALL,
        default=pending["proposed_domains"],
    )
    col_a, _ = st.columns([1, 3])
    if col_a.button("Approve & run", type="primary"):
        if not chosen:
            st.error("Pick at least one area.")
        else:
            _resume_after_approval(pending["thread_id"], chosen)
            st.rerun()
elif st.session_state.selected_thread_id is None:
    st.title("🧭 Multi-Agent Travel Planner")
    st.write("Plan a trip using the form in the sidebar, or open a past trip from your history.")
else:
    if st.session_state.last_run_state is not None:
        state = st.session_state.last_run_state
    else:
        state = get_trip(st.session_state.selected_thread_id)

    if state is None:
        st.error("That trip could not be found.")
    else:
        render_itinerary(
            destination=state["destination"],
            nights=state["nights"],
            budget_usd=state["budget_usd"],
            itinerary=state["itinerary"],
            total_cost=state["total_cost"],
            approved=state["approved"],
            revision_count=state["revision_count"],
            warnings=state.get("warnings", []),
        )

        st.divider()
        st.markdown("**Refine this trip**")
        refinement_message = st.text_input(
            "e.g. \"make it cheaper\", \"add a day trip\", \"skip the hotel research\"",
            key=f"refine-{st.session_state.selected_thread_id}",
            label_visibility="collapsed",
        )
        if st.button("Update trip", key=f"refine-btn-{st.session_state.selected_thread_id}"):
            if not refinement_message.strip():
                st.error("Enter a refinement request.")
            else:
                refine_trip(st.session_state.selected_thread_id, refinement_message.strip())
                st.rerun()

        if st.session_state.get("last_run_trace"):
            with st.expander("Agent trace (technical)"):
                for i, (node_name, update) in enumerate(st.session_state.last_run_trace, start=1):
                    st.markdown(f"**Step {i}: `{node_name}`**")
                    st.json(update)
