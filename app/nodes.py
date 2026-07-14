"""
All the reasoning nodes. planner and critic now call a real LLM (Groq) with
structured output; the subagents call real Tavily search (via tools.py).
The graph shape - fan-out, fan-in, the critic->planner loop - is unchanged
from the mocked version; only what happens INSIDE each node changed.
"""
from typing import Literal

from pydantic import BaseModel, Field

from app.llm import structured
from app.state import TripState
from app.tools import TOOLS_BY_DOMAIN

MAX_REVISIONS = 2
Domain = Literal["flights", "hotels", "activities", "visa"]


# ---------------------------------------------------------------------------
# PLANNER: an LLM call that decomposes the request into a list of independent
# subtasks. On a revision loop, it sees the critic's feedback and adapts.
# ---------------------------------------------------------------------------
class PlanDecision(BaseModel):
    domains: list[Domain] = Field(description="Which research domains are needed for this trip")
    reasoning: str = Field(description="One sentence explaining the domain choices")


planner_llm = structured(PlanDecision).with_retry(
    stop_after_attempt=5, wait_exponential_jitter=True
)


def _fallback_domains(state: TripState) -> list[str]:
    """Used only if the planner LLM is unreachable - the same conservative
    logic the original mocked (pre-AI) version used. Never guesses 'visa',
    since getting that wrong silently would be worse than omitting it."""
    domains = ["flights", "hotels"]
    if state["revision_count"] == 0:
        domains.append("activities")
    return domains


def planner(state: TripState) -> dict:
    refinement = state.get("refinement_request", "")

    if refinement and state["revision_count"] == 0:
        # Multi-turn: this is a follow-up against an EXISTING trip, not a
        # fresh plan. state['itinerary'] here is still last run's result -
        # the checkpointer restored it, since this invoke only sent a
        # partial update (see app/state.py:refinement_update).
        prompt = (
            f"You previously planned a trip to {state['destination']} ({state['nights']} nights, "
            f"budget ${state['budget_usd']}). The current itinerary is:\n{state['itinerary']}\n\n"
            f"The traveler now says: '{refinement}'\n\n"
            f"Decide the updated list of research domains needed to satisfy this request. "
            f"Keep 'flights' and 'hotels' unless the traveler specifically asks to remove them."
        )
    elif state["revision_count"] == 0:
        prompt = (
            f"Plan the research domains for a trip to {state['destination']} "
            f"for {state['nights']} nights, budget ${state['budget_usd']}. "
            f"Assume a traveler with a US passport unless the destination implies otherwise. "
            f"Always include 'flights' and 'hotels'. Include 'activities' unless the trip is "
            f"clearly business-only. Include 'visa' only if this destination actually requires "
            f"a visa or e-visa for most tourists."
        )
    else:
        prompt = (
            f"You already planned a trip to {state['destination']} ({state['nights']} nights, "
            f"budget ${state['budget_usd']}), and it was rejected: '{state['critic_feedback']}'. "
            f"Decide a revised list of research domains that will bring the cost down. "
            f"Keep 'flights' and 'hotels' - those are non-negotiable. Drop or keep 'activities' "
            f"and 'visa' based on what will help fit the budget."
        )

    try:
        decision = planner_llm.invoke(prompt)
        print(f"[planner] domains={decision.domains} ({decision.reasoning})")
        return {"domains": decision.domains}
    except Exception as exc:
        domains = _fallback_domains(state)
        print(f"[planner] LLM unavailable ({exc!r}); falling back to default domains={domains}")
        return {
            "domains": domains,
            "warnings": ["Planner AI was unavailable; used a default research plan."],
        }


def route_to_domains(state: TripState) -> list[str]:
    # Returning a LIST here is what triggers LangGraph's parallel fan-out:
    # every node named in this list runs concurrently in the same step.
    return state["domains"]


# ---------------------------------------------------------------------------
# CONTEXT ENGINEERING: each subagent gets a narrow, purpose-built context -
# NOT the full TripState, and NOT the planner's or critic's reasoning.
# ---------------------------------------------------------------------------
def build_domain_context(state: TripState, domain: str) -> dict:
    return {
        "destination": state["destination"],
        "nights": state["nights"],
        "domain": domain,
    }


def make_subagent(domain: str):
    """Factory so flights/hotels/activities/visa share one implementation."""
    def subagent(state: TripState) -> dict:
        ctx = build_domain_context(state, domain)
        summary, cost, degraded = TOOLS_BY_DOMAIN[domain](ctx["destination"], ctx["nights"])
        print(f"[{domain}] {summary} (${cost})")
        update = {f"{domain}_result": summary, f"{domain}_cost": cost}
        if degraded:
            update["warnings"] = [f"{domain.title()} used a fallback estimate (live search/AI was unavailable)."]
        return update
    return subagent


# ---------------------------------------------------------------------------
# SYNTHESIZER: fan-in point. Runs once after all triggered subagents finish,
# combining their independent outputs into one itinerary.
# ---------------------------------------------------------------------------
def synthesizer(state: TripState) -> dict:
    parts = []
    total = 0
    for domain in state["domains"]:
        result = state.get(f"{domain}_result", "")
        cost = state.get(f"{domain}_cost", 0)
        if result:
            parts.append(f"- {domain.title()}: {result}")
            total += cost

    itinerary = "\n".join(parts)
    print(f"[synthesizer] combined {len(parts)} domain results, total ${total}")
    return {"itinerary": itinerary, "total_cost": total}


# ---------------------------------------------------------------------------
# CRITIC: an LLM call that reviews the synthesized plan against the budget
# and either approves it or sends it back with concrete feedback.
# ---------------------------------------------------------------------------
class CriticVerdict(BaseModel):
    approved: bool = Field(description="True if the plan is acceptable, false if it needs revision")
    feedback: str = Field(description="If not approved, concrete guidance for the planner on what to change")


critic_llm = structured(CriticVerdict).with_retry(
    stop_after_attempt=5, wait_exponential_jitter=True
)


def _fallback_verdict(state: TripState) -> tuple[bool, str]:
    """Used only if the critic LLM is unreachable - a direct, deterministic
    budget comparison, same as the original mocked (pre-AI) critic."""
    over_by = state["total_cost"] - state["budget_usd"]
    if over_by <= 0:
        return True, "within budget"
    return False, f"over budget by ${over_by}"


def critic(state: TripState) -> dict:
    if state["revision_count"] >= MAX_REVISIONS:
        print(f"[critic] approved (max revisions reached, ${state['total_cost']} vs ${state['budget_usd']} budget)")
        return {"approved": True, "critic_feedback": "within budget"}

    warnings_update = {}
    try:
        verdict = critic_llm.invoke(
            f"Budget: ${state['budget_usd']}. Total planned cost: ${state['total_cost']}.\n"
            f"Itinerary:\n{state['itinerary']}\n\n"
            f"Is this plan acceptable? Approve if the total cost is at or under budget. "
            f"If not, give one concrete sentence of feedback on what to cut."
        )
        approved, feedback = verdict.approved, verdict.feedback
    except Exception as exc:
        print(f"[critic] LLM unavailable ({exc!r}); falling back to a direct budget comparison")
        approved, feedback = _fallback_verdict(state)
        warnings_update = {"warnings": ["Critic AI was unavailable; used a direct budget comparison."]}

    if approved:
        print(f"[critic] approved (${state['total_cost']} vs ${state['budget_usd']} budget)")
        return {"approved": True, "critic_feedback": feedback or "within budget", **warnings_update}

    print(f"[critic] rejected: {feedback} -> sending back to planner")
    return {
        "approved": False,
        "critic_feedback": feedback,
        "revision_count": state["revision_count"] + 1,
        **warnings_update,
    }


def route_after_critic(state: TripState) -> str:
    return "END" if state["approved"] else "planner"
