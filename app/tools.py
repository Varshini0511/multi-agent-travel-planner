"""
Real research tools, backed by Tavily web search + an LLM extraction step -
PLUS genuine dynamic tool selection: each domain gets a small agent loop
that decides, for itself, whether to search the web, convert a price to
USD, or stop and answer, repeating up to MAX_TOOL_STEPS times.

Why this isn't built on LangChain's native tool-calling (create_agent /
bind_tools): tested directly against this Groq-hosted Llama-3.3 model, and
it reliably generates malformed function-call syntax
(`<function=web_search{"query": "..."}</function>`, missing a closing `>`)
that Groq's own API then rejects with a 400 - reproducible even with a
single tool bound, and NOT fixed by retrying (temperature=0 means the same
broken output every time) or by disabling parallel tool calls. Rather than
depend on a provider/model combination that's provably unreliable at native
tool-calling, the loop below reuses the one mechanism that HAS been 100%
reliable throughout this app: `.with_structured_output()`. The LLM is asked,
each turn, to choose ONE action from a fixed set (the same shape as picking
a tool), we execute it in plain Python, and feed the result back in. This
is the identical decide -> act -> observe loop as 02_single_agent.py's
agent_brain/tool_node/route - just with a real model making the choice.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field
from tavily import TavilyClient

from app.config import get_secret
from app.llm import structured

tavily = TavilyClient(api_key=get_secret("TAVILY_API_KEY"))

MAX_RESULTS_PER_DOMAIN = 3
MAX_CHARS_PER_RESULT = 600  # keeps LLM input (and Groq's free-tier token budget) small
MAX_TOOL_STEPS = 3          # caps how many actions one domain's agent can take


# ---------------------------------------------------------------------------
# The two tools available to every domain's research loop.
# ---------------------------------------------------------------------------
def _web_search(query: str) -> str:
    response = tavily.search(query=query, max_results=MAX_RESULTS_PER_DOMAIN)
    return "\n\n".join(r["content"][:MAX_CHARS_PER_RESULT] for r in response.get("results", []))


FIXED_RATES_TO_USD = {
    "eur": 1.08, "gbp": 1.27, "vnd": 0.00004, "jpy": 0.0067,
    "inr": 0.012, "chf": 1.13, "aed": 0.27, "thb": 0.028, "usd": 1.0,
}


def _convert_to_usd(amount: float, currency: str) -> str:
    rate = FIXED_RATES_TO_USD.get(currency.strip().lower())
    if rate is None:
        return f"Unknown currency code '{currency}'; treat the amount as already being in USD."
    return f"{amount} {currency.upper()} is approximately ${round(amount * rate, 2)} USD."


# ---------------------------------------------------------------------------
# The "which tool next" decision - a structured choice, not native tool-calling.
# ---------------------------------------------------------------------------
class ToolChoice(BaseModel):
    action: Literal["web_search", "convert_to_usd", "done"] = Field(
        description="Which action to take next, or 'done' if enough information has been gathered")
    search_query: str = Field(default="", description="The query to search, only if action is 'web_search'")
    amount: float = Field(default=0.0, description="The amount to convert, only if action is 'convert_to_usd'")
    currency: str = Field(default="", description="3-letter currency code to convert from, only if action is 'convert_to_usd'")


tool_choice_llm = structured(ToolChoice).with_retry(
    stop_after_attempt=5, wait_exponential_jitter=True
)


class CostEstimate(BaseModel):
    summary: str = Field(description="A one or two sentence, traveler-facing summary of the findings")
    # A plain string, not int: Groq's strict tool-call schema sometimes emits a
    # numeric-looking string (e.g. "70") which fails strict integer validation
    # server-side. Letting the model write loosely and parsing here is more
    # robust than fighting its exact output format.
    cost_usd: str = Field(description="A single realistic USD cost estimate, digits only, e.g. '480'")


extractor = structured(CostEstimate).with_retry(
    stop_after_attempt=5, wait_exponential_jitter=True
)


def _parse_cost(raw: str) -> int:
    digits = re.sub(r"[^\d]", "", raw)
    return int(digits) if digits else 0


DOMAIN_TASKS = {
    "flights": "Research the cost of round-trip flights to {destination} for a {nights}-night trip.",
    "hotels": "Research the average per-night cost of a mid-range hotel in {destination}, "
              "then estimate the total for {nights} nights.",
    "activities": "Research popular tourist activities in {destination} and their typical costs.",
    "visa": "Research tourist visa requirements and fees for visiting {destination}.",
}

# Graceful degradation: if the loop or the final extraction fails even after
# retries (API outage, not just a transient rate limit), fall back to a
# rough flat estimate rather than crashing the whole trip-planning run.
FALLBACK_ESTIMATES = {
    "flights": lambda destination, nights: (
        f"Live flight search was unavailable, so this is a rough placeholder estimate for {destination}.", 480),
    "hotels": lambda destination, nights: (
        f"Live hotel search was unavailable, so this is a rough placeholder estimate for {destination} "
        f"({nights} nights at an assumed $110/night).", 110 * nights),
    "activities": lambda destination, nights: (
        f"Live activity search was unavailable, so this is a rough placeholder estimate for {destination}.", 150),
    "visa": lambda destination, nights: (
        f"Live visa search was unavailable, so this is a rough placeholder estimate for {destination}.", 25),
}


def _run_research_loop(domain: str, destination: str, nights: int) -> tuple[str, int, bool]:
    """Returns (summary, cost_usd, degraded)."""
    try:
        task = DOMAIN_TASKS[domain].format(destination=destination, nights=nights)
        transcript = [f"Task: {task}"]

        for _ in range(MAX_TOOL_STEPS):
            choice = tool_choice_llm.invoke(
                "\n\n".join(transcript) +
                "\n\nDecide the next action. Call web_search with a specific query if you still "
                "need information. Call convert_to_usd if a price you found is in a non-USD "
                "currency. Respond 'done' once you have enough information to answer."
            )
            if choice.action == "web_search":
                result = _web_search(choice.search_query)
                print(f"[{domain}] tool call: web_search({choice.search_query!r})")
                transcript.append(f"web_search({choice.search_query!r}) ->\n{result}")
            elif choice.action == "convert_to_usd":
                result = _convert_to_usd(choice.amount, choice.currency)
                print(f"[{domain}] tool call: convert_to_usd({choice.amount}, {choice.currency!r}) -> {result}")
                transcript.append(f"convert_to_usd({choice.amount}, {choice.currency!r}) -> {result}")
            else:
                print(f"[{domain}] tool call: done")
                break

        estimate = extractor.invoke(
            "\n\n".join(transcript) +
            "\n\nBased on the research above, write a short traveler-facing summary "
            "and give one realistic total USD cost estimate."
        )
        return estimate.summary, _parse_cost(estimate.cost_usd), False
    except Exception as exc:
        print(f"[tools] {domain} research loop failed ({exc!r}); using fallback estimate")
        summary, cost = FALLBACK_ESTIMATES[domain](destination, nights)
        return summary, cost, True


def search_flights(destination: str, nights: int) -> tuple[str, int, bool]:
    return _run_research_loop("flights", destination, nights)


def search_hotels(destination: str, nights: int) -> tuple[str, int, bool]:
    return _run_research_loop("hotels", destination, nights)


def search_activities(destination: str, nights: int) -> tuple[str, int, bool]:
    return _run_research_loop("activities", destination, nights)


def search_visa(destination: str, nights: int) -> tuple[str, int, bool]:
    return _run_research_loop("visa", destination, nights)


TOOLS_BY_DOMAIN = {
    "flights": search_flights,
    "hotels": search_hotels,
    "activities": search_activities,
    "visa": search_visa,
}
