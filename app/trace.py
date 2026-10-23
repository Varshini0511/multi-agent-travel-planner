"""
Observability + token/cost tracking.

A LangChain callback handler that records every LLM call made during one graph
run - which model answered, how long it took, and how many tokens it used -
WITHOUT touching any node. You attach it by passing it in the run config:

    tracer = RunTracer()
    graph.invoke(state, config={"configurable": {...}, "callbacks": [tracer]})
    print(tracer.summary())

Because it hooks the chat-model layer, it transparently sees retries AND the
Groq->Gemini fallback as separate calls - so the summary also reveals when the
primary provider was unavailable (calls attributed to the Gemini model).

Token counts come straight from each provider's usage metadata. Groq and Gemini
free tiers bill $0, so cost here is informational; swap in real per-1M-token
rates in MODEL_RATES_USD_PER_1M if you move to a paid tier.
"""
import time

from langchain_core.callbacks import BaseCallbackHandler

# Informational only (free tiers are $0). Fill with real rates on a paid tier.
MODEL_RATES_USD_PER_1M = {
    # "model-name": (input_rate, output_rate),
}


class RunTracer(BaseCallbackHandler):
    def __init__(self):
        self.calls = []          # one dict per LLM call
        self._starts = {}        # run_id -> start time

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._starts[run_id] = time.time()

    # chat models route through on_chat_model_start, but LangChain still fires
    # on_llm_end for both - we only need the timer started, so cover both.
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._starts[run_id] = time.time()

    def on_llm_end(self, response, *, run_id, **kwargs):
        latency_ms = round((time.time() - self._starts.pop(run_id, time.time())) * 1000)
        model, inp, out, tot = "?", 0, 0, 0
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            usage = getattr(msg, "usage_metadata", None) if msg else None
            if usage:
                inp = usage.get("input_tokens", 0)
                out = usage.get("output_tokens", 0)
                tot = usage.get("total_tokens", inp + out)
            meta = getattr(msg, "response_metadata", {}) if msg else {}
            model = (response.llm_output or {}).get("model_name") or meta.get("model_name", "?")
        except Exception:
            pass
        self.calls.append({"model": model, "latency_ms": latency_ms,
                           "input_tokens": inp, "output_tokens": out, "total_tokens": tot})

    def _cost_usd(self) -> float:
        total = 0.0
        for c in self.calls:
            rate = MODEL_RATES_USD_PER_1M.get(c["model"])
            if rate:
                total += c["input_tokens"] / 1e6 * rate[0] + c["output_tokens"] / 1e6 * rate[1]
        return round(total, 4)

    def summary(self) -> dict:
        by_model = {}
        for c in self.calls:
            m = by_model.setdefault(c["model"], {"calls": 0, "tokens": 0})
            m["calls"] += 1
            m["tokens"] += c["total_tokens"]
        return {
            "llm_calls": len(self.calls),
            "total_tokens": sum(c["total_tokens"] for c in self.calls),
            "total_latency_ms": sum(c["latency_ms"] for c in self.calls),
            "est_cost_usd": self._cost_usd(),
            "by_model": by_model,
        }
