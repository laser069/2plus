"""Latency benchmark: direct provider call vs full app pipeline.

Isolates where the app lags a raw model call, for both Ollama (local) and
OpenRouter (cloud). Reports cold (first, post-eviction) vs warm (resident) runs
so model-load thrash is separated from real inference + pipeline overhead.

Run:  python -m bench.latency_bench
      python -m bench.latency_bench --warm 3 --ollama-model qwen3:8b \
             --or-model tencent/hy3:free

Reuses the app's own client/agent so the comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import ollama

# Windows consoles default to cp1252 and choke on Δ/→/—. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from config.settings import (
    OLLAMA_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    LOG_DIR,
)
from serving.llm_client import LLMClient
from orchestrator.agent import Agent
from memory.convo import ConvoMemory

# Prompts chosen to exercise different router paths.
PROMPTS = {
    "direct": "hi how are you",
    "browser": "what is the latest news about AI today",
    "ambiguous": "remember my notes and search online for the current price",
}

_raw_ollama = ollama.Client(host=OLLAMA_BASE_URL)


# ── timing primitives ─────────────────────────────────────────────────────────

def _time_stream(gen) -> tuple[float, float, int]:
    """Consume a token generator. Returns (ttft_ms, total_ms, n_tokens)."""
    t0 = time.perf_counter()
    ttft: float | None = None
    n = 0
    for tok in gen:
        if not tok:
            continue
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
        n += 1
    total = (time.perf_counter() - t0) * 1000
    return (ttft if ttft is not None else total), total, n


# ── direct provider calls (no app pipeline) ───────────────────────────────────

def direct_ollama(model: str, prompt: str) -> tuple[float, float, int]:
    def gen():
        for chunk in _raw_ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            yield (chunk.message.content or "") if chunk.message else ""
    return _time_stream(gen())


def direct_openrouter(model: str, prompt: str) -> tuple[float, float, int]:
    from openai import OpenAI
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)

    def gen():
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices:
                yield chunk.choices[0].delta.content or ""
    return _time_stream(gen())


# ── full app pipeline ─────────────────────────────────────────────────────────

def app_path(model: str, prompt: str) -> tuple[float, float, int]:
    """Agent.run_stream — full routing/context/pipeline. Fresh memory each call
    so state doesn't leak between runs."""
    agent = Agent(ConvoMemory())

    def gen():
        for ev in agent.run_stream(prompt, model=model):
            if isinstance(ev, str):
                yield ev
            # dict events (routing/tool/done) carry no answer tokens
    return _time_stream(gen())


# ── cold/warm orchestration ───────────────────────────────────────────────────

def _ollama_stop(model: str) -> None:
    """Evict a model from VRAM so the next call is a genuine cold load."""
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass  # best-effort; harness still measures warm runs


def measure(label: str, fn, model: str, prompt: str, warm: int,
            cold: bool) -> dict:
    """Run one cold pass (optional) + `warm` warm passes; summarise."""
    runs: list[dict] = []

    if cold:
        _ollama_stop(model)
        ttft, total, n = fn(model, prompt)
        runs.append({"kind": "cold", "ttft_ms": ttft, "total_ms": total, "tokens": n})

    warm_ttft: list[float] = []
    warm_total: list[float] = []
    for _ in range(warm):
        ttft, total, n = fn(model, prompt)
        runs.append({"kind": "warm", "ttft_ms": ttft, "total_ms": total, "tokens": n})
        warm_ttft.append(ttft)
        warm_total.append(total)

    return {
        "label": label,
        "model": model,
        "prompt": prompt,
        "cold_ttft_ms": runs[0]["ttft_ms"] if cold else None,
        "cold_total_ms": runs[0]["total_ms"] if cold else None,
        "warm_ttft_ms_median": statistics.median(warm_ttft) if warm_ttft else None,
        "warm_total_ms_median": statistics.median(warm_total) if warm_total else None,
        "runs": runs,
    }


def _fmt(v) -> str:
    return "—" if v is None else f"{v:,.0f}"


def print_table(results: list[dict]) -> None:
    hdr = f"{'path':<20}{'prompt':<12}{'cold_ttft':>12}{'cold_tot':>12}{'warm_ttft':>12}{'warm_tot':>12}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['label']:<20}{r['prompt_key']:<12}"
            f"{_fmt(r['cold_ttft_ms']):>12}{_fmt(r['cold_total_ms']):>12}"
            f"{_fmt(r['warm_ttft_ms_median']):>12}{_fmt(r['warm_total_ms_median']):>12}"
        )
    print("\n(all values ms; ttft = time-to-first-token)")

    # Delta summary: app overhead vs direct, warm.
    print("\nΔ app overhead vs direct (warm ttft, ms):")
    by_key: dict[str, dict[str, float]] = {}
    for r in results:
        by_key.setdefault(r["prompt_key"], {})[r["label"]] = r["warm_ttft_ms_median"]
    for key, m in by_key.items():
        for prov, direct_l, app_l in (
            ("ollama", "direct_ollama", "app_ollama"),
            ("openrouter", "direct_openrouter", "app_openrouter"),
        ):
            d, a = m.get(direct_l), m.get(app_l)
            if d and a:
                print(f"  {key:<12} {prov:<11} direct={d:,.0f}  app={a:,.0f}  Δ={a - d:+,.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", type=int, default=3, help="warm runs per path")
    ap.add_argument("--no-cold", action="store_true", help="skip cold-load runs")
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--or-model", default="tencent/hy3:free")
    ap.add_argument("--skip-openrouter", action="store_true")
    ap.add_argument("--prompts", nargs="*", default=list(PROMPTS.keys()),
                    help="subset of: " + ", ".join(PROMPTS))
    args = ap.parse_args()

    from config.settings import MODEL_ROUTER
    ollama_model = args.ollama_model or MODEL_ROUTER["default"]
    cold = not args.no_cold

    results: list[dict] = []
    for key in args.prompts:
        prompt = PROMPTS[key]

        r = measure("direct_ollama", direct_ollama, ollama_model, prompt, args.warm, cold)
        r["prompt_key"] = key
        results.append(r)

        r = measure("app_ollama", app_path, ollama_model, prompt, args.warm, cold)
        r["prompt_key"] = key
        results.append(r)

        if not args.skip_openrouter:
            if not OPENROUTER_API_KEY:
                print("! OPENROUTER_API_KEY not set — skipping OpenRouter paths")
                args.skip_openrouter = True
            else:
                # cloud model never resident locally → cold flag irrelevant
                r = measure("direct_openrouter", direct_openrouter, args.or_model, prompt, args.warm, False)
                r["prompt_key"] = key
                results.append(r)
                r = measure("app_openrouter", app_path, args.or_model, prompt, args.warm, False)
                r["prompt_key"] = key
                results.append(r)

    print_table(results)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(LOG_DIR) / f"bench_{ts}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nraw → {out}")


if __name__ == "__main__":
    main()
