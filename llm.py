"""
llm.py — the ONE place our app talks to the local AI model.

Everyone else (prompts.py, app.py, retriever.py) should call
`generate_response()` from here instead of calling Ollama directly.
That way, if we change the model or settings, we only change this file.

How it works:
    1. Ollama runs in the background on this laptop (port 11434).
    2. We send it a list of chat messages.
    3. It runs Qwen3-4B locally and sends back the reply text.
No internet or paid API is used.
"""

import os
import re

import httpx
from ollama import Client, ResponseError

# ---------------------------------------------------------------------------
# Settings (change these here, not in other files)
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemma3:12b"  # = gemma3:12b-it-q4_K_M, 8.2 GB (this folder's experiment)
# The original project (../travel-phrase-tutor) uses "qwen3:4b-instruct".

# Sampling settings recommended on the Qwen3-4B-Instruct-2507 model card.
# num_ctx = how many tokens the model can "see" at once (prompt + reply).
# 8192 leaves room for retrieved RAG passages plus a short conversation.
DEFAULT_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "num_ctx": 8192,
    "num_predict": 1500,   # cap reply length so one runaway reply can't block Ollama
}

# How long Ollama keeps the model in memory after a call, so the next call
# is fast instead of reloading it.
DEFAULT_KEEP_ALIVE = "10m"

# Where Ollama is running. Override with the OLLAMA_HOST environment variable
# (e.g. if we ever use the MSU GPU machine as a backup).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

VALID_ROLES = {"system", "user", "assistant"}

# Give up on a single request after this many seconds, so a stuck Ollama shows
# an error instead of leaving the app spinning forever.
REQUEST_TIMEOUT = 120


class LLMError(Exception):
    """Raised with a plain-English message when the model call fails."""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def generate_response(messages, model_config=None) -> str:
    """
    Send a conversation to the local model and return its reply as text.

    messages: list of dicts, oldest first, e.g.
        [
            {"role": "system", "content": "You are a friendly Spanish tutor."},
            {"role": "user", "content": "How do I order a coffee?"},
        ]
    model_config: optional dict to override defaults, e.g.
        {"temperature": 0.2}          -> less random replies
        {"model": "qwen3:4b-instruct"} -> pick a different Ollama model
        {"format": {...JSON schema...}} -> force the reply to be JSON that
                                           matches the schema (structured output)

    Returns: the model's reply (a string).
    Raises: LLMError if Ollama isn't running, the model isn't downloaded,
            or the messages are malformed.
    """
    return generate_response_with_stats(messages, model_config)["text"]


def generate_response_with_stats(messages, model_config=None) -> dict:
    """
    Same as generate_response(), but also returns timing numbers we need
    for the hardware/latency section of the report.

    Returns a dict:
        text              the reply
        model             model name used
        total_seconds     whole request, start to finish
        load_seconds      time spent loading the model into memory
                          (large on the first call, ~0 after)
        prompt_tokens     tokens in our input
        output_tokens     tokens the model generated
        tokens_per_second generation speed
    """
    _check_messages(messages)
    model, options, keep_alive, fmt = _split_config(model_config)

    client = Client(host=OLLAMA_HOST, timeout=REQUEST_TIMEOUT)
    try:
        response = client.chat(
            model=model,
            messages=messages,
            options=options,
            keep_alive=keep_alive,
            format=fmt,
        )
    except httpx.TimeoutException as e:
        raise LLMError(
            f"Ollama didn't answer within {REQUEST_TIMEOUT} seconds. It may be stuck: "
            "restart it with  brew services restart ollama"
        ) from e
    except ConnectionError as e:
        raise LLMError(
            f"Can't reach Ollama at {OLLAMA_HOST}. Is it running? "
            "Start it with:  brew services start ollama"
        ) from e
    except ResponseError as e:
        if e.status_code == 404:
            raise LLMError(
                f"Model '{model}' isn't downloaded. Run:  ollama pull {model}"
            ) from e
        raise LLMError(f"Ollama returned an error: {e.error}") from e

    # Ollama reports durations in nanoseconds; convert to seconds.
    eval_seconds = _ns_to_s(response.eval_duration)
    output_tokens = response.eval_count or 0
    return {
        # Strip empty/leftover <think> blocks (the LoRA model learned to emit them).
        "text": re.sub(r"<think>.*?</think>\s*", "", response.message.content or "", flags=re.S).strip(),
        "model": model,
        "total_seconds": _ns_to_s(response.total_duration),
        "load_seconds": _ns_to_s(response.load_duration),
        "prompt_tokens": response.prompt_eval_count or 0,
        "output_tokens": output_tokens,
        "tokens_per_second": (
            round(output_tokens / eval_seconds, 1) if eval_seconds else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Helpers (internal — other files shouldn't need these)
# ---------------------------------------------------------------------------

def _check_messages(messages):
    """Catch common mistakes early with a clear message."""
    if not isinstance(messages, list) or not messages:
        raise LLMError("messages must be a non-empty list of dicts.")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise LLMError(f"messages[{i}] needs 'role' and 'content' keys.")
        if m["role"] not in VALID_ROLES:
            raise LLMError(
                f"messages[{i}] has role '{m['role']}'; "
                f"use one of {sorted(VALID_ROLES)}."
            )


def _split_config(model_config):
    """Merge user overrides with defaults; pull out non-sampling keys."""
    config = dict(model_config or {})
    model = config.pop("model", DEFAULT_MODEL)
    keep_alive = config.pop("keep_alive", DEFAULT_KEEP_ALIVE)
    fmt = config.pop("format", None)  # None = free text; dict = JSON schema
    options = {**DEFAULT_OPTIONS, **config}
    return model, options, keep_alive, fmt


def _ns_to_s(nanoseconds):
    return round((nanoseconds or 0) / 1e9, 2)


def unload(model: str) -> None:
    """Free a model's memory right away (keep_alive=0), e.g. when the app
    switches models. Errors are ignored: unloading is only housekeeping."""
    try:
        Client(host=OLLAMA_HOST, timeout=30).generate(model=model, prompt="", keep_alive=0)
    except Exception:
        pass
