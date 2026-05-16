"""Small OpenAI-compatible client helpers for GENE-Arena PES judges."""

from __future__ import annotations

import os
import time

import httpx


def _make_httpx(timeout: float = 300.0) -> httpx.Client:
    return httpx.Client(trust_env=False, timeout=httpx.Timeout(timeout, connect=15.0))


def _model_prefers_completion_tokens(model: str) -> bool:
    name = (model or "").lower()
    return name.startswith("gpt-5") or name.startswith("o")


def _supports_custom_temperature(model: str) -> bool:
    name = (model or "").lower()
    return not (name.startswith("gpt-5") or name.startswith("o"))


def _completion_token_kwargs(token_budget: int, prefer_completion: bool = True) -> dict:
    return {"max_completion_tokens": token_budget} if prefer_completion else {"max_tokens": token_budget}


def _remove_token_args(call: dict) -> None:
    call.pop("max_completion_tokens", None)
    call.pop("max_tokens", None)


def _error_requests_completion_tokens(message: str) -> bool:
    msg = (message or "").lower()
    return (
        "use 'max_completion_tokens'" in msg
        or 'use "max_completion_tokens"' in msg
        or (
            ("unsupported parameter: 'max_tokens'" in msg or "unsupported parameter: \"max_tokens\"" in msg)
            and "max_completion_tokens" in msg
        )
    )


def _error_requests_max_tokens(message: str) -> bool:
    msg = (message or "").lower()
    return (
        "use 'max_tokens'" in msg
        or 'use "max_tokens"' in msg
        or (
            (
                "unsupported parameter: 'max_completion_tokens'" in msg
                or "unsupported parameter: \"max_completion_tokens\"" in msg
            )
            and "max_tokens" in msg
        )
    )


def _is_rate_limit_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status == 429:
        return True
    response = getattr(error, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    msg = str(error).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg or "rate_limit" in msg


def _retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        value = None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except Exception:
        return None


def _create_with_rate_limit_retries(create_fn, call: dict):
    retries = int(os.environ.get("GENE_ARENA_LLM_RETRIES", "4"))
    base = float(os.environ.get("GENE_ARENA_LLM_RETRY_BASE_SECONDS", "8"))
    max_sleep = float(os.environ.get("GENE_ARENA_LLM_RETRY_MAX_SECONDS", "90"))
    for attempt in range(retries + 1):
        try:
            return create_fn(**call)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt >= retries:
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = min(max_sleep, base * (2 ** attempt))
            print(f"  [LLM] rate limited; retry {attempt + 1}/{retries} after {delay:.1f}s")
            time.sleep(delay)


def chat_completion_create(
    client,
    *,
    model: str,
    messages: list[dict],
    token_budget: int,
    temperature: float | None = None,
    prefer_completion: bool = True,
    **kwargs,
):
    create_fn = client.chat.completions.create
    call = {
        "model": model,
        "messages": messages,
        **_completion_token_kwargs(token_budget, prefer_completion=prefer_completion),
        **kwargs,
    }
    if temperature is not None and _supports_custom_temperature(model):
        call["temperature"] = temperature
    try:
        return _create_with_rate_limit_retries(create_fn, call)
    except TypeError as exc:
        msg = str(exc)
        if "max_completion_tokens" in call and (
            "unexpected keyword" in msg or "unexpected argument" in msg or "got an unexpected" in msg
        ):
            _remove_token_args(call)
            call["extra_body"] = {"max_completion_tokens": token_budget}
            return _create_with_rate_limit_retries(create_fn, call)
        if _error_requests_completion_tokens(msg):
            _remove_token_args(call)
            call["max_completion_tokens"] = token_budget
            return _create_with_rate_limit_retries(create_fn, call)
        if _error_requests_max_tokens(msg):
            _remove_token_args(call)
            call["max_tokens"] = token_budget
            return _create_with_rate_limit_retries(create_fn, call)
        raise
    except Exception as exc:
        msg = str(exc)
        if _error_requests_completion_tokens(msg):
            _remove_token_args(call)
            call["max_completion_tokens"] = token_budget
            return _create_with_rate_limit_retries(create_fn, call)
        if _error_requests_max_tokens(msg):
            _remove_token_args(call)
            call["max_tokens"] = token_budget
            return _create_with_rate_limit_retries(create_fn, call)
        raise


def make_client(provider_cfg: dict):
    """Create a client exposing chat.completions.create()."""
    from openai import OpenAI

    return OpenAI(
        api_key=provider_cfg.get("api_key", ""),
        base_url=provider_cfg.get("base_url"),
        http_client=_make_httpx(),
    )
