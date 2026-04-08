"""Retry utilities for resilient operations."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

import structlog

log = structlog.get_logger()

T = TypeVar("T")


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """Retry an async function with exponential backoff."""
    last_exception: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_attempts:
                log.error(
                    "retry.max_attempts_reached",
                    func=func.__name__,
                    attempts=max_attempts,
                    error=str(e),
                )
                raise
            delay = min(
                base_delay * (2 ** (attempt - 1)) if exponential else base_delay,
                max_delay,
            )
            log.warning(
                "retry.retrying",
                func=func.__name__,
                attempt=attempt,
                delay=delay,
                error=str(e),
            )
            await asyncio.sleep(delay)
    raise last_exception  # pragma: no cover – should never reach here
