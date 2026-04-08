"""Tests for teams_attendant.utils.retry."""

from __future__ import annotations

import time

import pytest

from teams_attendant.utils.retry import retry_async


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TrackedError(Exception):
    """A custom exception for testing retryable_exceptions filtering."""


class _NonRetryableError(Exception):
    """An exception that should never be retried."""


# ---------------------------------------------------------------------------
# Basic success / failure
# ---------------------------------------------------------------------------


class TestRetrySuccess:
    """retry_async should return the result when the function succeeds."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self) -> None:
        call_count = 0

        async def _ok() -> str:
            nonlocal call_count
            call_count += 1
            return "done"

        result = await retry_async(_ok, max_attempts=3)
        assert result == "done"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_succeeds_on_second_try(self) -> None:
        call_count = 0

        async def _flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise _TrackedError("transient")
            return "recovered"

        result = await retry_async(
            _flaky,
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(_TrackedError,),
        )
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_succeeds_on_last_attempt(self) -> None:
        call_count = 0

        async def _flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _TrackedError("transient")
            return "finally"

        result = await retry_async(
            _flaky,
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(_TrackedError,),
        )
        assert result == "finally"
        assert call_count == 3


# ---------------------------------------------------------------------------
# Exhausted retries
# ---------------------------------------------------------------------------


class TestRetryExhaustion:
    """retry_async should raise after max_attempts are exhausted."""

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self) -> None:
        call_count = 0

        async def _always_fail() -> str:
            nonlocal call_count
            call_count += 1
            raise _TrackedError(f"fail-{call_count}")

        with pytest.raises(_TrackedError, match="fail-3"):
            await retry_async(
                _always_fail,
                max_attempts=3,
                base_delay=0.01,
                retryable_exceptions=(_TrackedError,),
            )
        assert call_count == 3


# ---------------------------------------------------------------------------
# Exponential backoff timing
# ---------------------------------------------------------------------------


class TestExponentialBackoff:
    """Verify that delays grow exponentially."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self) -> None:
        timestamps: list[float] = []

        async def _fail() -> None:
            timestamps.append(time.monotonic())
            raise _TrackedError("boom")

        with pytest.raises(_TrackedError):
            await retry_async(
                _fail,
                max_attempts=3,
                base_delay=0.1,
                exponential=True,
                retryable_exceptions=(_TrackedError,),
            )

        assert len(timestamps) == 3
        # First delay ~0.1s (base_delay * 2^0)
        gap1 = timestamps[1] - timestamps[0]
        # Second delay ~0.2s (base_delay * 2^1)
        gap2 = timestamps[2] - timestamps[1]

        assert gap1 >= 0.08, f"First gap too short: {gap1}"
        assert gap2 >= 0.15, f"Second gap too short: {gap2}"
        assert gap2 > gap1, "Exponential backoff should increase delays"

    @pytest.mark.asyncio
    async def test_linear_backoff(self) -> None:
        timestamps: list[float] = []

        async def _fail() -> None:
            timestamps.append(time.monotonic())
            raise _TrackedError("boom")

        with pytest.raises(_TrackedError):
            await retry_async(
                _fail,
                max_attempts=3,
                base_delay=0.1,
                exponential=False,
                retryable_exceptions=(_TrackedError,),
            )

        assert len(timestamps) == 3
        gap1 = timestamps[1] - timestamps[0]
        gap2 = timestamps[2] - timestamps[1]

        # Both gaps should be ~0.1s (constant)
        assert gap1 >= 0.08
        assert gap2 >= 0.08
        assert abs(gap1 - gap2) < 0.05, "Linear backoff should keep delays constant"

    @pytest.mark.asyncio
    async def test_max_delay_cap(self) -> None:
        timestamps: list[float] = []

        async def _fail() -> None:
            timestamps.append(time.monotonic())
            raise _TrackedError("boom")

        with pytest.raises(_TrackedError):
            await retry_async(
                _fail,
                max_attempts=4,
                base_delay=0.1,
                max_delay=0.15,
                exponential=True,
                retryable_exceptions=(_TrackedError,),
            )

        assert len(timestamps) == 4
        # All gaps should be capped at max_delay (0.15)
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            assert gap < 0.25, f"Gap {i} exceeded max_delay cap: {gap}"


# ---------------------------------------------------------------------------
# Retryable vs non-retryable exceptions
# ---------------------------------------------------------------------------


class TestRetryableExceptions:
    """Only exceptions in retryable_exceptions should trigger retry."""

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self) -> None:
        call_count = 0

        async def _fail() -> None:
            nonlocal call_count
            call_count += 1
            raise _NonRetryableError("fatal")

        with pytest.raises(_NonRetryableError, match="fatal"):
            await retry_async(
                _fail,
                max_attempts=3,
                base_delay=0.01,
                retryable_exceptions=(_TrackedError,),
            )
        assert call_count == 1, "Non-retryable exception should not trigger retry"

    @pytest.mark.asyncio
    async def test_multiple_retryable_types(self) -> None:
        call_count = 0

        async def _alternate_fail() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _TrackedError("first")
            if call_count == 2:
                raise ValueError("second")
            return "ok"

        result = await retry_async(
            _alternate_fail,
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(_TrackedError, ValueError),
        )
        assert result == "ok"
        assert call_count == 3


# ---------------------------------------------------------------------------
# Arguments forwarding
# ---------------------------------------------------------------------------


class TestArgumentForwarding:
    """retry_async should forward *args and **kwargs correctly."""

    @pytest.mark.asyncio
    async def test_forwards_args_and_kwargs(self) -> None:
        async def _add(a: int, b: int, *, extra: int = 0) -> int:
            return a + b + extra

        result = await retry_async(_add, 2, 3, extra=10, max_attempts=1)
        assert result == 15
