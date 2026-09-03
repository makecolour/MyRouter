"""Retries must not multiply.

    python -m unittest discover -s tests -t .

Three retry mechanisms sit on top of each other: qwen's agent loop, MyRouter's
own transient retry, and gemini_webapi's @running(retry=5) ladder. None knows
about the others, so a single failing turn re-sent a >100 KB prompt up to twelve
times and one answer took 17 minutes to arrive.

These tests pin the two rules that stop it, plus the assumption about the
vendored library that the second rule depends on.
"""

import asyncio
import inspect
import unittest

from gemini_webapi.exceptions import APIError, GeminiError
from gemini_webapi.utils.decorators import running
from gemini_webapi.exceptions import (
    ModelInvalidError,
    TemporarilyBlockedError,
    TimeoutError as GeminiTimeoutError,
    UsageLimitExceededError,
)

from app.pool import _PooledGeminiClient
from app.routes.chat import _is_transient_upstream

# Verbatim from gemini_webapi/client.py, with the line that raises each.
SILENTLY_ABORTED = APIError(  # :1926
    "read_chat polling timed out waiting for the model to finish. The original "
    "request may have been silently aborted by Google."
)
UNKNOWN_CODE = APIError(  # :1609
    "Failed to generate contents (stream). Unknown API error code: 1096. "
    "This might be a temporary Google service issue."
)
CONNECTION_LOST = GeminiError(  # :1921
    "The connection to Gemini was lost while generating the response, and "
    "recovery timed out. Please try sending your prompt again."
)


class TransientSplit(unittest.TestCase):
    """Retry here only what the library's ladder did not already retry."""

    def test_api_errors_are_not_retried_again(self):
        """@running(retry=5) already sent these six times."""
        for exc in (SILENTLY_ABORTED, UNKNOWN_CODE):
            with self.subTest(exc=str(exc)[:40]):
                self.assertFalse(_is_transient_upstream(exc))

    def test_a_lost_connection_is_retried(self):
        """A GeminiError: the ladder catches APIError only, so this is its only retry."""
        self.assertTrue(_is_transient_upstream(CONNECTION_LOST))

    def test_account_faults_are_never_retried(self):
        for exc in (
            UsageLimitExceededError("quota exhausted"),
            ModelInvalidError("no such model"),
            TemporarilyBlockedError("429"),
            GeminiTimeoutError("timed out"),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.assertFalse(_is_transient_upstream(exc))

    def test_unrelated_exceptions_are_not_retried(self):
        self.assertFalse(_is_transient_upstream(ValueError("nope")))

    def test_the_two_classes_really_are_siblings(self):
        """The whole split rests on this. If upstream ever makes APIError a
        GeminiError subclass, the isinstance guard silently starts retrying
        everything again."""
        self.assertFalse(issubclass(APIError, GeminiError))
        self.assertFalse(issubclass(GeminiError, APIError))


class LadderCap(unittest.TestCase):
    """`current_retry` is how MyRouter caps the vendored ladder.

    Driven through the REAL @running decorator rather than by introspection:
    functools.wraps makes inspect.signature report the wrapped function, which
    hides the kwarg entirely. If an upgrade ever drops it, the cap stops working
    and every failing turn silently goes back to six sends of the whole prompt.
    """

    def _run(self, **kwargs):
        """Drive one decorated asyncgen; return (attempts, kwargs it saw, delays)."""

        class FakeClient:
            _running = True

            def __init__(self):
                self.attempts = 0
                self.seen = None
                self.closed = 0

            async def close(self):
                self.closed += 1

        @running(retry=5)
        async def _generate(client, prompt, **inner):
            client.attempts += 1
            client.seen = inner
            raise APIError("boom")
            yield  # noqa: unreachable - makes this an async generator

        client = FakeClient()
        delays = []
        real_sleep = asyncio.sleep

        async def no_sleep(seconds):
            delays.append(seconds)
            await real_sleep(0)

        async def drive():
            with self.assertRaises(APIError):
                async for _ in _generate(client, "hi", **kwargs):
                    pass

        asyncio.sleep = no_sleep
        try:
            asyncio.run(drive())
        finally:
            asyncio.sleep = real_sleep
        return client, delays

    def test_uncapped_is_six_attempts(self):
        """What we were paying: six sends of the prompt, 75s of back-off."""
        client, delays = self._run()
        self.assertEqual(client.attempts, 6)
        self.assertEqual(delays, [5, 10, 15, 20, 25])
        self.assertEqual(sum(delays), 75)

    def test_current_retry_caps_the_ladder(self):
        client, delays = self._run(current_retry=1)
        self.assertEqual(client.attempts, 2)

    def test_the_kwarg_never_reaches_the_request(self):
        """It must be popped by the wrapper - kwargs are documented as going
        through to curl_cffi, which would reject it."""
        client, _ = self._run(current_retry=1)
        self.assertEqual(client.seen, {})

    def test_the_backoff_is_computed_from_the_decorator_not_the_cap(self):
        """The quirk the comment in _ask_gemini records: the delay is
        (retry - current_retry + 1) * 5, so capping buys FEWER retries spaced
        WIDER - one 25s pause, not one 5s pause."""
        _, delays = self._run(current_retry=1)
        self.assertEqual(delays, [25])

    def test_exhaustion_closes_the_client(self):
        """Which is what makes the next attempt re-init - see CheapReinit."""
        client, _ = self._run(current_retry=1)
        self.assertEqual(client.closed, 1)

    def test_ask_gemini_caps_both_paths(self):
        """A session turn and a stateless turn must both be capped - the
        conversation path is the one BotGymRam actually uses."""
        from app.routes.chat import _ask_gemini

        source = inspect.getsource(_ask_gemini)
        self.assertEqual(source.count("**capped"), 2)
        self.assertIn("gemini_generate_retries", source)


class CheapReinit(unittest.TestCase):
    """A dropped socket must not cost eight RPCs to replace.

    Every RPC in _init_rpc is stubbed to record itself, so this drives the real
    dispatch (including super()._init_rpc) with nothing mocked out.
    """

    RPCS = (
        "_fetch_user_status",
        "_fetch_preferences",
        "_sync_activity",
        "_fetch_recent_chats",
        "_fetch_usage_info",
        "_fetch_quota",
        "_fetch_extra_quota",
        "_fetch_abuse_status",
    )

    def setUp(self):
        calls = []

        class Spy(_PooledGeminiClient):
            def __init__(self):  # no network, no cookies
                pass

        for name in self.RPCS:
            def record(self, _name=name):
                calls.append(_name)

            setattr(Spy, name, _as_async(record))

        self.calls = calls
        self.client = Spy()

    def test_the_first_init_runs_the_full_set(self):
        """Startup logging - quota, abuse status, model list - is unchanged."""
        asyncio.run(self.client._init_rpc())
        self.assertEqual(self.calls, list(self.RPCS))

    def test_later_inits_run_only_the_two_that_matter(self):
        asyncio.run(self.client._init_rpc())
        self.calls.clear()
        asyncio.run(self.client._init_rpc())
        self.assertEqual(self.calls, ["_fetch_user_status", "_sync_activity"])

    def test_account_status_is_never_skipped(self):
        """_fetch_user_status sets account_status, and client.py:1893 gates
        stream recovery on it - skipping it would silently disable recovery."""
        asyncio.run(self.client._init_rpc())
        self.calls.clear()
        for _ in range(3):
            asyncio.run(self.client._init_rpc())
        self.assertEqual(self.calls.count("_fetch_user_status"), 3)


def _as_async(fn):
    async def wrapper(self):
        return fn(self)

    return wrapper


if __name__ == "__main__":
    unittest.main()
