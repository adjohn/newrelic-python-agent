# Copyright 2010 New Relic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for OpenAI streams that stop being read before they are exhausted.

The LLM function trace is entered when ``create(stream=True)`` is called and normally exited when the
stream is exhausted. These tests cover callers that stop early (``break``, ``close()``, ``with``/``async with``),
which must still exit the trace once and record the events built from what was received. If the trace stays open,
the transaction can't be completed and is dropped.

These tests replay the same cassette interactions as test_responses_stream.py and test_chat_completion_stream.py.
"""

import gc

import openai
import pytest
from conftest import OPENAI_VERSION
from test_chat_completion_stream import _test_openai_chat_completion_messages, chat_completion_recorded_events
from test_responses_stream import INSTRUCTIONS, MODEL, PROMPT, responses_stream_recorded_events
from testing_support.fixtures import reset_core_stats_engine, validate_attributes
from testing_support.ml_testing_utils import events_sans_llm_metadata, set_trace_info
from testing_support.validators.validate_custom_event import validate_custom_event_count
from testing_support.validators.validate_custom_events import validate_custom_events
from testing_support.validators.validate_transaction_count import validate_transaction_count
from testing_support.validators.validate_transaction_metrics import validate_transaction_metrics

from newrelic.api.background_task import background_task

pytestmark = pytest.mark.skipif(OPENAI_VERSION < (1, 0), reason="Stream objects only exist in OpenAI v1+")

LLM_TRACE_METRICS = [("Llm/completion/OpenAI/create", 1)]


def _responses_partial_events(events):
    """Events recorded when the stream is closed after the first text delta but before the terminal event."""
    events = events_sans_llm_metadata(events)
    for _, attrs in events:
        # The model, usage and finish reason only arrive in the terminal event.
        attrs.pop("response.model", None)
        attrs.pop("response.choices.finish_reason", None)
        attrs.pop("token_count", None)
        for key in [key for key in attrs if key.startswith("response.usage.")]:
            del attrs[key]
    return events


def _chat_partial_events(events):
    """Events recorded when the stream is closed after the first content chunk but before the final chunk."""
    events = events_sans_llm_metadata(events)
    # The finish reason arrives in the last chunk.
    events[0][1].pop("response.choices.finish_reason")
    return events


def _full_events(events):
    return events_sans_llm_metadata(events)


responses_full_events = _full_events(responses_stream_recorded_events)
responses_partial_events = _responses_partial_events(responses_stream_recorded_events)
chat_full_events = _full_events(chat_completion_recorded_events)
chat_partial_events = _chat_partial_events(chat_completion_recorded_events)


def _responses_create(client):
    return client.responses.create(
        model=MODEL, instructions=INSTRUCTIONS, input=PROMPT, temperature=0.7, max_output_tokens=500, stream=True
    )


def _chat_create(client):
    return client.chat.completions.create(
        model="gpt-5.1",
        messages=_test_openai_chat_completion_messages,
        temperature=0.7,
        max_completion_tokens=500,
        stream=True,
    )


def _responses_is_text(event):
    return event.type == "response.output_text.delta"


def _responses_is_terminal(event):
    return event.type == "response.completed"


def _chat_has_content(chunk):
    return bool(chunk.choices and chunk.choices[0].delta.content)


def _read_until(stream, predicate):
    for event in stream:
        if predicate(event):
            return
    raise AssertionError("Stream ended before the expected event")


async def _aread_until(stream, predicate):
    async for event in stream:
        if predicate(event):
            return
    raise AssertionError("Stream ended before the expected event")


def _txn_validators(name, events, count):
    """Combine validators shared by every test: one transaction, one LLM trace, and the expected events."""

    def decorator(func):
        func = background_task()(func)
        func = validate_attributes("agent", ["llm"])(func)
        func = validate_transaction_metrics(
            name=f"test_stream_early_close:{name}",
            scoped_metrics=LLM_TRACE_METRICS,
            rollup_metrics=LLM_TRACE_METRICS,
            custom_metrics=[(f"Supportability/Python/ML/OpenAI/{openai.__version__}", 1)],
            background_task=True,
        )(func)
        func = validate_custom_event_count(count=count)(func)
        if events is not None:
            func = validate_custom_events(events)(func)
        func = validate_transaction_count(1)(func)
        return reset_core_stats_engine()(func)

    return decorator


# Responses API


@_txn_validators("test_responses_sync_full_read_records_once", responses_full_events, 4)
def test_responses_sync_full_read_records_once(set_trace_info, sync_openai_client):
    set_trace_info()
    stream = _responses_create(sync_openai_client)
    for _ in stream:
        pass
    # Closing a stream that already finished must not record anything again.
    stream.close()


@_txn_validators("test_responses_sync_close_after_partial_read", responses_partial_events, 4)
def test_responses_sync_close_after_partial_read(set_trace_info, sync_openai_client):
    set_trace_info()
    stream = _responses_create(sync_openai_client)
    _read_until(stream, _responses_is_text)
    stream.close()


@_txn_validators("test_responses_sync_context_manager_partial_read", responses_partial_events, 4)
def test_responses_sync_context_manager_partial_read(set_trace_info, sync_openai_client):
    set_trace_info()
    with _responses_create(sync_openai_client) as stream:
        _read_until(stream, _responses_is_text)


@_txn_validators("test_responses_sync_close_without_reading", None, 0)
def test_responses_sync_close_without_reading(set_trace_info, sync_openai_client):
    set_trace_info()
    _responses_create(sync_openai_client).close()


@_txn_validators("test_responses_sync_break_after_completed_with_close", responses_full_events, 4)
def test_responses_sync_break_after_completed_with_close(set_trace_info, sync_openai_client):
    set_trace_info()
    with _responses_create(sync_openai_client) as stream:
        for event in stream:
            if _responses_is_terminal(event):
                break


@_txn_validators("test_responses_sync_break_after_completed_without_close", responses_full_events, 4)
def test_responses_sync_break_after_completed_without_close(set_trace_info, sync_openai_client):
    # The terminal event carries usage and the full output, so events are recorded without a close().
    set_trace_info()
    _read_until(_responses_create(sync_openai_client), _responses_is_terminal)


@_txn_validators("test_responses_async_full_read_records_once", responses_full_events, 4)
def test_responses_async_full_read_records_once(set_trace_info, async_openai_client, loop):
    set_trace_info()

    async def _run():
        stream = await _responses_create(async_openai_client)
        async for _ in stream:
            pass
        await stream.close()

    loop.run_until_complete(_run())


@_txn_validators("test_responses_async_aclose_after_partial_read", responses_partial_events, 4)
def test_responses_async_aclose_after_partial_read(set_trace_info, async_openai_client, loop):
    set_trace_info()

    async def _run():
        stream = await _responses_create(async_openai_client)
        await _aread_until(stream, _responses_is_text)
        await stream.close()

    loop.run_until_complete(_run())


@_txn_validators("test_responses_async_context_manager_partial_read", responses_partial_events, 4)
def test_responses_async_context_manager_partial_read(set_trace_info, async_openai_client, loop):
    set_trace_info()

    async def _run():
        async with await _responses_create(async_openai_client) as stream:
            await _aread_until(stream, _responses_is_text)

    loop.run_until_complete(_run())


@_txn_validators("test_responses_async_break_after_completed_with_context_manager", responses_full_events, 4)
def test_responses_async_break_after_completed_with_context_manager(set_trace_info, async_openai_client, loop):
    # The Strands pattern: break on response.completed, then close via "async with".
    set_trace_info()

    async def _run():
        async with await _responses_create(async_openai_client) as stream:
            async for event in stream:
                if _responses_is_terminal(event):
                    break

    loop.run_until_complete(_run())


@_txn_validators("test_responses_async_break_after_completed_without_close", responses_full_events, 4)
def test_responses_async_break_after_completed_without_close(set_trace_info, async_openai_client, loop):
    set_trace_info()

    async def _run():
        await _aread_until(await _responses_create(async_openai_client), _responses_is_terminal)

    loop.run_until_complete(_run())


# Chat Completions API


@_txn_validators("test_chat_sync_full_read_records_once", chat_full_events, 4)
def test_chat_sync_full_read_records_once(set_trace_info, sync_openai_client):
    set_trace_info()
    stream = _chat_create(sync_openai_client)
    for _ in stream:
        pass
    stream.close()


@_txn_validators("test_chat_sync_close_after_partial_read", chat_partial_events, 4)
def test_chat_sync_close_after_partial_read(set_trace_info, sync_openai_client):
    set_trace_info()
    stream = _chat_create(sync_openai_client)
    _read_until(stream, _chat_has_content)
    stream.close()


@_txn_validators("test_chat_sync_context_manager_partial_read", chat_partial_events, 4)
def test_chat_sync_context_manager_partial_read(set_trace_info, sync_openai_client):
    set_trace_info()
    with _chat_create(sync_openai_client) as stream:
        _read_until(stream, _chat_has_content)


@_txn_validators("test_chat_sync_close_without_reading", None, 0)
def test_chat_sync_close_without_reading(set_trace_info, sync_openai_client):
    set_trace_info()
    _chat_create(sync_openai_client).close()


@_txn_validators("test_chat_async_full_read_records_once", chat_full_events, 4)
def test_chat_async_full_read_records_once(set_trace_info, async_openai_client, loop):
    set_trace_info()

    async def _run():
        stream = await _chat_create(async_openai_client)
        async for _ in stream:
            pass
        await stream.close()

    loop.run_until_complete(_run())


@_txn_validators("test_chat_async_aclose_after_partial_read", chat_partial_events, 4)
def test_chat_async_aclose_after_partial_read(set_trace_info, async_openai_client, loop):
    set_trace_info()

    async def _run():
        stream = await _chat_create(async_openai_client)
        await _aread_until(stream, _chat_has_content)
        await stream.close()

    loop.run_until_complete(_run())


@_txn_validators("test_chat_async_context_manager_partial_read", chat_partial_events, 4)
def test_chat_async_context_manager_partial_read(set_trace_info, async_openai_client, loop):
    set_trace_info()

    async def _run():
        async with await _chat_create(async_openai_client) as stream:
            await _aread_until(stream, _chat_has_content)

    loop.run_until_complete(_run())


@_txn_validators("test_chat_sync_abandoned_after_partial_read", None, 0)
def test_chat_sync_abandoned_after_partial_read(set_trace_info, sync_openai_client):
    set_trace_info()
    stream = _chat_create(sync_openai_client)
    _read_until(stream, _chat_has_content)
    del stream
    gc.collect()


@_txn_validators("test_responses_sync_abandoned_after_partial_read", None, 0)
def test_responses_sync_abandoned_after_partial_read(set_trace_info, sync_openai_client):
    set_trace_info()
    stream = _responses_create(sync_openai_client)
    _read_until(stream, _responses_is_text)
    del stream
    gc.collect()
