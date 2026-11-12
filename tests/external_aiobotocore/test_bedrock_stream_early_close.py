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

import gc
import json
from collections import namedtuple

import pytest
from conftest import BOTOCORE_VERSION
from testing_support.fixtures import reset_core_stats_engine
from testing_support.ml_testing_utils import set_trace_info
from testing_support.validators.validate_custom_event import validate_custom_event_count
from testing_support.validators.validate_custom_events import validate_custom_events
from testing_support.validators.validate_llm_stream_timing import validate_llm_stream_timing
from testing_support.validators.validate_transaction_count import validate_transaction_count
from testing_support.validators.validate_transaction_metrics import validate_transaction_metrics

from newrelic.api.background_task import background_task

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
PROMPT = "What is 212 degrees Fahrenheit converted to Celsius?"
INVOKE_BODY = json.dumps(
    {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7,
        "max_tokens": 100,
    }
).encode("utf-8")

Api = namedtuple("Api", "fixture metric create is_text partial_text sequence request_events partial_events")


async def _converse_stream(client):
    response = await client.converse_stream(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": PROMPT}]}],
        system=[{"text": "You are a scientist."}],
        inferenceConfig={"temperature": 0.7, "maxTokens": 100},
    )
    return response["stream"]


async def _invoke_model_stream(client):
    response = await client.invoke_model_with_response_stream(
        body=INVOKE_BODY, modelId=MODEL_ID, accept="application/json", contentType="application/json"
    )
    return response["body"]


CONVERSE = Api(
    fixture="bedrock_converse_server",
    metric=("Llm/completion/Bedrock/converse_stream", 1),
    create=_converse_stream,
    is_text=lambda event: "contentBlockDelta" in event,
    partial_text="To",
    sequence=2,
    request_events=3,
    partial_events=4,
)
INVOKE = Api(
    fixture="bedrock_server",
    metric=("Llm/completion/Bedrock/invoke_model_with_response_stream", 1),
    create=_invoke_model_stream,
    is_text=lambda event: b"content_block_delta" in event["chunk"]["bytes"],
    partial_text="212 degrees",
    sequence=1,
    request_events=2,
    partial_events=3,
)


@pytest.fixture(params=[CONVERSE, INVOKE], ids=["converse_stream", "invoke_model_with_response_stream"])
def api(request):
    return request.param


@pytest.fixture
def client(api, request):
    return request.getfixturevalue(api.fixture)


def _assistant_message(api):
    return [
        (
            {"type": "LlmChatCompletionMessage"},
            {
                "id": None,
                "request_id": None,
                "span_id": None,
                "trace_id": "trace-id",
                "completion_id": None,
                "sequence": api.sequence,
                "role": "assistant",
                "is_response": True,
                "content": api.partial_text,
                "vendor": "bedrock",
                "ingest_source": "Python",
                "response.model": MODEL_ID,
            },
        )
    ]


def _validators(name, api, count, events=None, timing=False):
    def decorator(func):
        func = background_task(name=name)(func)
        func = validate_transaction_metrics(
            name=name,
            scoped_metrics=[api.metric],
            rollup_metrics=[api.metric],
            custom_metrics=[(f"Supportability/Python/ML/Bedrock/{BOTOCORE_VERSION}", 1)],
            background_task=True,
        )(func)
        func = validate_custom_event_count(count=count)(func)
        if events:
            func = validate_custom_events(events)(func)
        if timing:
            func = validate_llm_stream_timing()(func)
        func = validate_transaction_count(1)(func)
        return reset_core_stats_engine()(func)

    return decorator


async def _read_until_text(stream, api):
    async for event in stream:
        if api.is_text(event):
            return
    raise AssertionError("Stream ended before any text was received")


def test_bedrock_async_stream_close_after_partial_read(loop, client, api, set_trace_info):
    @_validators("test_async_close_after_partial_read", api, api.partial_events, _assistant_message(api), timing=True)
    def _test():
        set_trace_info()

        async def coro():
            stream = await api.create(client)
            await _read_until_text(stream, api)
            stream.close()

        loop.run_until_complete(coro())

    _test()


def test_bedrock_async_stream_aclose_after_partial_read(loop, client, api, set_trace_info):
    @_validators("test_async_aclose_after_partial_read", api, api.partial_events, _assistant_message(api), timing=True)
    def _test():
        set_trace_info()

        async def coro():
            stream = await api.create(client)
            iterator = stream.__aiter__()
            await _read_until_text(iterator, api)
            await iterator.aclose()

        loop.run_until_complete(coro())

    _test()


def test_bedrock_async_stream_close_without_reading(loop, client, api, set_trace_info):
    @_validators("test_async_close_without_reading", api, api.request_events)
    def _test():
        set_trace_info()

        async def coro():
            (await api.create(client)).close()

        loop.run_until_complete(coro())

    _test()


def test_bedrock_async_stream_break_without_close(loop, client, api, set_trace_info):
    @_validators("test_async_break_without_close", api, api.partial_events, _assistant_message(api), timing=True)
    def _test():
        set_trace_info()

        async def coro():
            stream = await api.create(client)
            await _read_until_text(stream, api)
            return stream

        stream = loop.run_until_complete(coro())

    _test()


def test_bedrock_async_stream_abandoned_after_partial_read(loop, client, api, set_trace_info):
    @_validators("test_async_abandoned_after_partial_read", api, 0)
    def _test():
        set_trace_info()

        async def coro():
            stream = await api.create(client)
            await _read_until_text(stream, api)
            del stream
            gc.collect()

        loop.run_until_complete(coro())

    _test()


def test_bedrock_async_stream_abandoned_never_read(loop, client, api, set_trace_info):
    @_validators("test_async_abandoned_never_read", api, 0)
    def _test():
        set_trace_info()

        async def coro():
            await api.create(client)
            gc.collect()

        loop.run_until_complete(coro())

    _test()


def test_bedrock_async_stream_close_swallows_instrumentation_errors(loop, client, api, set_trace_info, monkeypatch):
    def explode(self, *args, **kwargs):
        raise RuntimeError("instrumentation failure")

    monkeypatch.setattr(
        "newrelic.hooks.external_botocore.BedrockRecordEventMixin.record_events_on_stop_iteration", explode
    )

    @background_task(name="test_async_close_swallows_instrumentation_errors")
    def _test():
        set_trace_info()

        async def coro():
            stream = await api.create(client)
            iterator = stream.__aiter__()
            await _read_until_text(iterator, api)
            stream.close()
            await iterator.aclose()

        loop.run_until_complete(coro())

    _test()
