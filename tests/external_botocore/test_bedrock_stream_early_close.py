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

METRIC = ("Llm/completion/Bedrock/converse_stream", 1)
INVOKE_METRIC = ("Llm/completion/Bedrock/invoke_model_with_response_stream", 1)
INVOKE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
INVOKE_BODY = json.dumps(
    {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": "What is 212 degrees Fahrenheit converted to Celsius?"}],
        "temperature": 0.7,
        "max_tokens": 100,
    }
).encode("utf-8")
MESSAGE = [{"role": "user", "content": [{"text": "What is 212 degrees Fahrenheit converted to Celsius?"}]}]


@pytest.fixture
def held_streams():
    """Keeps streams referenced until after the transaction exits."""
    streams = []
    yield streams
    streams.clear()


def _converse_stream(client):
    return client.converse_stream(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=MESSAGE,
        system=[{"text": "You are a scientist."}],
        inferenceConfig={"temperature": 0.7, "maxTokens": 100},
    )["stream"]


def _invoke_model_stream(client):
    return client.invoke_model_with_response_stream(
        body=INVOKE_BODY, modelId=INVOKE_MODEL_ID, accept="application/json", contentType="application/json"
    )["body"]


def _assistant_message(content_check, sequence=2):
    return [
        (
            {"type": "LlmChatCompletionMessage"},
            {
                "id": None,
                "request_id": None,
                "span_id": None,
                "trace_id": "trace-id",
                "completion_id": None,
                "sequence": sequence,
                "role": "assistant",
                "is_response": True,
                "content": content_check,
                "vendor": "bedrock",
                "ingest_source": "Python",
                "response.model": "anthropic.claude-3-sonnet-20240229-v1:0",
            },
        )
    ]


def _validators(name, count, events=None, metric=METRIC, timing=False):
    def decorator(func):
        func = background_task(name=name)(func)
        func = validate_transaction_metrics(
            name=name,
            scoped_metrics=[metric],
            rollup_metrics=[metric],
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


def _read_until_text(stream):
    iterator = iter(stream)
    for event in iterator:
        if "contentBlockDelta" in event:
            return
    raise AssertionError("Stream ended before any text was received")


def _read_until_invoke_text(stream):
    for event in stream:
        if b"content_block_delta" in event["chunk"]["bytes"]:
            return
    raise AssertionError("Stream ended before any text was received")


def test_bedrock_converse_stream_full_read_then_close_records_once(set_trace_info, bedrock_converse_server):
    @_validators("test_full_read_then_close", 4)
    def _test():
        set_trace_info()
        stream = _converse_stream(bedrock_converse_server)
        for _ in stream:
            pass
        stream.close()

    _test()


def test_bedrock_converse_stream_close_after_partial_read(set_trace_info, bedrock_converse_server):
    @_validators("test_close_after_partial_read", 4, _assistant_message("To"), timing=True)
    def _test():
        set_trace_info()
        stream = _converse_stream(bedrock_converse_server)
        _read_until_text(stream)
        stream.close()

    _test()


def test_bedrock_converse_stream_close_without_reading(set_trace_info, bedrock_converse_server):
    @_validators("test_close_without_reading", 3)
    def _test():
        set_trace_info()
        _converse_stream(bedrock_converse_server).close()

    _test()


def test_bedrock_converse_stream_abandoned_after_partial_read(set_trace_info, bedrock_converse_server):
    @_validators("test_abandoned_after_partial_read", 0)
    def _test():
        set_trace_info()
        stream = _converse_stream(bedrock_converse_server)
        _read_until_text(stream)
        del stream
        gc.collect()

    _test()


def test_bedrock_converse_stream_abandoned_never_read(set_trace_info, bedrock_converse_server):
    @_validators("test_abandoned_never_read", 0)
    def _test():
        set_trace_info()
        _converse_stream(bedrock_converse_server)
        gc.collect()

    _test()


def test_bedrock_invoke_model_stream_full_read_then_close_records_once(set_trace_info, bedrock_server):
    @_validators("test_invoke_full_read_then_close", 3, metric=INVOKE_METRIC)
    def _test():
        set_trace_info()
        stream = _invoke_model_stream(bedrock_server)
        for _ in stream:
            pass
        stream.close()

    _test()


def test_bedrock_invoke_model_stream_close_after_partial_read(set_trace_info, bedrock_server):
    @_validators(
        "test_invoke_close_after_partial_read",
        3,
        _assistant_message("212 degrees", sequence=1),
        INVOKE_METRIC,
        timing=True,
    )
    def _test():
        set_trace_info()
        stream = _invoke_model_stream(bedrock_server)
        _read_until_invoke_text(stream)
        stream.close()

    _test()


def test_bedrock_invoke_model_stream_close_without_reading(set_trace_info, bedrock_server):
    @_validators("test_invoke_close_without_reading", 2, metric=INVOKE_METRIC)
    def _test():
        set_trace_info()
        _invoke_model_stream(bedrock_server).close()

    _test()


def test_bedrock_invoke_model_stream_abandoned_after_partial_read(set_trace_info, bedrock_server):
    @_validators("test_invoke_abandoned_after_partial_read", 0, metric=INVOKE_METRIC)
    def _test():
        set_trace_info()
        stream = _invoke_model_stream(bedrock_server)
        _read_until_invoke_text(stream)
        del stream
        gc.collect()

    _test()


def test_bedrock_invoke_model_stream_abandoned_never_read(set_trace_info, bedrock_server):
    @_validators("test_invoke_abandoned_never_read", 0, metric=INVOKE_METRIC)
    def _test():
        set_trace_info()
        _invoke_model_stream(bedrock_server)
        gc.collect()

    _test()


@pytest.mark.parametrize("api", ["converse", "invoke"])
def test_bedrock_stream_close_swallows_instrumentation_errors(
    set_trace_info, bedrock_converse_server, bedrock_server, monkeypatch, api
):
    def explode(self, *args, **kwargs):
        self._nr_ft.__exit__(None, None, None)
        raise RuntimeError("instrumentation failure")

    monkeypatch.setattr(
        "newrelic.hooks.external_botocore.BedrockRecordEventMixin.record_events_on_stop_iteration", explode
    )

    @background_task(name="test_close_swallows_instrumentation_errors")
    def _test():
        set_trace_info()
        if api == "converse":
            stream = _converse_stream(bedrock_converse_server)
            _read_until_text(stream)
        else:
            stream = _invoke_model_stream(bedrock_server)
            _read_until_invoke_text(stream)
        stream.close()

    _test()


def test_bedrock_converse_stream_break_without_close_while_held(set_trace_info, bedrock_converse_server, held_streams):
    @_validators("test_break_without_close_while_held", 4, _assistant_message("To"), timing=True)
    def _test():
        set_trace_info()
        stream = _converse_stream(bedrock_converse_server)
        held_streams.append(stream)
        _read_until_text(stream)

    _test()


def test_bedrock_invoke_model_stream_break_without_close_while_held(set_trace_info, bedrock_server, held_streams):
    @_validators(
        "test_invoke_break_without_close_while_held",
        3,
        _assistant_message("212 degrees", sequence=1),
        INVOKE_METRIC,
        timing=True,
    )
    def _test():
        set_trace_info()
        stream = _invoke_model_stream(bedrock_server)
        held_streams.append(stream)
        _read_until_invoke_text(stream)

    _test()
