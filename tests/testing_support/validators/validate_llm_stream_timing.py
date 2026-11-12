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

from newrelic.common.object_wrapper import function_wrapper, transient_function_wrapper
from testing_support.fixtures import catch_background_exceptions


def validate_llm_stream_timing():
    """Assert LLM summary events carry timestamp and time_to_first_token and every request message carries a timestamp."""

    @function_wrapper
    def _validate_wrapper(wrapped, instance, args, kwargs):
        recorded_events = []

        @transient_function_wrapper("newrelic.core.stats_engine", "StatsEngine.record_transaction")
        @catch_background_exceptions
        def _capture(wrapped, instance, args, kwargs):
            result = wrapped(*args, **kwargs)
            recorded_events[:] = list(instance._custom_events)
            return result

        val = _capture(wrapped)(*args, **kwargs)

        summaries = [e for e in recorded_events if e[0]["type"] == "LlmChatCompletionSummary"]
        messages = [e for e in recorded_events if e[0]["type"] == "LlmChatCompletionMessage"]
        assert len(summaries) == 1, f"Expected 1 summary, got {len(summaries)}"
        assert summaries[0][1].get("timestamp") is not None, f"summary missing timestamp: {summaries[0][1]}"
        assert summaries[0][1].get("time_to_first_token") is not None, (
            f"summary missing time_to_first_token: {summaries[0][1]}"
        )
        for message in messages:
            if message[1].get("is_response"):
                continue
            assert message[1].get("timestamp") is not None, f"message missing timestamp: {message[1]}"
        return val

    return _validate_wrapper
