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

import asyncio

import pytest
from strands import Agent, tool
from testing_support.fixtures import dt_enabled, reset_core_stats_engine
from testing_support.validators.validate_span_events import validate_span_events
from testing_support.validators.validate_transaction_metrics import validate_transaction_metrics

from newrelic.api.background_task import background_task

from ._mock_model_provider import MockedModelProvider
from ._test_agents import add_exclamation, single_tool_model

AGENT_METRIC = "Llm/agent/Strands/strands.agent.agent:Agent.stream_async/my_agent"
TOOL_METRIC = "Llm/tool/Strands/strands.tools.executors._executor:ToolExecutor._stream/add_exclamation"
AGENT_SPAN = {"subcomponent": '{"type": "APM-AI_AGENT", "name": "my_agent"}'}
TOOL_SPAN = {"subcomponent": '{"type": "APM-AI_TOOL", "name": "add_exclamation"}'}
PROMPT = 'Add an exclamation to the word "Hello"'


@dt_enabled
@reset_core_stats_engine()
@validate_transaction_metrics(
    "mlmodel_strands.test_stream_early_close:test_agent_stream_break_after_first_event",
    scoped_metrics=[(AGENT_METRIC, 1)],
    rollup_metrics=[(AGENT_METRIC, 1)],
    background_task=True,
)
@validate_span_events(count=1, exact_agents=AGENT_SPAN)
@background_task()
def test_agent_stream_break_after_first_event(loop, set_trace_info, single_tool_model):
    set_trace_info()
    agent = Agent(name="my_agent", model=single_tool_model, tools=[add_exclamation])

    async def _consume():
        async for _ in agent.stream_async(PROMPT):
            break

    loop.run_until_complete(_consume())


@dt_enabled
@reset_core_stats_engine()
@validate_transaction_metrics(
    "mlmodel_strands.test_stream_early_close:test_agent_stream_aclose_early",
    scoped_metrics=[(AGENT_METRIC, 1)],
    rollup_metrics=[(AGENT_METRIC, 1)],
    background_task=True,
)
@validate_span_events(count=1, exact_agents=AGENT_SPAN)
@background_task()
def test_agent_stream_aclose_early(loop, set_trace_info, single_tool_model):
    set_trace_info()
    agent = Agent(name="my_agent", model=single_tool_model, tools=[add_exclamation])

    async def _consume():
        stream = agent.stream_async(PROMPT)
        async for _ in stream:
            break
        await stream.aclose()

    loop.run_until_complete(_consume())


@dt_enabled
@reset_core_stats_engine()
@validate_transaction_metrics(
    "mlmodel_strands.test_stream_early_close:test_agent_stream_never_iterated",
    scoped_metrics=[(AGENT_METRIC, 1)],
    rollup_metrics=[(AGENT_METRIC, 1)],
    background_task=True,
)
@validate_span_events(count=1, exact_agents=AGENT_SPAN)
@background_task()
def test_agent_stream_never_iterated(set_trace_info, single_tool_model):
    set_trace_info()
    agent = Agent(name="my_agent", model=single_tool_model, tools=[add_exclamation])
    agent.stream_async(PROMPT)


@dt_enabled
@reset_core_stats_engine()
@validate_transaction_metrics(
    "mlmodel_strands.test_stream_early_close:test_agent_stream_break_after_tool_result",
    scoped_metrics=[(AGENT_METRIC, 1), (TOOL_METRIC, 1)],
    rollup_metrics=[(AGENT_METRIC, 1), (TOOL_METRIC, 1)],
    background_task=True,
)
@validate_span_events(count=1, exact_agents=AGENT_SPAN)
@validate_span_events(count=1, exact_agents=TOOL_SPAN)
@background_task()
def test_agent_stream_break_after_tool_result(loop, set_trace_info, single_tool_model):
    set_trace_info()
    agent = Agent(name="my_agent", model=single_tool_model, tools=[add_exclamation])

    async def _consume():
        async for event in agent.stream_async(PROMPT):
            if "message" in event and event["message"]["content"][0].get("toolResult"):
                break

    loop.run_until_complete(_consume())


@tool
async def slow_tool(message: str) -> str:
    """Sleeps for a long time."""
    await asyncio.sleep(30)
    return message


@pytest.fixture
def slow_tool_model():
    return MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [
                    {"text": "Calling slow_tool"},
                    {"toolUse": {"name": "slow_tool", "toolUseId": "123", "input": {"message": "Hello"}}},
                ],
            },
            {"role": "assistant", "content": [{"text": "Success!"}]},
        ]
    )


@dt_enabled
@reset_core_stats_engine()
@validate_transaction_metrics(
    "mlmodel_strands.test_stream_early_close:test_agent_cancelled_mid_tool",
    scoped_metrics=[(AGENT_METRIC, 1)],
    rollup_metrics=[(AGENT_METRIC, 1)],
    background_task=True,
)
@validate_span_events(count=1, exact_agents=AGENT_SPAN)
@background_task()
def test_agent_cancelled_mid_tool(loop, set_trace_info, slow_tool_model):
    set_trace_info()
    agent = Agent(name="my_agent", model=slow_tool_model, tools=[slow_tool])

    async def _consume():
        async def _run():
            async for _ in agent.stream_async(PROMPT):
                pass

        task = asyncio.ensure_future(_run())
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    loop.run_until_complete(_consume())
