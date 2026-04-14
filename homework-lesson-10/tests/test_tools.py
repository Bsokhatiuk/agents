import json
import os
import sys

import pytest
from deepeval import assert_test, evaluate
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.planner import planner_agent
from agents.research import agent as research_agent
from config import settings
from supervisor import (
    critique as supervisor_critique_tool,
    plan as supervisor_plan_tool,
    research as supervisor_research_tool,
    supervisor,
    write_report as supervisor_write_report_tool,
)


# Planner should use web_search and/or knowledge_search for exploration.
# Researcher should use web_search, read_url, knowledge_search.
# Supervisor should call write_report after an APPROVE critique.
tool_metric = ToolCorrectnessMetric(threshold=0.5, model="gpt-5.4-mini")


def _tool_call_key(name: str, input_parameters) -> tuple[str, str]:
    try:
        payload = json.dumps(input_parameters or {}, sort_keys=True, ensure_ascii=False)
    except TypeError:
        payload = str(input_parameters)
    return name, payload


def _extract_tool_calls_from_message(message) -> list[ToolCall]:
    calls: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()

    raw_tool_calls = []

    message_tool_calls = getattr(message, "tool_calls", None)
    if isinstance(message_tool_calls, list):
        raw_tool_calls.extend(message_tool_calls)

    content_blocks = getattr(message, "content_blocks", None) or []
    for block in content_blocks:
        if isinstance(block, dict) and block.get("type") == "tool_call":
            raw_tool_calls.append(block)

    for call in raw_tool_calls:
        if isinstance(call, dict):
            name = call.get("name")
            input_parameters = (
                call.get("args")
                or call.get("input")
                or call.get("input_parameters")
                or {}
            )
        else:
            name = getattr(call, "name", None)
            input_parameters = (
                getattr(call, "args", None)
                or getattr(call, "input", None)
                or getattr(call, "input_parameters", None)
                or {}
            )

        if not name:
            continue

        key = _tool_call_key(name, input_parameters)
        if key in seen:
            continue
        seen.add(key)

        calls.append(ToolCall(name=name, input_parameters=input_parameters))

    return calls


def _extract_tool_calls_from_result(result: dict) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()

    for message in result.get("messages", []):
        for tool_call in _extract_tool_calls_from_message(message):
            key = _tool_call_key(tool_call.name, tool_call.input_parameters)
            if key in seen:
                continue
            seen.add(key)
            tool_calls.append(tool_call)

    return tool_calls


def _extract_tool_calls_from_supervisor_stream(chunks) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()

    for chunk in chunks:
        if chunk.get("type") != "updates":
            continue

        for step_data in chunk.get("data", {}).values():
            if not isinstance(step_data, dict):
                continue

            for message in step_data.get("messages", []):
                for tool_call in _extract_tool_calls_from_message(message):
                    key = _tool_call_key(tool_call.name, tool_call.input_parameters)
                    if key in seen:
                        continue
                    seen.add(key)
                    tool_calls.append(tool_call)

    return tool_calls


def run_planner_tools(query: str) -> list[ToolCall]:
    result = planner_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return _extract_tool_calls_from_result(result)


def run_researcher_tools(request: str, case_id: int) -> list[ToolCall]:
    result = research_agent.invoke(
        {"messages": [{"role": "user", "content": request}]},
        config={
            "configurable": {
                "thread_id": f"{settings.thread_prefix}-test-tools-researcher-{case_id}",
            }
        },
    )
    return _extract_tool_calls_from_result(result)


def run_supervisor_tools(query: str) -> list[ToolCall]:
    chunks = list(
        supervisor.stream(
            {"messages": [{"role": "user", "content": query}]},
            config={
                "configurable": {
                    "thread_id": f"{settings.thread_prefix}-test-tools-supervisor",
                }
            },
            stream_mode=["updates", "custom"],
            version="v2",
        )
    )
    return _extract_tool_calls_from_supervisor_stream(chunks)


def test_planner_tools() -> None:
    query = (
        "Create a research plan comparing naive RAG with hybrid retrieval and reranking. "
        "Explore both the web and the local knowledge base before finalizing the plan."
    )

    test_case = LLMTestCase(
        input=query,
        tools_called=run_planner_tools(query),
        expected_tools=[
            ToolCall(name="web_search"),
            ToolCall(name="knowledge_search"),
        ],
    )

    assert_test(test_case, [tool_metric])


def test_researcher_tools() -> None:
    request = """Research plan:
- Goal: Compare naive RAG with hybrid retrieval and reranking.
- Search queries:
  - naive RAG vs hybrid retrieval
  - BM25 dense retrieval reranking
- Sources to check:
  - knowledge_base
  - web
- Output format: Concise Markdown report with sources.

Follow the plan above. Use the local knowledge base, search the web, and read at least one relevant web page before writing the report.
"""

    test_case = LLMTestCase(
        input=request,
        tools_called=run_researcher_tools(request, case_id=1),
        expected_tools=[
            ToolCall(name="knowledge_search"),
            ToolCall(name="web_search"),
            ToolCall(name="read_url"),
        ],
    )

    assert_test(test_case, [tool_metric])


def test_supervisor_save(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_plan(query: str, **_: object) -> str:
        return json.dumps(
            {
                "goal": "Compare naive RAG with hybrid retrieval and reranking",
                "search_queries": [
                    "naive RAG vs hybrid retrieval",
                    "reranking cross-encoder retrieval",
                ],
                "sources_to_check": ["knowledge_base", "web"],
                "output_format": "Markdown report with sources",
            }
        )

    def fake_research(query: str, **_: object) -> str:
        return """# RAG Comparison

## Findings
- Naive RAG uses a single dense retrieval step.
- Hybrid retrieval combines dense and lexical search.
- Reranking improves final precision.

## Sources
- Local knowledge base
- Example web source
"""

    def fake_critique(query: str, **_: object) -> str:
        return json.dumps(
            {
                "verdict": "APPROVE",
                "is_fresh": True,
                "is_complete": True,
                "is_well_structured": True,
                "strengths": ["Grounded and complete"],
                "gaps": [],
                "revision_requests": [],
            }
        )

    def fake_write_report(filename: str, content: str) -> str:
        return f"Report saved successfully: output/{filename}"

    monkeypatch.setattr(supervisor_plan_tool, "func", fake_plan)
    monkeypatch.setattr(supervisor_research_tool, "func", fake_research)
    monkeypatch.setattr(supervisor_critique_tool, "func", fake_critique)
    monkeypatch.setattr(supervisor_write_report_tool, "func", fake_write_report)

    query = "Compare naive RAG with hybrid retrieval and reranking."
    test_case = LLMTestCase(
        input=query,
        tools_called=run_supervisor_tools(query),
        expected_tools=[ToolCall(name="write_report")],
    )

    assert_test(test_case, [tool_metric])


if __name__ == "__main__":
    planner_query = (
        "Create a research plan comparing naive RAG with hybrid retrieval and reranking. "
        "Explore both the web and the local knowledge base before finalizing the plan."
    )
    researcher_request = """Research plan:
- Goal: Compare naive RAG with hybrid retrieval and reranking.
- Search queries:
  - naive RAG vs hybrid retrieval
  - BM25 dense retrieval reranking
- Sources to check:
  - knowledge_base
  - web
- Output format: Concise Markdown report with sources.

Follow the plan above. Use the local knowledge base, search the web, and read at least one relevant web page before writing the report.
"""

    evaluate(
        [
            LLMTestCase(
                input=planner_query,
                tools_called=run_planner_tools(planner_query),
                expected_tools=[
                    ToolCall(name="web_search"),
                    ToolCall(name="knowledge_search"),
                ],
            ),
            LLMTestCase(
                input=researcher_request,
                tools_called=run_researcher_tools(researcher_request, case_id=1),
                expected_tools=[
                    ToolCall(name="knowledge_search"),
                    ToolCall(name="web_search"),
                    ToolCall(name="read_url"),
                ],
            ),
        ],
        [tool_metric],
    )
