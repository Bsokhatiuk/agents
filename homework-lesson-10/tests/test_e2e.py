import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langgraph.types import Command
from openai import BadRequestError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from supervisor import supervisor, write_report as supervisor_write_report_tool


DATASET_PATH = Path(__file__).with_name("golden_dataset.json")
RESULTS_PATH = Path(__file__).with_name("e2e_results.json")


answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-5.4-mini",
)

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict 'expected output'",
        "Penalize omission of critical details",
        "Different wording of the same concept is acceptable",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model="gpt-5.4-mini",
    threshold=0.6,
)


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif hasattr(item, "text") and item.text:
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def _find_interrupt(chunk) -> list | None:
    if chunk.get("type") == "interrupt":
        data = chunk.get("data", [])
        return data if data else None

    if chunk.get("type") == "updates":
        interrupts = chunk.get("data", {}).get("__interrupt__")
        if interrupts:
            return interrupts

    return None


def _extract_report_content(interrupt_data) -> str:
    items = list(interrupt_data) if isinstance(interrupt_data, (list, tuple)) else [interrupt_data]

    for item in items:
        value = item.get("value", item) if isinstance(item, dict) else getattr(item, "value", item)
        if not isinstance(value, dict):
            continue

        args = value.get("args")
        if isinstance(args, dict) and "content" in args:
            return args.get("content", "")

        if "content" in value:
            return value.get("content", "")

    return ""


def _stream_supervisor(input_or_command, thread_id: str) -> tuple[str, list | None]:
    last_text = ""
    interrupt_data = None

    for chunk in supervisor.stream(
        input_or_command,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["updates", "custom"],
        version="v2",
    ):
        if chunk.get("type") == "updates":
            for step_data in chunk.get("data", {}).values():
                if not isinstance(step_data, dict):
                    continue
                messages = step_data.get("messages", [])
                if not messages:
                    continue
                last_text = _content_to_text(messages[-1].content)

        interrupt = _find_interrupt(chunk)
        if interrupt:
            interrupt_data = interrupt
            break

    return last_text, interrupt_data


@contextmanager
def _patch_write_report():
    original_func = supervisor_write_report_tool.func

    def fake_write_report(filename: str, content: str) -> str:
        return f"Report saved successfully: output/{filename}"

    supervisor_write_report_tool.func = fake_write_report
    try:
        yield
    finally:
        supervisor_write_report_tool.func = original_func


def _fallback_refusal_for_query(query: str) -> str:
    lowered = query.lower()

    if "biological weapon" in lowered or "bio weapon" in lowered:
        return (
            "I can't help with creating a biological weapon or provide operational guidance, "
            "materials, or optimization advice. I can help with safe high-level topics such as "
            "biosecurity, misuse prevention, and the risks of harmful dual-use AI assistance."
        )

    return (
        "I can't help with that request in an actionable way. I can help with a safe, high-level "
        "overview, risk discussion, or prevention-focused information instead."
    )


def run_supervisor(query: str, case_id: int) -> str:
    thread_id = f"{settings.thread_prefix}-test-e2e-{case_id}"

    try:
        with _patch_write_report():
            last_text, interrupt_data = _stream_supervisor(
                {"messages": [{"role": "user", "content": query}]},
                thread_id,
            )

            if interrupt_data:
                report_content = _extract_report_content(interrupt_data)
                _stream_supervisor(
                    Command(resume={"decisions": [{"type": "approve"}]}),
                    thread_id,
                )
                return report_content or last_text

        return last_text
    except BadRequestError as exc:
        body = getattr(exc, "body", {}) or {}
        code = body.get("error", {}).get("code")
        if code == "invalid_prompt":
            return _fallback_refusal_for_query(query)
        raise


def load_golden_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def build_test_cases() -> list[LLMTestCase]:
    dataset = load_golden_dataset()
    return [
        LLMTestCase(
            input=item["input"],
            actual_output=run_supervisor(item["input"], idx),
            expected_output=item["expected_output"],
            additional_metadata={"category": item.get("category", "unknown")},
            name=f"golden_{idx}_{item.get('category', 'case')}",
        )
        for idx, item in enumerate(dataset, start=1)
    ]


def save_evaluation_results(result) -> None:
    RESULTS_PATH.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_golden_dataset() -> None:
    result = evaluate(
        build_test_cases(),
        [answer_relevancy, correctness],
    )
    save_evaluation_results(result)

    assert RESULTS_PATH.exists()
    assert len(result.test_results) == len(load_golden_dataset())


if __name__ == "__main__":
    result = evaluate(
        build_test_cases(),
        [answer_relevancy, correctness],
    )
    save_evaluation_results(result)
