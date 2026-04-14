import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.planner import planner_agent

plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries (not vague)",
        "Check that sources_to_check includes relevant sources for the topic",
        "Check that the output_format matches what the user asked for",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5.4-mini",
    threshold=0.7,
)

TEST_QUERIES = [
    "Compare naive RAG with hybrid retrieval and reranking.",
    "How does RAG reduce hallucinations, and why does it still not eliminate them?",
    "Summarize the main stages of a RAG pipeline from chunking to final answer generation.",
]


def run_planner(query: str) -> str:
    result = planner_agent.invoke({"messages": [{"role": "user", "content": query}]})
    plan = result["structured_response"]
    return plan.model_dump_json()


if __name__ == "__main__":
    test_cases = [
        LLMTestCase(input=q, actual_output=run_planner(q))
        for q in TEST_QUERIES
    ]
    evaluate(test_cases, [plan_quality])
