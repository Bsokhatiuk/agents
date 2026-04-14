import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval import assert_test, evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.research import agent as research_agent
from config import settings


groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract every factual claim from 'actual output'",
        "For each claim, check if it can be directly supported by 'retrieval context'",
        "Claims not present in retrieval context count as ungrounded, even if true",
        "Score = number of grounded claims / total claims",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-5.4-mini",
    threshold=0.7,
)


TEST_CASES = [
    {
        "input": """Based only on the research notes below, write a concise Markdown report that compares naive RAG, hybrid retrieval, and reranking. Do not add facts that are not explicitly stated in the notes.

Research notes:
- Naive RAG uses a single dense retrieval step over fixed-size chunks.
- Hybrid retrieval combines dense retrieval with BM25 lexical search, which improves recall for both semantic meaning and exact terms.
- Reranking applies a cross-encoder to rescore the top retrieved chunks and improve precision.
- Hybrid retrieval plus reranking usually produces better answer quality than naive RAG, but it adds latency and system complexity.
""",
        "retrieval_context": [
            "Naive RAG uses a single dense retrieval step over fixed-size chunks.",
            "Hybrid retrieval combines dense retrieval with BM25 lexical search and improves recall for both semantic meaning and exact terms.",
            "Reranking applies a cross-encoder to rescore the top retrieved chunks and improve precision.",
            "Hybrid retrieval plus reranking usually produces better answer quality than naive RAG, but it adds latency and system complexity.",
        ],
    },
    {
        "input": """Based only on the research notes below, explain how RAG reduces hallucinations and why it still cannot eliminate them. Return a short Markdown report and avoid adding outside information.

Research notes:
- RAG reduces hallucinations by grounding the model on retrieved external documents instead of relying only on parametric memory.
- Retrieved context can still be incomplete, irrelevant, or outdated.
- If retrieval misses the needed evidence, the model may still guess or overgeneralize.
- The generation step can still introduce unsupported claims even when some context is correct.
""",
        "retrieval_context": [
            "RAG reduces hallucinations by grounding the model on retrieved external documents instead of relying only on parametric memory.",
            "Retrieved context can still be incomplete, irrelevant, or outdated.",
            "If retrieval misses the needed evidence, the model may still guess or overgeneralize.",
            "The generation step can still introduce unsupported claims even when some context is correct.",
        ],
    },
    {
        "input": """Using only the notes below, summarize the main stages of a RAG pipeline from chunking to final answer generation. Keep the answer concise and grounded strictly in the notes.

Research notes:
- Chunking splits source documents into smaller passages, sometimes with overlap.
- Embedding converts each chunk into a vector representation for indexing.
- Retrieval selects the most relevant chunks for a user query.
- Reranking reorders the retrieved candidates to keep the most useful passages near the top.
- Generation combines the user query with the selected context to produce the final answer.
""",
        "retrieval_context": [
            "Chunking splits source documents into smaller passages, sometimes with overlap.",
            "Embedding converts each chunk into a vector representation for indexing.",
            "Retrieval selects the most relevant chunks for a user query.",
            "Reranking reorders the retrieved candidates to keep the most useful passages near the top.",
            "Generation combines the user query with the selected context to produce the final answer.",
        ],
    },
]


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


def run_research(prompt: str, case_id: int) -> str:
    result = research_agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={
            "configurable": {
                "thread_id": f"{settings.thread_prefix}-test-researcher-{case_id}",
            }
        },
    )
    return _content_to_text(result["messages"][-1].content)


def build_test_cases() -> list[LLMTestCase]:
    return [
        LLMTestCase(
            input=item["input"],
            actual_output=run_research(item["input"], idx),
            retrieval_context=item["retrieval_context"],
        )
        for idx, item in enumerate(TEST_CASES, start=1)
    ]


def test_research_grounded() -> None:
    for test_case in build_test_cases():
        assert_test(test_case, [groundedness])


if __name__ == "__main__":
    evaluate(build_test_cases(), [groundedness])
