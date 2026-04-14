import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.critic import critic_agent

critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Check that the critique identifies specific issues, not vague complaints",
        "Check that revision_requests are actionable (researcher can act on them)",
        "If verdict is APPROVE, gaps list should be empty or contain only minor items",
        "If verdict is REVISE, there must be at least one revision_request",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-5.4-mini",
    threshold=0.7,
)

# Each item is a (user_request, research_findings) pair.
# The critic receives both so it can evaluate completeness relative to the original request.
TEST_INPUTS = [
    (
        "Compare naive RAG with hybrid retrieval and reranking.",
        """# RAG Approaches Comparison

## Executive Summary
Naive RAG relies on a single dense retrieval step over fixed-size chunks. Hybrid retrieval
combines semantic (dense) and lexical (BM25) search, improving recall for both meaning and
exact terms. Reranking adds a cross-encoder pass to re-score the top candidates.

## Findings
- Naive RAG: simple, fast, but misses exact-term queries and rare terminology.
- Hybrid retrieval: better recall, handles technical queries where semantic search fails.
- Reranking: improves precision by prioritising the most relevant passages before generation.

## Conclusion
Hybrid retrieval + reranking consistently outperforms naive RAG for technical domains.

## Sources
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
""",
    ),
    (
        "How does RAG reduce hallucinations, and why does it still not eliminate them?",
        """# RAG and Hallucinations

## Summary
RAG helps by providing retrieved context to the model.

## Findings
RAG reduces hallucinations because it gives the model external documents.
Sometimes the model still makes mistakes.

## Conclusion
RAG is good but not perfect.
""",
    ),
    (
        "Summarize the main stages of a RAG pipeline from chunking to final answer generation.",
        """# RAG Pipeline Stages

## Executive Summary
A RAG pipeline transforms raw documents into grounded answers through several stages.

## Findings
1. **Chunking** — documents are split into fixed or semantic chunks with optional overlap.
2. **Embedding** — chunks are encoded into dense vectors and stored in a vector index.
3. **Retrieval** — at query time, the top-k most similar chunks are retrieved (dense, sparse, or hybrid).
4. **Reranking** — a cross-encoder scores the candidates and keeps only the most relevant.
5. **Generation** — the LLM receives the query + retrieved context and produces a grounded answer.

## Risks / Trade-offs
- Chunk size affects recall vs. context quality trade-off.
- Reranking adds latency but improves precision.

## Sources
- Lewis et al. (2020), RAG paper
- LangChain documentation on retrievers
""",
    ),
]


def run_critic(user_request: str, research_findings: str) -> str:
    input_text = f"User request: {user_request}\n\nResearch findings:\n{research_findings}"
    result = critic_agent.invoke({"messages": [{"role": "user", "content": input_text}]})
    critique = result["structured_response"]
    return critique.model_dump_json()


if __name__ == "__main__":
    test_cases = [
        LLMTestCase(
            input=f"User request: {req}\n\nResearch findings:\n{findings}",
            actual_output=run_critic(req, findings),
        )
        for req, findings in TEST_INPUTS
    ]
    evaluate(test_cases, [critique_quality])
