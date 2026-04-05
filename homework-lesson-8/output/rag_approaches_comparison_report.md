# Comparing RAG Approaches: Naive, Sentence-Window, and Parent-Child

## Executive Summary

This report compares three common RAG retrieval patterns:

1. **Naive chunk retrieval**: retrieve fixed chunks directly.
2. **Sentence-window retrieval**: retrieve sentence-level units, then expand each hit with surrounding sentences.
3. **Parent-child retrieval**: index smaller child chunks, but return a larger parent chunk or document at retrieval time.

### Bottom line
- **Naive retrieval** is the simplest baseline and remains useful when documents are already well-structured, queries are broad, and latency or implementation complexity must stay low.
- **Sentence-window retrieval** is a documented framework pattern in LlamaIndex, intended to improve local matching while restoring nearby context before synthesis.
- **Parent-child retrieval** is a documented framework pattern in LangChain and related ecosystem docs, intended to balance fine-grained matching with broader returned context.
- The strongest evidence available supports that **chunk granularity matters**, but does **not** establish a universal winner among these three methods across tasks.
- In practice, **reranking, hybrid retrieval, and better chunk formation** may influence quality as much as or more than the choice among these three patterns.

---

## Findings

## 1) Definitions and documented mechanics

### Naive chunk retrieval
Documents are split into chunks once, embeddings are created for those chunks, and retrieval returns those chunks directly.

**Established by sources**
- This is the standard baseline implied by common RAG pipelines.
- Anthropic’s contextual retrieval article describes the standard chunk → embed → retrieve pattern and notes that traditional chunking can remove context when chunks are too isolated.

**Core trade-off**
- The same unit must serve both retrieval precision and generation context.
- Smaller chunks can match more precisely; larger chunks can provide more context.

---

### Sentence-window retrieval
Index at sentence granularity, retrieve the most relevant sentence nodes, then expand each retrieved sentence with neighboring sentences before passing context to the LLM.

**Established by sources**
- LlamaIndex’s official sentence-window materials indicate a pattern where:
  - documents are split into sentence-level nodes,
  - surrounding sentences are stored in metadata,
  - retrieved nodes are replaced or expanded with the stored window before synthesis.

**Source confidence note**
- In this report, LlamaIndex support is **official but snippet-supported** because the full pages were not directly readable in the tool session.
- Therefore, the broad pattern is reasonably supported, but detailed implementation specifics should be treated as tentative unless verified separately.

---

### Parent-child retrieval
Split documents into smaller **child chunks** for embedding and search, but return a larger **parent chunk** or **parent document** after matching.

**Established by sources**
- LangChain’s `ParentDocumentRetriever` documentation explains that:
  - documents are split into smaller chunks for embedding and vector search,
  - retrieval returns the larger parent document rather than the small child chunk,
  - this is meant to balance finer-grained retrieval with fuller context.
- MongoDB’s LangChain integration docs describe the same pattern and rationale.

---

## 2) Direct comparison

## Side-by-side comparison table

| Dimension | Naive | Sentence-window | Parent-child |
|---|---|---|---|
| **Indexed unit** | fixed chunk | sentence or very small unit | small child chunk |
| **Returned unit** | same chunk | retrieved sentence plus nearby sentences | linked parent chunk/document |
| **Retrieval granularity** | medium, depends on chunk size | very fine-grained | fine-grained at retrieval stage |
| **Returned context size** | medium, fixed by chunking | local context window | larger section/document context |
| **Typical precision behavior** | depends heavily on chunk size | often strong for local evidence | often strong due to small child chunks |
| **Typical recall/context behavior** | may miss split context | good for local context, weaker for distant context | better when broader section context is needed |
| **Index/storage overhead** | lowest of the three | higher due to many sentence nodes | higher due to child embeddings + parent linkage |
| **Retrieval/orchestration complexity** | low | medium | medium to high |
| **Prompt-token cost** | low to moderate | moderate | moderate to high |
| **Latency** | lowest | moderate | moderate to higher |
| **Failure mode** | context fragmentation or diluted chunks | right sentence, insufficient broader context | right child, but noisy/large parent |
| **Best fit** | simple baseline, broad questions, cost-sensitive systems | local evidence lookup in prose | section-level reasoning over long documents |

### Comparison summary
- **Naive** uses one retrieval unit and one returned unit, so its main weakness is the precision-vs-context trade-off.
- **Sentence-window** separates retrieval precision from generation context, but only within a **local** neighborhood.
- **Parent-child** also separates retrieval precision from generation context, but at a **broader structural** level.

---

## 3) Strengths, weaknesses, and practical trade-offs

### Naive retrieval
**Strengths**
- Easiest to build and operate.
- Lowest orchestration complexity.
- Good baseline for A/B testing.

**Weaknesses**
- Sensitive to chunk size and overlap.
- Vulnerable to boundary errors and fragmented context.
- Can underperform when answers depend on nearby but separate chunks.

**Best fit**
- Small or clean corpora.
- Broad paragraph-level questions.
- Teams wanting a baseline before investing in more retrieval structure.

---

### Sentence-window retrieval
**Strengths**
- Fine-grained matching can recover exact local evidence.
- Restores adjacent context that a single sentence alone lacks.
- Natural fit for prose-heavy documents.

**Weaknesses**
- Depends on sentence segmentation quality.
- Can struggle when required context is farther away than the window.
- Less natural for tables, code, lists, or irregular formatting.

**Best fit**
- Policies, manuals, articles, and other expository prose.
- Questions answerable from a narrow local span.
- Cases where naive chunking is too coarse.

---

### Parent-child retrieval
**Strengths**
- Searches small units but returns larger context.
- Better suited than sentence-window when useful context spans multiple paragraphs.
- Strong fit for long, structured documents.

**Weaknesses**
- More tuning and orchestration.
- Parent chunks can become noisy or expensive.
- Requires parent/child linkage and deduplication logic.

**Best fit**
- Long reports, manuals, policy docs, and PDFs with meaningful section structure.
- Questions needing broader passage context.

---

## 4) Evidence: established vs. inferred

## Documented behavior / pattern
These points are supported by accessible sources:
- Traditional RAG commonly chunks documents and retrieves chunks directly.  
  **Source:** Anthropic.
- Traditional chunking can remove context.  
  **Source:** Anthropic.
- Parent-child retrieval is supported by LangChain and documented as small-chunk retrieval plus larger-parent return.  
  **Sources:** LangChain, MongoDB.
- Sentence-window retrieval exists as a documented LlamaIndex pattern involving sentence nodes and metadata windows.  
  **Source status:** official but snippet-supported only.
- Recent 2024–2025 research supports that retrieval granularity matters.  
  **Sources:** 2024 RAG evaluation survey; 2025 Mix-of-Granularity; 2025 Reconstructing Context.

## Practitioner guidance / expected trade-offs
These are reasonable design expectations, but not universal empirical facts:
- Smaller retrieval units often improve matching precision.
- Larger returned contexts can improve answerability when surrounding detail matters.
- Sentence-window tends to help when answers are locally concentrated.
- Parent-child tends to help when answers require broader section context.
- Naive retrieval can be sufficient when chunking already matches the corpus well.

---

## 5) Research freshness and empirical caveats

### What newer sources add
- A 2024 RAG evaluation survey reinforces that benchmark coverage is incomplete and evaluation remains fragmented.
- A 2025 COLING paper argues that one fixed chunk granularity is often suboptimal and that varying granularity can improve performance.
- A 2025 preprint on advanced chunking/context-preserving strategies reports meaningful trade-offs between coherence and efficiency.

### Important caveat
These sources strengthen the argument that **granularity and context reconstruction matter**, but they do **not** provide a definitive three-way benchmark proving naive vs. sentence-window vs. parent-child performance across all corpora.

---

## 6) Recommendation matrix

| Scenario | Recommended approach | Why |
|---|---|---|
| Fast MVP | Naive | simplest baseline |
| Tight latency/cost budget | Naive | lowest overhead |
| Fact lookup in prose | Sentence-window | precise local retrieval plus nearby context |
| Clause/policy/manual lookup | Sentence-window or Parent-child | depends on whether local or section context dominates |
| Long reports and manuals | Parent-child | broader context after precise match |
| Multi-paragraph explanation | Parent-child | better section-level grounding |
| Small clean corpus | Naive | often sufficient |
| Noisy retrieval candidates | Any + reranking | reranking may matter more than chunking choice |
| Exact terms, IDs, codes | Any + hybrid retrieval | lexical retrieval complements embeddings |

### Practical decision rule
Choose by dominant failure mode:
- **Naive** if simplicity matters most.
- **Sentence-window** if the answer is local but isolated sentences are too thin.
- **Parent-child** if small hits need broader surrounding section context.

---

## 7) How to evaluate on your corpus

A practical comparison should test all three methods on the same documents and queries.

### Recommended metrics
**Retrieval metrics**
- Recall@k
- Precision@k
- nDCG or MRR
- context sufficiency / context precision

**Generation metrics**
- answer correctness
- faithfulness / groundedness
- citation quality
- completeness

**Operational metrics**
- indexing time
- storage size
- retrieval latency
- prompt-token count
- end-to-end cost

### Suggested A/B test plan
1. Build the same corpus three ways: naive, sentence-window, parent-child.
2. Use the same embedding model, vector store, and LLM.
3. Create a query set with at least three categories:
   - local fact lookup,
   - section-level explanation,
   - multi-paragraph synthesis.
4. Measure retrieval and answer metrics side by side.
5. Inspect failure modes manually:
   - missing local context,
   - fragmented context,
   - noisy large-context returns,
   - exact-match misses.
6. Repeat with and without:
   - reranking,
   - hybrid retrieval,
   - improved chunking boundaries.

This usually reveals whether chunking choice is the main bottleneck or whether retrieval-stack improvements matter more.

---

## 8) Positioning against common 2025 alternatives

These three methods remain relevant, but they now sit alongside other high-impact retrieval improvements:

- **Semantic / structure-aware chunking**: better boundaries can improve naive and parent-child retrieval substantially.
- **Reranking**: often one of the highest-leverage upgrades for retrieval quality.
- **Hybrid retrieval**: lexical + dense retrieval helps with exact terms, IDs, and technical strings.
- **Contextual retrieval**: adds explanatory context before indexing and may outperform plain chunking in some ambiguity-heavy cases.

These methods are largely **complementary** rather than replacements. In many production systems, they matter as much as the specific choice among naive, sentence-window, and parent-child.

---

## Conclusion

The three approaches are best understood as different responses to the same RAG granularity problem:

- **Naive retrieval** uses one chunk size and accepts its trade-offs.
- **Sentence-window retrieval** retrieves very small local evidence and restores nearby context.
- **Parent-child retrieval** retrieves very small evidence and restores broader structural context.

The most defensible conclusion is not that one method universally wins, but that each fits a different failure mode:
- use **naive** for simplicity and cost control,
- use **sentence-window** for local evidence in prose,
- use **parent-child** for broader section-level context.

Then validate on your own corpus, because current evidence supports the importance of chunk granularity but does not establish a universal ranking of these three methods.

---

## Sources

### Directly accessed and used for important claims
1. **LangChain official reference – ParentDocumentRetriever**  
   https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever

2. **MongoDB official docs – Perform Parent Document Retrieval with MongoDB and LangChain**  
   https://www.mongodb.com/docs/atlas/ai-integrations/langchain/parent-document-retrieval/

3. **Anthropic – Contextual Retrieval**  
   https://www.anthropic.com/news/contextual-retrieval

4. **COLING 2025 – Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation**  
   https://aclanthology.org/2025.coling-main.384/

5. **arXiv 2024 – Evaluation of Retrieval-Augmented Generation: A Survey**  
   https://arxiv.org/abs/2405.07437

6. **arXiv 2025 – Reconstructing Context: Evaluating Advanced Chunking Strategies for Retrieval-Augmented Generation**  
   https://arxiv.org/abs/2504.19754

### Official but snippet-only / tentative evidence
7. **LlamaIndex – Sentence Window Retriever pack**  
   https://developers.llamaindex.ai/python/framework-api-reference/packs/sentence_window_retriever/

8. **LlamaIndex – Metadata Replacement + Node Sentence Window example**  
   https://developers.llamaindex.ai/python/examples/node_postprocessor/metadatareplacementdemo/

### Source note
Some LlamaIndex sentence-window details were only available via official search snippets in this session, so those specific implementation details are marked as tentative.