# Comparison of Naive RAG, Sentence-Window Retrieval, and Parent-Child Retrieval

## Executive Summary

These three patterns differ mainly in **what gets embedded/retrieved** and **how much surrounding context is returned to the LLM**:

- **Naive RAG**: retrieve fixed chunks and pass those chunks directly to the LLM.
- **Sentence-window retrieval**: retrieve at the **sentence** level, then expand each hit with neighboring sentences.
- **Parent-child retrieval**: retrieve **small child chunks** for precision, but return the larger **parent document/chunk** for generation.

The main trade-off is consistent across the three: **smaller retrieval units improve precision**, while **larger returned context improves completeness and grounding**. Sentence-window and parent-child retrieval are **related but not the same**:
- sentence-window retrieval expands around a **sentence** using neighboring sentences;
- parent-child retrieval maps a matched **child chunk** to its **parent document or larger chunk**.

## Findings

### 1) Definitions

#### Naive RAG
In this context, **naive RAG** means the baseline pipeline where:
1. documents are chunked into fixed-size segments,
2. chunks are embedded and indexed,
3. retrieval returns the most similar chunks,
4. those same chunks are inserted into the prompt for generation.

It typically **omits**:
- sentence-level metadata windows,
- hierarchical parent-child mapping,
- contextual expansion after retrieval,
- specialized reranking or multi-stage retrieval.

This aligns with the general RAG description in the retrieved sources: a query is encoded, relevant documents are selected, and the LLM generates using the retrieved text. The baseline is simply the simplest form of that pipeline.

#### Sentence-window retrieval
LlamaIndex’s sentence-window pack describes a pipeline that:
- builds input nodes from text,
- uses **sentences as retrieval units**,
- stores surrounding sentence context as metadata,
- and **inserts the surrounding context back into the node after retrieval** before synthesis.

That makes it a **sentence-level retrieval with local context expansion** pattern.

#### Parent-child retrieval
LangChain’s `ParentDocumentRetriever` is described as:
- splitting documents into smaller chunks for embedding and vector search,
- but returning the **original parent documents rather than individual chunks**.

So the retrieval unit is **child chunks**, while the returned generation context is **parent-level content**.

### 2) Distinction between sentence-window and parent-child retrieval

They should be treated as **different techniques**, not the same pattern.

- **Sentence-window retrieval**: retrieve a sentence, then expand to neighboring sentences via metadata/context window.
- **Parent-child retrieval**: retrieve a smaller child chunk, then replace it with its larger parent document/chunk.

These can both be viewed as **context-expansion RAG variants**, but they expand context differently:
- sentence-window = local adjacent-sentence expansion,
- parent-child = hierarchical chunk-to-parent expansion.

### 3) Authoritative source support

The most relevant authoritative sources found were:
- **LangChain docs** for `ParentDocumentRetriever`
- **LlamaIndex docs** for `SentenceWindowRetrieverPack`
- LlamaIndex node parser docs and examples indicating sentence-level nodes with surrounding metadata

The earlier web results also surfaced blog posts and tutorials, but those were not necessary for the core distinction and should be treated as secondary.

## Analysis / Comparison

### Structured Comparison Table

| Approach | Retrieval granularity | Context expansion method | Indexing / storage overhead | Latency | Accuracy / groundedness tradeoff | Typical use cases |
|---|---|---|---|---|---|---|
| **Naive RAG** | Fixed chunks | None; retrieved chunk is passed directly | Lowest complexity; one representation per chunk | Usually lowest pipeline overhead | Often less contextual completeness; may miss surrounding evidence | Simple prototypes, small corpora, baseline comparisons |
| **Sentence-window retrieval** | Sentences | Add neighboring sentences around each hit using metadata/window expansion | Higher storage than naive because sentence nodes need window metadata | Slightly higher than naive due to post-retrieval expansion | Better local context and grounding than sentence-only retrieval; may still be limited if relevant evidence spans beyond the window | Technical docs, policy text, narrow factual passages |
| **Parent-child retrieval** | Small child chunks for indexing; parent documents for return | Retrieve child chunks, then map to parent doc/chunk | Higher than naive due to dual representation or mapping between child and parent | Often higher than naive because of extra lookup step | Better chance of preserving broader evidence and coherence; may reduce precision if parents are large | Long documents, reports, manuals, legal/regulated text |

### Strengths and Weaknesses

#### Naive RAG
**Strengths**
- Simple to implement
- Minimal indexing complexity
- Easy to debug and benchmark as a baseline

**Weaknesses**
- Retrieved chunks may lack enough surrounding context
- Can fragment evidence across chunk boundaries
- Less robust when answers depend on nearby text outside the retrieved chunk

#### Sentence-window retrieval
**Strengths**
- Preserves local context around a precise sentence hit
- Helps when the answer is anchored in a specific line but needs nearby qualifiers
- Good fit for dense prose and technical text

**Weaknesses**
- Depends on the chosen window size
- Can still miss evidence outside the immediate neighborhood
- Sentence segmentation and metadata management add complexity

#### Parent-child retrieval
**Strengths**
- Balances precise retrieval with broader returned context
- Useful when small chunks improve match quality but larger context is needed to answer faithfully
- More flexible than naive chunk retrieval for long documents

**Weaknesses**
- More storage and retrieval plumbing
- If parent documents are too large, returned context may include irrelevant material
- More moving parts to tune: child size, parent size, mapping strategy

### Guidance on when to use which

- Use **naive RAG** when you need a baseline or a low-complexity prototype.
- Use **sentence-window retrieval** when relevant evidence is usually local and adjacent, such as in manuals, procedures, or policy text.
- Use **parent-child retrieval** when retrieval precision improves with smaller chunks but the generator needs larger surrounding context to answer well.

## Risks / Trade-offs

- **No universal performance ranking**: there is no evidence here that one method is always better across all corpora and tasks.
- **Benchmark dependence**: any claim that sentence-window or parent-child retrieval “improves accuracy” should be tied to a specific evaluation setup.
- **Window size / parent size sensitivity**: both advanced methods depend heavily on segmentation choices.
- **Operational complexity**: parent-child retrieval generally requires more indexing and metadata management than naive RAG.
- **Grounding vs. verbosity**: larger returned context can improve grounding but may also introduce irrelevant text or distract the generator.

A retrieved benchmark-related snippet mentioned improved groundedness for sentence-window retrieval in one notebook/blog context, but that is not strong enough to generalize without the original evaluation details. It should not be treated as a universal claim.

## Conclusion

- **Naive RAG** is the simplest baseline: retrieve chunks, pass them to the LLM.
- **Sentence-window retrieval** and **parent-child retrieval** are **distinct advanced patterns** that both add context, but in different ways.
- Sentence-window retrieval expands **around a sentence**; parent-child retrieval returns a **larger parent document/chunk** for a smaller retrieved child chunk.
- For implementation decisions, the main considerations are **retrieval granularity, context completeness, indexing overhead, and expected document structure**.

## Sources

### Authoritative / primary
- LangChain Classic docs: `ParentDocumentRetriever`  
  https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever
- LlamaIndex docs: `SentenceWindowRetrieverPack`  
  https://developers.llamaindex.ai/python/framework-api-reference/packs/sentence_window_retriever/
- LlamaIndex docs / examples: `SentenceWindowNodeParser` and metadata window examples  
  https://docs.llamaindex.ai/en/v0.10.20/examples/node_postprocessor/MetadataReplacementDemo.html

### Supporting background
- General RAG overview in local knowledge base: retrieval-augmented generation overview and baseline retrieval description
- Benchmark-related search result pointing to RAGBench:  
  https://arxiv.org/abs/2407.11005