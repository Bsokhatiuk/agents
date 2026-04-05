"""
Hybrid retrieval module.

Combines semantic search (vector DB) + BM25 (lexical) + cross-encoder reranking.
"""

import json

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from config import settings as config



def get_retriever():
    lc_embeddings = OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=config.api_key.get_secret_value()
    )

    vectorstore = FAISS.load_local(
        folder_path=config.index_dir,
        embeddings=lc_embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": config.retrieval_top_k})

    with open(config.bm25_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    documents = []
    for row in payload["documents"]:
        documents.append(
            Document(
                page_content=row["text"],
                metadata=row.get("metadata", {})
            )
        )

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = config.retrieval_top_k

    # Ensemble: combine with weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=[0.4, 0.6])


    reranker_model = HuggingFaceCrossEncoder(model_name=config.reranker_model_name)
    compressor = CrossEncoderReranker(
        model=reranker_model,
        top_n=config.rerank_top_n
    )

    reranking_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    return reranking_retriever

