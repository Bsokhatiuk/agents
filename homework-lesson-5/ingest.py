from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings as config


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "files": {}
        }
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest["updated_at"] = now_iso()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def get_supported_files(data_dir: Path) -> List[Path]:
    files = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def load_documents(file_path: Path, data_dir: Path, file_hash: str) -> List[Document]:
    rel_path = file_path.relative_to(data_dir).as_posix()
    ext = file_path.suffix.lower()

    base_meta = {
        "source": rel_path,
        "file_name": file_path.name,
        "file_ext": ext,
        "file_sha256": file_hash,
    }

    if ext == ".pdf":
        raw_docs = PyPDFLoader(str(file_path)).load()
        docs = []
        for i, doc in enumerate(raw_docs):
            docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **base_meta,
                        "page": doc.metadata.get("page", i),
                        "page_label": doc.metadata.get("page_label"),
                    },
                )
            )
        return docs

    if ext in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            return []
        return [
            Document(
                page_content=text,
                metadata={
                    **base_meta,
                    "page": None,
                    "page_label": None,
                },
            )
        ]

    return []


def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=getattr(config, "chunk_size", 1000),
        chunk_overlap=getattr(config, "chunk_overlap", 200),
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def make_chunk_id(
    text: str,
    source: str,
    file_sha256: str,
    chunk_index: int,
    page: int | None,
) -> str:
    raw = f"{source}|{file_sha256}|{page}|{chunk_index}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def split_documents(
    docs: List[Document],
    splitter: RecursiveCharacterTextSplitter,
) -> Tuple[List[Document], List[str]]:
    chunks = splitter.split_documents(docs)

    chunk_ids: List[str] = []
    final_chunks: List[Document] = []

    for i, chunk in enumerate(chunks):
        chunk_id = make_chunk_id(
            text=chunk.page_content,
            source=chunk.metadata["source"],
            file_sha256=chunk.metadata["file_sha256"],
            chunk_index=i,
            page=chunk.metadata.get("page"),
        )
        chunk.metadata["chunk_id"] = chunk_id
        chunk.metadata["chunk_index"] = i
        chunk.metadata["ingested_at"] = now_iso()

        chunk_ids.append(chunk_id)
        final_chunks.append(chunk)

    return final_chunks, chunk_ids





def build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=getattr(config, "embedding_model", "text-embedding-3-large"),
        api_key=config.api_key.get_secret_value(),
    )


def load_or_create_vectorstore(index_dir: Path, embeddings: OpenAIEmbeddings) -> FAISS:
    index_file = index_dir / "index.faiss"
    pkl_file = index_dir / "index.pkl"

    if index_file.exists() and pkl_file.exists():
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    dim = len(embeddings.embed_query("dimension probe"))
    index = faiss.IndexFlatL2(dim)

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


def export_bm25_json(vectorstore: FAISS, output_path: Path) -> None:
    docs_dict = getattr(vectorstore.docstore, "_dict", {})

    rows = []
    for doc_id, doc in docs_dict.items():
        rows.append(
            {
                "id": doc_id,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    payload = {
        "generated_at": now_iso(),
        "count": len(rows),
        "documents": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_current_file_state(data_dir: Path) -> Dict[str, dict]:
    current_files = {}

    for path in get_supported_files(data_dir):
        rel_path = path.relative_to(data_dir).as_posix()
        current_files[rel_path] = {
            "path": path,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }

    return current_files


def ingest() -> None:
    data_dir = Path(config.data_dir)
    index_dir = Path(config.index_dir)

    manifest_path = Path(
        getattr(config, "manifest_path", index_dir / "manifest.json")
    )
    bm25_json_path = Path(
        getattr(config, "bm25_json_path", index_dir / "bm25_chunks.json")
    )

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    index_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    embeddings = build_embeddings()
    vectorstore = load_or_create_vectorstore(index_dir, embeddings)
    splitter = make_splitter()

    current_files = build_current_file_state(data_dir)
    old_files = manifest.get("files", {})

    current_paths = set(current_files.keys())
    old_paths = set(old_files.keys())

    deleted_paths = old_paths - current_paths
    changed_or_new_paths = []

    for rel_path, file_info in current_files.items():
        old_info = old_files.get(rel_path)
        if old_info is None or old_info.get("sha256") != file_info["sha256"]:
            changed_or_new_paths.append(rel_path)

    print(f"Found files: {len(current_files)}")
    print(f"Deleted files: {len(deleted_paths)}")
    print(f"New/changed files: {len(changed_or_new_paths)}")

    for rel_path in sorted(deleted_paths):
        old_chunk_ids = old_files[rel_path].get("chunk_ids", [])
        if old_chunk_ids:
            vectorstore.delete(ids=old_chunk_ids)
        manifest["files"].pop(rel_path, None)
        print(f"Deleted from index: {rel_path}")

    for rel_path in sorted(changed_or_new_paths):
        file_info = current_files[rel_path]
        file_path = file_info["path"]
        file_hash = file_info["sha256"]

        old_info = old_files.get(rel_path)
        if old_info:
            old_chunk_ids = old_info.get("chunk_ids", [])
            if old_chunk_ids:
                vectorstore.delete(ids=old_chunk_ids)
            print(f"Updating: {rel_path}")
        else:
            print(f"Adding: {rel_path}")

        docs = load_documents(file_path, data_dir, file_hash)
        chunks, chunk_ids = split_documents(docs, splitter)

        if chunks:
            vectorstore.add_documents(documents=chunks, ids=chunk_ids)

        manifest["files"][rel_path] = {
            "source": rel_path,
            "file_name": file_path.name,
            "file_ext": file_path.suffix.lower(),
            "sha256": file_hash,
            "size": file_info["size"],
            "mtime_ns": file_info["mtime_ns"],
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunk_ids),
            "updated_at": now_iso(),
        }

        print(f"  -> chunks: {len(chunk_ids)}")

    vectorstore.save_local(str(index_dir))
    save_manifest(manifest_path, manifest)
    export_bm25_json(vectorstore, bm25_json_path)

    print("Ingest complete")
    print(f"FAISS index: {index_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"BM25 JSON: {bm25_json_path}")


if __name__ == "__main__":
    ingest()