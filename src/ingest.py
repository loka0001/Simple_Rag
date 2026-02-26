# src/ingest.py
"""
Data ingestion, chunking, and indexing pipeline.

Responsibilities:
- Load documents from various file formats (txt, pdf, csv, docx).
- Auto-detect and handle Arabic encoding / normalization.
- Split documents into token-aware chunks.
- Embed chunks using a multilingual model and persist them to ChromaDB.
- Optionally index chunks into Elasticsearch for hybrid (BM25) search.
"""

import logging
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from src.arabic_utils import (
    detect_language,
    load_arabic_directory,
)
from src.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

def load_documents(data_dir: str) -> List[Document]:
    """
    Load documents from a directory, using Arabic-aware loaders for .txt/.pdf/.docx
    and standard loaders for .csv and other formats.

    Automatically detects Arabic content and applies normalization.

    Args:
        data_dir: Path to the directory containing raw documents.

    Returns:
        A flat list of LangChain Document objects with language metadata.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Arabic-aware loader handles .txt, .pdf, .docx (with encoding detection + normalization)
    logger.info("Loading documents from '%s' with Arabic-aware loaders...", data_dir)
    docs = load_arabic_directory(data_dir)

    # Also load CSV with the standard loader (CSV doesn't need Arabic-specific handling)
    csv_loader = DirectoryLoader(str(path), glob="**/*.csv", loader_cls=CSVLoader, silent_errors=True)
    try:
        csv_docs = csv_loader.load()
        docs.extend(csv_docs)
    except Exception as exc:
        logger.warning("CSV loading error: %s", exc)

    # Tag each document with its detected language if not already tagged
    for doc in docs:
        if "language" not in doc.metadata:
            doc.metadata["language"] = detect_language(doc.page_content)

    ar_count = sum(1 for d in docs if d.metadata.get("language") == "ar")
    logger.info(
        "Loaded %d documents total (%d Arabic, %d other).",
        len(docs), ar_count, len(docs) - ar_count,
    )
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split a list of documents into token-aware chunks.

    Arabic documents use the same splitter but with slightly larger overlap
    to preserve semantic boundaries in morphologically rich Arabic text.

    Args:
        documents:     Source documents to split.
        chunk_size:    Maximum token count per chunk. Defaults to config value.
        chunk_overlap: Number of overlapping tokens between adjacent chunks.

    Returns:
        A list of chunked Document objects with inherited metadata.
    """
    cfg = get_settings()
    chunks: List[Document] = []

    for doc in documents:
        lang = doc.metadata.get("language", "en")
        # Use larger overlap for Arabic to preserve morphological context
        overlap = (chunk_overlap or cfg.chunk_overlap) * 2 if lang == "ar" else (chunk_overlap or cfg.chunk_overlap)

        splitter = TokenTextSplitter(
            chunk_size=chunk_size or cfg.chunk_size,
            chunk_overlap=overlap,
        )
        doc_chunks = splitter.split_documents([doc])
        # Propagate language metadata to chunks
        for chunk in doc_chunks:
            chunk.metadata["language"] = lang
        chunks.extend(doc_chunks)

    logger.info("Produced %d chunks from %d documents.", len(chunks), len(documents))
    return chunks


# ---------------------------------------------------------------------------
# Embedding & Vector Store
# ---------------------------------------------------------------------------

def get_chroma_collection(collection_name: Optional[str] = None) -> chromadb.Collection:
    """
    Return (or create) a persistent ChromaDB collection.

    Args:
        collection_name: Override the default collection name.

    Returns:
        A ChromaDB Collection object.
    """
    cfg = get_settings()
    client = chromadb.PersistentClient(
        path=cfg.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    name = collection_name or cfg.chroma_collection_name
    collection = client.get_or_create_collection(name=name)
    logger.info("Using ChromaDB collection '%s' at %s", name, cfg.chroma_persist_dir)
    return collection


def index_chunks(
    chunks: List[Document],
    collection: chromadb.Collection,
    embedding_model: Optional[str] = None,
) -> None:
    """
    Embed document chunks using a multilingual model and upsert them into ChromaDB.

    The default embedding model (paraphrase-multilingual-MiniLM-L12-v2) supports
    50+ languages including Arabic, enabling semantic search across mixed corpora.

    Args:
        chunks:          Chunked documents to index.
        collection:      Target ChromaDB collection.
        embedding_model: Sentence-Transformers model name. Defaults to config.
    """
    cfg = get_settings()
    model_name = embedding_model or cfg.embedding_model
    model = SentenceTransformer(model_name)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    logger.info(
        "Embedding %d chunks with multilingual model '%s'...", len(chunks), model_name
    )
    try:
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Indexed %d chunks into ChromaDB.", len(chunks))
    except Exception as exc:
        logger.error("Failed to index chunks into ChromaDB: %s", exc)
        raise


def index_to_elasticsearch(chunks: List[Document], index_name: Optional[str] = None) -> None:
    """
    Index document chunks into Elasticsearch for keyword-based (BM25) retrieval.

    Arabic documents are indexed with their normalized text and tagged with
    language metadata for potential language-specific querying.

    Args:
        chunks:     Chunked documents to index.
        index_name: Override the default Elasticsearch index name.
    """
    from elasticsearch import Elasticsearch, ConnectionError as ESConnectionError

    cfg = get_settings()
    if not cfg.es_host:
        logger.warning("ES_HOST not configured; skipping Elasticsearch indexing.")
        return

    try:
        es = Elasticsearch(
            hosts=[cfg.es_host],
            http_auth=(cfg.es_username, cfg.es_password),
        )
        if not es.ping():
            raise ESConnectionError("Elasticsearch ping failed.")
    except Exception as exc:
        logger.error("Cannot connect to Elasticsearch: %s", exc)
        return

    idx = index_name or cfg.es_index_name
    if not es.indices.exists(index=idx):
        # Arabic analyzer support for better tokenization
        es.indices.create(
            index=idx,
            body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "arabic_analyzer": {
                                "type": "arabic"
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "arabic",
                            "fields": {"english": {"type": "text", "analyzer": "english"}},
                        },
                        "language": {"type": "keyword"},
                        "source": {"type": "keyword"},
                    }
                },
            },
        )
        logger.info("Created Elasticsearch index '%s' with Arabic analyzer.", idx)

    for i, chunk in enumerate(chunks):
        try:
            es.index(
                index=idx,
                id=i,
                body={
                    "content": chunk.page_content,
                    "language": chunk.metadata.get("language", "unknown"),
                    "source": chunk.metadata.get("source", ""),
                },
            )
        except Exception as exc:
            logger.warning("Failed to index chunk %d to ES: %s", i, exc)

    logger.info("Indexed %d chunks into Elasticsearch index '%s'.", len(chunks), idx)


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def run_ingestion_pipeline(data_dir: str, use_elasticsearch: bool = False) -> None:
    """
    Full ingestion pipeline: load → chunk → embed → index.

    Arabic files are automatically detected and processed with appropriate
    encoding detection and text normalization.

    Args:
        data_dir:          Directory containing raw source documents.
        use_elasticsearch: Also index into Elasticsearch when True.
    """
    documents = load_documents(data_dir)
    if not documents:
        logger.warning("No documents found in '%s'. Ingestion aborted.", data_dir)
        return

    chunks = chunk_documents(documents)
    collection = get_chroma_collection()
    index_chunks(chunks, collection)

    if use_elasticsearch:
        index_to_elasticsearch(chunks)

    logger.info("Ingestion pipeline complete.")
