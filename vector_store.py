import logging
import os
import pickle
from typing import Any

import faiss
import numpy as np

from chunker import chunk_documents
from data_loader import load_all_documents
from embeddings import MODEL_NAME, generate_embeddings, load_embedding_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)
EMBEDDING_MODEL = None
FAISS_BUNDLE = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.index")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.pkl")
EMBEDDINGS_CACHE_PATH = os.path.join(BASE_DIR, "embeddings.npy")


def _validate_embeddings(embeddings: np.ndarray, expected_count: int) -> np.ndarray:
    """Validate and normalize embeddings before indexing."""
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2-D NumPy array.")

    if embeddings.shape[0] != expected_count:
        raise ValueError(
            f"Embedding count ({embeddings.shape[0]}) does not match chunk count ({expected_count})."
        )

    return embeddings.astype(np.float32, copy=False)

def get_embedding_model():
    global EMBEDDING_MODEL

    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = load_embedding_model(MODEL_NAME)

    return EMBEDDING_MODEL

def clear_cache():
    global FAISS_BUNDLE
    global EMBEDDING_MODEL

    FAISS_BUNDLE = None
    EMBEDDING_MODEL = None

def build_faiss_index(
    data_dir: str = DATA_DIR,
    index_path: str = FAISS_INDEX_PATH,
    chunks_path: str = CHUNKS_PATH,
    embeddings_path: str = EMBEDDINGS_CACHE_PATH,
) -> dict[str, Any]:
    """
    Build a FAISS index for the support-document corpus.

    The function loads the source documents, chunks them, then either loads
    existing embeddings from embeddings.npy or generates new ones and saves
    them for future runs. It writes the FAISS index, chunk cache, and
    embedding cache to disk.
    """
    global FAISS_BUNDLE
    logger.info("Starting FAISS index build from %s", data_dir)
    
    try:
        embeddings: np.ndarray

        if (
            os.path.exists(index_path)
            and os.path.exists(chunks_path)
            and os.path.exists(embeddings_path)
        ):
            logger.info("Using existing FAISS index and caches.")

            return load_faiss_index(
                index_path=index_path,
                chunks_path=chunks_path,
                embeddings_path=embeddings_path,
            )

        else:
            logger.info("Cache not found. Rebuilding corpus.")

            documents = load_all_documents(data_dir=data_dir)
            chunks = chunk_documents(documents)

            if not chunks:
                raise ValueError("No chunks were generated from the input documents.")

            embeddings, chunks = generate_embeddings(chunks)
            embeddings = np.asarray(embeddings, dtype=np.float32)

            np.save(embeddings_path, embeddings)

        with open(chunks_path, "wb") as handle:
            pickle.dump(chunks, handle)

        embeddings = _validate_embeddings(embeddings, expected_count=len(chunks))

        dimension = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        faiss.write_index(index, index_path)

            
        total_vectors = int(index.ntotal)
        logger.info("FAISS index saved to %s", index_path)
        logger.info("Chunk cache saved to %s", chunks_path)
        logger.info("Embedding cache saved to %s", embeddings_path)
        logger.info("Total vectors indexed: %d", total_vectors)
        logger.info("Embedding dimension: %d", dimension)
        logger.info("Index type: %s", type(index).__name__)

        print("\n================ FAISS INDEX SUMMARY ================")
        print(f"Total vectors indexed:        {total_vectors:,}")
        print(f"Embedding dimension:          {dimension}")
        print(f"Index type:                   {type(index).__name__}")
        print(f"Save location:                {index_path}")
        print(f"Chunks cache:                 {chunks_path}")
        print(f"Embeddings cache:             {embeddings_path}")
        
        print("===================================================")

        bundle = {
            "index": index,
            "chunks": chunks,
            "embeddings": embeddings,
            "index_path": index_path,
            "chunks_path": chunks_path,
            "embeddings_path": embeddings_path,
            "total_vectors": total_vectors,
            "embedding_dimension": dimension,
        }

        FAISS_BUNDLE = bundle

        return bundle

    except Exception as exc:
        logger.exception("FAISS index build failed.")
        raise RuntimeError(f"FAISS index build failed: {exc}") from exc


def load_faiss_index(
    index_path: str = FAISS_INDEX_PATH,
    chunks_path: str = CHUNKS_PATH,
    embeddings_path: str = EMBEDDINGS_CACHE_PATH,
) -> dict[str, Any]:
    """Load an existing FAISS index and its associated chunk metadata."""
    global FAISS_BUNDLE

    # Return cached bundle if already loaded
    if FAISS_BUNDLE is not None:
        logger.info("Using cached FAISS bundle.")
        return FAISS_BUNDLE

    logger.info("Loading FAISS index from %s", index_path)

    try:
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index file not found: {index_path}"
            )

        if not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"Chunk cache file not found: {chunks_path}"
            )

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embedding cache file not found: {embeddings_path}"
            )

        # Load FAISS index
        index = faiss.read_index(index_path)

        # Load chunk metadata
        with open(chunks_path, "rb") as handle:
            chunks = pickle.load(handle)

        # Load embeddings
        embeddings = np.load(embeddings_path)
        embeddings = _validate_embeddings(
            embeddings,
            expected_count=len(chunks),
        )

        bundle = {
            "index": index,
            "chunks": chunks,
            "embeddings": embeddings,
            "index_path": index_path,
            "chunks_path": chunks_path,
            "embeddings_path": embeddings_path,
            "total_vectors": int(index.ntotal),
            "embedding_dimension": int(index.d),
        }

        # Cache for future requests
        FAISS_BUNDLE = bundle

        logger.info(
            "Loaded FAISS index with %d vectors.",
            bundle["total_vectors"],
        )

        return bundle

    except Exception as exc:
        logger.exception("Failed to load FAISS index.")
        raise RuntimeError(f"Failed to load FAISS index: {exc}") from exc


def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search the indexed chunks using semantic similarity."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A non-empty query string is required for semantic search.")

    logger.info("Running semantic search for query: %s", query)

    try:
        index_bundle = load_faiss_index()
        index = index_bundle["index"]
        chunks = index_bundle["chunks"]

        model = get_embedding_model()
        query_embedding = model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        if query_embedding.shape[1] != index.d:
            raise ValueError(
                f"Query embedding dimension ({query_embedding.shape[1]}) does not match index dimension ({index.d})."
            )

        effective_top_k = min(max(int(top_k), 1), len(chunks))
        distances, indices = index.search(query_embedding, effective_top_k)

        results: list[dict[str, Any]] = []
        for score, chunk_index in zip(distances[0], indices[0]):
            if chunk_index < 0 or chunk_index >= len(chunks):
                continue

            chunk = chunks[int(chunk_index)]
            results.append(
                {
                    "score": float(score),
                    "text": chunk.page_content,
                    "metadata": dict(chunk.metadata),
                }
            )

        logger.info("Retrieved %d chunks for query '%s'.", len(results), query)
        return results

    except Exception as exc:
        logger.exception("Semantic search failed for query '%s'.", query)
        raise RuntimeError(f"Semantic search failed: {exc}") from exc


def main() -> None:
    """Build the FAISS index and run a sample search."""
    logger.info("Starting FAISS vector store demo.")

    try:
        build_faiss_index()
        sample_query = "How do I replace the toner cartridge?"
        print(f"\nSample query: {sample_query}")

        results = search(sample_query, top_k=5)
        print("\nRetrieved chunks:")
        for idx, result in enumerate(results, start=1):
            print(f"\n--- Result {idx} (score={result['score']:.4f}) ---")
            print(result["text"])
            print("Metadata:")
            print(result["metadata"])

    except Exception as exc:
        logger.exception("Vector store demo failed.")
        raise RuntimeError(f"Vector store demo failed: {exc}") from exc


if __name__ == "__main__":
    main()
