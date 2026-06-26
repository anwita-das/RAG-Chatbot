import logging
import time
from typing import List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise ImportError(
        "sentence-transformers is required for embeddings generation. "
        "Install it with: pip install sentence-transformers"
    ) from exc

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.docstore.document import Document

from chunker import chunk_documents
from data_loader import load_all_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Load and cache the sentence embedding model.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier for the embedding model.

    Returns
    -------
    SentenceTransformer
        A loaded embedding model instance.
    """
    try:
        logger.info("Loading embedding model: %s", model_name)
        model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")
        return model
    except Exception as exc:
        logger.exception("Failed to load embedding model '%s'.", model_name)
        raise RuntimeError(f"Unable to load embedding model '{model_name}': {exc}") from exc


def generate_embeddings(chunks: List[Document]) -> Tuple[np.ndarray, List[Document]]:
    """
    Generate embeddings for a list of LangChain documents.

    Parameters
    ----------
    chunks:
        A list of LangChain Document objects whose page_content should be embedded.

    Returns
    -------
    tuple[list[list[float]], list[Document]]
        A tuple containing the generated embeddings and the input chunk list.
    """
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("generate_embeddings() requires a non-empty list of LangChain Document chunks.")

    start_time = time.perf_counter()

    try:
        valid_chunks = [chunk for chunk in chunks if isinstance(chunk, Document) and isinstance(chunk.page_content, str) and chunk.page_content.strip()]
        if not valid_chunks:
            raise ValueError("No valid document text found in the provided chunks.")

        logger.info("Generating embeddings for %d chunks using %s", len(valid_chunks), MODEL_NAME)

        model = load_embedding_model(MODEL_NAME)
        texts = [chunk.page_content for chunk in valid_chunks]

        embeddings = model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        embedding_dimension = embeddings.shape[1]
        elapsed_seconds = time.perf_counter() - start_time

        logger.info("Embedding generation completed successfully.")
        logger.info("Chunk count: %d", len(valid_chunks))
        logger.info("Embedding dimension: %d", embedding_dimension)
        logger.info("Total embeddings generated: %d", len(embeddings))
        logger.info("Time taken: %.4f seconds", elapsed_seconds)

        print("\n================ EMBEDDING SUMMARY ================")
        print(f"Number of chunks:             {len(valid_chunks):,}")
        print(f"Embedding dimension:          {embedding_dimension}")
        print(f"Total embeddings generated:   {len(embeddings):,}")
        print(f"Time taken:                   {elapsed_seconds:.4f} seconds")
        print("==================================================")

        return embeddings, valid_chunks

    except Exception as exc:
        logger.exception("Embedding generation failed.")
        raise RuntimeError(f"Embedding generation failed: {exc}") from exc


def main() -> None:
    """
    Load, chunk, and embed the customer support dataset.

    This function validates the embedding pipeline end to end without using
    retrieval, FAISS, Groq, or Streamlit.
    """
    logger.info("Starting embedding pipeline.")

    try:
        documents = load_all_documents()
        logger.info("Loaded %d raw documents for chunking.", len(documents))

        chunks = chunk_documents(documents)
        logger.info("Chunking complete. %d chunks created.", len(chunks))

        embeddings, chunk_list = generate_embeddings(chunks)

        print("\nEmbedding pipeline completed successfully.")
        print(f"Returned embeddings: {len(embeddings):,}")
        print(f"Returned chunks:     {len(chunk_list):,}")

    except Exception as exc:
        logger.exception("Embedding pipeline failed.")
        raise RuntimeError(f"Embedding pipeline failed: {exc}") from exc


if __name__ == "__main__":
    main()
