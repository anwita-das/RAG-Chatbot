import statistics
from typing import List

from data_loader import Document, load_all_documents

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise ImportError(
            "LangChain RecursiveCharacterTextSplitter is required. "
            "Install langchain or langchain-text-splitters."
        ) from exc


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split LangChain documents into chunks while preserving metadata.

    Parameters
    ----------
    documents:
        Original documents loaded from the data loader.

    Returns
    -------
    list[Document]
        Chunked documents with metadata preserved.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(documents)

    if not chunks:
        print("\nNo chunks generated.")
        return []

    chunk_lengths = [len(chunk.page_content) for chunk in chunks]

    print("\n================ CHUNKING SUMMARY ================")
    print(f"Original document count:     {len(documents):,}")
    print(f"Chunk count after splitting: {len(chunks):,}")
    print(f"Average chunk length:        {statistics.mean(chunk_lengths):.2f} characters")
    print(f"Minimum chunk length:        {min(chunk_lengths) if chunk_lengths else 0} characters")
    print(f"Maximum chunk length:        {max(chunk_lengths) if chunk_lengths else 0} characters")
    print("==================================================")

    # Debug: inspect one chunk
    print("\n--------------- SAMPLE CHUNK ---------------")
    print(chunks[0].page_content[:500])

    print("\nMetadata:")
    print(chunks[0].metadata)
    print("--------------------------------------------")

    return chunks


if __name__ == "__main__":
    documents = load_all_documents()
    chunks = chunk_documents(documents)
    print(f"Loaded {len(documents):,} documents and created {len(chunks):,} chunks.")
