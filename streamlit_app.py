import streamlit as st
from typing import Any

from rag_chain import DEFAULT_TOP_K, MODEL_NAME, get_answer
import vector_store
from vector_store import (
    get_embedding_model,
    load_faiss_index,
)

def load_faiss_bundle() -> dict[str, Any] | None:
    try:
        return load_faiss_index()
    except Exception as exc:
        st.session_state.index_error = str(exc)
        return None


@st.cache_resource(show_spinner=False)
def get_embedding_model_cached():
    return get_embedding_model()


@st.cache_resource(show_spinner=False)
def get_faiss_bundle_cached():
    return load_faiss_bundle()


def initialize_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "index_error" not in st.session_state:
        st.session_state.index_error = ""
    if "groq_error" not in st.session_state:
        st.session_state.groq_error = ""


def render_sidebar(bundle: dict[str, Any] | None, embedding_loaded: bool) -> None:
    st.sidebar.title("System Status")

    if bundle is not None:
        st.sidebar.success("FAISS Index Loaded")
    else:
        st.sidebar.error("FAISS Index Not Loaded")

    if embedding_loaded:
        st.sidebar.success("Embedding Model Loaded")
    else:
        st.sidebar.warning("Embedding Model Not Loaded")

    st.sidebar.divider()
    st.sidebar.markdown("**Groq Model**")
    st.sidebar.write(MODEL_NAME)

    st.sidebar.markdown("**Top K Retrieval**")
    st.sidebar.write(DEFAULT_TOP_K)

    st.sidebar.divider()
    st.sidebar.markdown("### Dataset Statistics")
    if bundle is not None:
        chunks = bundle.get("chunks", [])
        st.sidebar.markdown(f"- Total Chunks: {len(chunks):,}")
        st.sidebar.markdown(
            f"- Total Vectors: {bundle.get('total_vectors', 'N/A'):,}"
        )
        st.sidebar.markdown(
            f"- Embedding Dimension: {bundle.get('embedding_dimension', 'N/A')}"
        )
    else:
        st.sidebar.write("No dataset stats available.")

    st.sidebar.divider()
    st.sidebar.markdown("### Chat Controls")
    if st.sidebar.button("Clear Chat"):
        st.session_state.history = []
        st.rerun()


def render_chat_history() -> None:
    for entry in st.session_state.history:
        role = entry.get("role")
        content = entry.get("content", "")
        sources = entry.get("sources")

        with st.chat_message(role):
            st.markdown(content)

        if role == "assistant" and sources:
            with st.expander("View Retrieved Sources"):
                for idx, source in enumerate(sources, start=1):
                    metadata = source.get("metadata", {})
                    score = source.get("score")
                    source_file = metadata.get("source_file", metadata.get("source", "Unknown"))
                    page_number = metadata.get("page_number", metadata.get("page", "N/A"))
                    source_type = metadata.get("source_type", "Unknown")

                    st.markdown(f"**Source {idx}**")
                    st.markdown(f"- **File:** {source_file}")
                    st.markdown(f"- **Type:** {source_type}")
                    st.markdown(f"- **Page:** {page_number}")
                    st.markdown(f"- **Score:** {score:.4f}" if isinstance(score, (int, float)) else "- **Score:** N/A")
                    st.write(metadata)
                    st.write(source.get("text", ""))
                    if idx < len(sources):
                        st.markdown("---")


def main() -> None:
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
    st.title("Customer Support Chatbot")
    st.write("Ask questions about the knowledge base and get answers with source-backed retrieval.")

    initialize_session_state()

    embeddings_available = vector_store.EMBEDDING_MODEL is not None
    faiss_bundle = None
    try:
        faiss_bundle = get_faiss_bundle_cached()
    except Exception:
        faiss_bundle = None

    try:
        if not embeddings_available:
            _ = get_embedding_model_cached()
            embeddings_available = True
    except Exception as exc:
        embeddings_available = False
        st.session_state.groq_error = str(exc)

    render_sidebar(faiss_bundle, embeddings_available)

    if st.session_state.index_error:
        st.error(f"FAISS load error: {st.session_state.index_error}")
    if not embeddings_available:
        st.error("Embedding model failed to load. Check the server log and ensure sentence-transformers is installed.")

    if faiss_bundle is None or not embeddings_available:
        st.stop()

    render_chat_history()
    user_input = st.chat_input("Ask a question")

    if user_input:
        user_input = user_input.strip()

        if not user_input:
            st.warning("Please enter a question before sending.")
        else:
            st.session_state.history.append({"role": "user", "content": user_input})

            try:
                with st.spinner("Searching documents and generating answer..."):
                    retrieval = get_answer(
                        user_input,
                        return_sources=True,
                    )

                answer = retrieval.get("answer", "")
                sources = retrieval.get("sources", [])

                st.session_state.history.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
                st.rerun()

            except Exception as exc:
                error_message = str(exc)
                st.session_state.history.append(
                    {"role": "assistant", "content": f"Error: {error_message}", "sources": []}
                )
                st.rerun()
    


if __name__ == "__main__":
    main()
