import logging
import os
from typing import Any, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from vector_store import search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_TOP_K = 5
GROQ_CLIENT = None

def _load_env() -> None:
    load_dotenv()

def _get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Please set GROQ_API_KEY in the .env file."
        )
    return api_key

def get_groq_client():
    global GROQ_CLIENT

    if GROQ_CLIENT is None:
        _load_env()

        GROQ_CLIENT = ChatGroq(
            model=MODEL_NAME,
            api_key=_get_groq_api_key(),
            temperature=0.0,
            max_tokens=512,
            streaming=False,
        )

    return GROQ_CLIENT

def retrieve_context(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    return_sources: bool = False,
) -> str | dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A non-empty query is required for context retrieval.")

    logger.info("Retrieving context for query: %s", query)
    results = search(query, top_k=top_k)
    
    logger.info("Retrieved %d chunks for query.", len(results))

    chunk_texts: List[str] = []
    for idx, chunk in enumerate(results, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source_file", "Unknown")
        source_type = metadata.get("source_type", "Unknown")

        chunk_texts.append(
            f"""
--- Chunk {idx} ---
Source: {source}
Type: {source_type}

{chunk['text']}
"""
        )

    context = "\n\n".join(chunk_texts)
    if return_sources:
        return {"context": context, "sources": results}
    return context


def generate_answer(question: str, context: str) -> str:
    if not isinstance(context, str) or not context.strip():
        return "I could not find that information in the available documents."

    prompt = build_prompt(question, context)

    logger.info("Sending prompt to Groq.")
    client = get_groq_client()

    messages = [[HumanMessage(content=prompt)]]
    response = client.generate(messages)

    generations = getattr(response, "generations", None)
    if not generations or not generations[0]:
        raise RuntimeError("Groq response contained no generated text.")

    answer = generations[0][0].text
    if answer is None:
        raise RuntimeError("Groq response did not include an answer text.")

    logger.info("Groq request succeeded.")
    return answer.strip()


def build_prompt(question: str, context: str) -> str:
    return f"""
You are a helpful customer support assistant.

Answer the user's question using ONLY the information present in the provided context.

Rules:
- Do not make up information.
- Do not use outside knowledge.
- Answer only from the provided context.
- Do not repeat the user's question.
- Answer naturally as a customer support assistant.
- Use bullet points when helpful.
- If the answer cannot be found in the context, say:
  "I could not find that information in the available documents."
- Keep answers concise and clear.

Context:
{context}

Question:
{question}

Answer:
"""


def get_answer(question: str, top_k: int = DEFAULT_TOP_K, return_sources: bool = False) -> str | dict[str, Any]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Please provide a non-empty question.")

    _load_env()
    retrieval = retrieve_context(
        question,
        top_k=top_k,
        return_sources=True,
    )
    context = retrieval["context"]
    sources = retrieval.get("sources", [])

    if not context.strip():
        if return_sources:
            return {
                "answer": "I could not find that information in the available documents.",
                "sources": sources,
            }
        return "I could not find that information in the available documents."

    try:
        answer = generate_answer(question, context)
    except Exception as exc:
        logger.exception("Groq request failed.")
        raise RuntimeError(f"Groq API request failed: {exc}") from exc

    if return_sources:
        return {"answer": answer, "sources": sources}
    return answer


def main() -> None:
    print("Retrieval-Augmented Generation CLI. Type 'exit' to quit.")

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            print("Please enter a question or type 'exit'.")
            continue

        if question.lower() == "exit":
            print("Goodbye.")
            break

        try:
            answer = get_answer(question)
            print("\nAnswer:\n", answer)
        except Exception as exc:
            logger.error("Failed to get answer: %s", exc)
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
