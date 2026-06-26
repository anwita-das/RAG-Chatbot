# RAG-Based Customer Support Chatbot 🤖

## Overview

This project is a **Retrieval-Augmented Generation (RAG)** based customer support chatbot that answers user queries using information from product manuals, FAQ documents, and Amazon product Question-Answer data instead of relying solely on a Large Language Model (LLM).

Instead of allowing the LLM to generate answers from its own knowledge, the system first retrieves the most relevant information from the knowledge base using semantic search and then asks the LLM to generate an answer only from the retrieved context. This reduces hallucinations and improves factual accuracy.

---

# Features

* Multi-source document ingestion
* Automatic document chunking
* Semantic embeddings using Sentence Transformers
* Fast similarity search using FAISS
* Retrieval-Augmented Generation (RAG)
* Groq Llama 3.3 70B integration
* Source-aware retrieval
* Cached embeddings and vector index for faster startup
* Command Line Interface (CLI)

---

# Project Workflow

```
               Documents
        (CSV + JSON + PDF Manuals)
                     │
                     ▼
              Data Loader
                     │
                     ▼
          Recursive Text Chunking
                     │
                     ▼
      Sentence Embedding Generation
        (BAAI/bge-small-en-v1.5)
                     │
                     ▼
          FAISS Vector Index Creation
                     │
                     ▼
              User Question
                     │
                     ▼
          Query Embedding Creation
                     │
                     ▼
        Semantic Similarity Search
                     │
                     ▼
       Top-K Relevant Document Chunks
                     │
                     ▼
      Prompt Construction (RAG Prompt)
                     │
                     ▼
          Groq Llama 3.3 70B LLM
                     │
                     ▼
             Final Generated Answer
```

---

# Tech Stack

### Programming Language

* Python 3.11+

### Libraries

* LangChain
* Sentence Transformers
* FAISS
* NumPy
* python-dotenv
* LangChain-Groq

### Embedding Model

* BAAI/bge-small-en-v1.5

### Large Language Model

* Llama-3.3-70B-Versatile (Groq API)

### Vector Store

* FAISS (Facebook AI Similarity Search)

### Data Sources

* Amazon Product Question Answer Dataset
* Ecommerce FAQ Dataset
* Product Manuals (PDF)

---

# Project Structure

```
chatbot/
│
├── data/
│   ├── manuals/
│   ├── Ecommerce_FAQ_Chatbot_dataset.json
│   └── single_qna.csv
│
├── data_loader.py
├── chunker.py
├── embeddings.py
├── vector_store.py
├── rag_chain.py
│
├── chunks.pkl
├── embeddings.npy
├── faiss_index.index
│
├── .env
├── requirements.txt
└── README.md
```

---

# How It Works

## 1. Data Loading

The system loads documents from multiple sources:

* Amazon Product Question-Answer dataset
* Ecommerce FAQ dataset
* Product manuals (PDF)

Each source is converted into LangChain `Document` objects while preserving metadata.

---

## 2. Document Chunking

Large documents are split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`.

Current configuration:

* Chunk Size: 500 characters
* Chunk Overlap: 50 characters

Chunking improves retrieval accuracy because embeddings represent smaller, more focused pieces of information.

---

## 3. Embedding Generation

Each chunk is converted into a dense vector using:

**BAAI/bge-small-en-v1.5**

The vectors are normalized before being stored.

Generated embeddings are cached inside:

```
embeddings.npy
```

This prevents regenerating embeddings every time the application starts.

---

## 4. Vector Store

The embeddings are indexed using **FAISS IndexFlatIP** for fast semantic similarity search.

The generated index is stored as:

```
faiss_index.index
```

The corresponding chunk objects are stored in:

```
chunks.pkl
```

These cached files allow future runs to load instantly without rebuilding the vector database.

---

## 5. Query Processing

When a user asks a question:

1. The question is embedded using the same embedding model.
2. FAISS searches for the most similar chunks.
3. The top-K relevant chunks are retrieved.
4. Retrieved chunks are combined into a single context.

---

## 6. Retrieval-Augmented Generation (RAG)

The retrieved context is inserted into a prompt along with the user's question.

The prompt instructs the LLM to:

* answer only from the provided context
* avoid hallucinations
* state when the answer is unavailable

The prompt is then sent to Groq's hosted Llama model.

---

## 7. Response Generation

Groq's Llama 3.3 70B model generates a natural language response using only the retrieved document context.

---

# Datasets Used

## Amazon Question Answer Dataset

This project uses the **Amazon Question/Answer Dataset** by Pranesh Mukhopadhyay, which contains Amazon customer questions and answers across multiple product categories. The original dataset is based on research data collected by Prof. Julian McAuley (UC San Diego).

Dataset:

https://www.kaggle.com/datasets/praneshmukhopadhyay/amazon-questionanswer-dataset

**Note:** The dataset is **not included in this repository** because it exceeds GitHub's file size limits. Please download it manually from Kaggle and place the required CSV file inside the `data/` directory before running the project.

Additional data sources include:

* Ecommerce FAQ dataset
* Product manuals in PDF format

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd chatbot
```

---

## 2. Create a virtual environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download the Amazon Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/praneshmukhopadhyay/amazon-questionanswer-dataset

Copy the required CSV file (`single_qna.csv`) into:

```
data/
```

---

## 5. Add your Groq API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

# Running the Project

### Build embeddings and FAISS index (first run)

```bash
python vector_store.py
```

This creates:

* `embeddings.npy`
* `chunks.pkl`
* `faiss_index.index`

These files are reused in future runs.

---

### Start the chatbot

```bash
python rag_chain.py
```

Example:

```
Question:
How do I replace the toner cartridge?

Answer:
...
```

---

# Future Improvements

* BM25 + Dense Hybrid Retrieval
* Cross-Encoder Re-ranking
* Streamlit Web Interface
* Conversation Memory
* Query Expansion
* Metadata Filtering
* Support for Multiple Embedding Models
* Docker Deployment
* REST API using FastAPI
* Evaluation using RAGAS

---

# Author

**Anwita Das**
