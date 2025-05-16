# 🐉 RAG: Game of Thrones Chatbot

![A Song of Ice and Fire Box Set](images/george-r-r-martin-s-a-game-of-thrones-5-book-boxed-set-song-of-ice-and-fire-series.jpg)

## 🧠 Introduction

Welcome to my **Game of Thrones AI Chatbot**, powered by **RAG (Retrieval-Augmented Generation)**.  
This project combines my passion for the *Game of Thrones* universe with what I’ve learned during a recent AI Apprenticeship.

The goal?  
To build a **lore-accurate question-answering system** that understands the complex world of Westeros — from the Red Keep to the Wall.

---

## 📚 Dataset

The chatbot’s knowledge base consists of the following five books by George R. R. Martin:

- *A Game of Thrones*
- *A Clash of Kings*
- *A Storm of Swords*
- *A Feast for Crows*
- *A Dance with Dragons*

> ⚠️ **Note:** Due to copyright restrictions, I cannot provide the raw text files or links to the books.

---

## 🧩 How It Works: RAG Pipeline

RAG stands for **Retrieval-Augmented Generation** — a framework that enhances LLMs by pairing them with external knowledge sources.

![RAG](images/classicalrag.png)

Here’s how the pipeline operates behind the scenes:

### 1. 🔧 Data Preparation

- The full book texts are **chunked** into overlapping sections.
- These chunks are converted into **dense embeddings** using a pre-trained model (e.g., OpenAI, HuggingFace).
- We store the chunks + metadata in a **FAISS** vector database for efficient similarity search.

### 2. 🔎 Query + Retrieval

- When a user submits a question (e.g., *"What happened at the Red Wedding?"*), it’s also embedded.
- The embedding is used to search the vector store and retrieve the top `k` most relevant passages.

### 3. 💬 Generation

- The **query + retrieved context** is injected into a prompt template.
- This template is passed to a **Large Language Model (LLM)** like GPT-4, Gemini, or Mixtral via LangChain.
- The model then generates an answer grounded in the books' content — not just general internet knowledge.

---

## 🧠 Models & Tooling

The system uses several components:

| Component            | Tool/Library              |
|----------------------|---------------------------|
| Embeddings           | ![Hugging Face](images/huggingface.png) |
| Vector Store         | ![FAISS](images/faiss.jpg) |
| LLM Interface        | ![Langchain](images/langchain.png) |
| Reranker (Optional)  | ![Cohere](images/cohere.png) |
| Observability        | ![Langfuse](images/langfuse.png) |
| Evaluation Metrics   | ![RAGAS](images/ragas.png) |
| Frontend   | ![Gradio](images/gradio.jpg) |

> 🔁 Supports both single-query and multi-query retrieval  
> 🔄 Reranking enabled via `ContextualCompressionRetriever`

---

## 🧪 Evaluation with RAGAS

I also implemented an **automated evaluation pipeline** using [RAGAS](https://github.com/explodinggradients/ragas):

- ✅ **Faithfulness** – Is the answer backed by the retrieved context?
- ✅ **Answer Relevance** – Does the answer fully respond to the query?
- ✅ **Context Precision & Recall** – Are the retrieved documents relevant and sufficient?

Evaluations are reproducible and logged to `Langfuse`, enabling robust testing across LLMs, retrievers, and prompts.

---

## ✨ Example Queries

- *Who is Jon Snow's real mother?*  
- *What are the three betrayals Daenerys was warned of?*  
- *Describe the Red Wedding in detail.*  
- *What does the prophecy of Azor Ahai say?*

The chatbot provides **text-grounded answers**, referencing exact content from the books — not hallucinations.

---

## 🚀 Launch Instructions

To launch the Gradio interface locally:

```bash
python app.py
```

---

## 🧱 Docker Setup

To containerize and run the chatbot using Docker, follow these steps:

### 📋 Prerequisites

Before building the Docker image, ensure the following:

- Docker is installed on your system ([Get Docker](https://docs.docker.com/get-docker/))
- You have a valid `requirements.txt` in the project root
- Your `.env` file (containing API keys and environment variables) exists under `src/.env`

### 📦 Build Docker Image

Run this command from the project root to build the image:

```bash
bash scripts/build_docker.sh
```

After building the image, run the below to spin up a docker container:

```bash
bash scripts/run_docker.sh
```