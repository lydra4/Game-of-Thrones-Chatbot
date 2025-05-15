# RAG: Game of Thrones Chatbot

![Alt text](images/george-r-r-martin-s-a-game-of-thrones-5-book-boxed-set-song-of-ice-and-fire-series.jpg)

## Overview

I just completed an AI Apprenticeship program and immersed myself in state of the art Generative AI models. 

On a personal side, I am engrossed in the Game of Thrones Media therefore I thought to combine both. 

Thus the idea was formed to create a Game of Thrones Chatbot.

## Datasets

The dataset are the five books by George R R Martin (shown in the above image).

Due to copyright issues, I am unable to provide a link to the 5 books.

## RAG Overview

**RAG** stands for **Retrieval-Augmented Generation** — a technique that enhances language models by allowing them to access external information sources.

Here’s a step-by-step overview of how the chatbot functions:

1. **Data Preparation**  
   The raw dataset (the full text of *A Song of Ice and Fire* books) is chunked into manageable sections and converted into numerical representations (embeddings) using a pre-trained embedding model.

2. **Storage**  
   These embeddings are stored in a **vector database**, allowing for fast and efficient similarity-based retrieval.

3. **User Query**  
   When a user asks a question (e.g., *"Who is Jon Snow's real mother?"*), the query is also converted into an embedding.

4. **Contextual Retrieval**  
   The vector database is searched to find chunks of text that are most similar to the query. These chunks serve as relevant **context** for the answer.

5. **Response Generation**  
   The original query, the retrieved context, and a prompt template (defining the task and tone) are passed to the **LLM** (Large Language Model).

6. **Answer**  
   The LLM generates a response that blends its general knowledge with the retrieved, grounded information — ensuring answers are both relevant and lore-accurate.

---

This approach allows the chatbot to answer detailed lore questions about Game of Thrones, even though the base language model was never explicitly trained on the books themselves.

---
