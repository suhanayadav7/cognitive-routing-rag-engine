# cognitive-routing-rag-engine

Vector-based persona routing, LangGraph autonomous content generation, and RAG-powered debate engine with prompt injection defense — built on ChromaDB, LangChain, and Groq LLaMA.

## Setup

```bash
git clone https://github.com/suhanayadav7/cognitive-routing-rag-engine
cd cognitive-routing-rag-engine
pip install -r requirements.txt
cp .env.example .env  # add your Groq API key
python main.py
```

## Phases

**Phase 1 – Vector Persona Router**: Embeds 3 bot personas into ChromaDB. Routes incoming posts to relevant bots using cosine similarity.

**Phase 2 – LangGraph Content Engine**: 3-node state machine (decide_search → web_search → draft_post). Each bot researches a topic and generates a structured JSON tweet.

**Phase 3 – RAG Combat Engine**: Full thread context injected as RAG prompt. Bot defends its position against human replies. Includes prompt injection defense via system-level persona immutability rules.

## Tech Stack

- ChromaDB (vector store)
- LangChain + LangGraph (orchestration)
- Groq LLaMA 3.3 70B (LLM)
- Python 3.11
