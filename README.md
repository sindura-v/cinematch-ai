# CineMatch AI — Personalised Movie Recommender

An AI-powered movie recommendation app built with RAG 
(Retrieval Augmented Generation).

## How it works
1. Describe your mood tonight in plain English
2. The app searches your personal watch history using 
   semantic similarity (RAG retrieval)
3. Fetches highly-rated recent movies from TMDB 
   matching your taste
4. GPT-4o-mini generates a personalised recommendation 
   pitch based on your history

## RAG Architecture
- Ingestion: personal watchlist embedded using 
  OpenAI text-embedding-3-small and stored in ChromaDB
- Retrieval: mood query embedded and matched against 
  watchlist vectors using cosine similarity
- Generation: GPT-4o-mini generates recommendation 
  using retrieved context + TMDB data

## Tech stack
- Python 3.13
- OpenAI API (embeddings + GPT-4o-mini)
- ChromaDB — vector database
- TMDB API — movie database
- Streamlit — web interface
- pandas, requests, python-dotenv

## Setup
1. Clone the repo
2. python -m venv venv and activate
3. pip install -r requirements.txt
4. Copy .env.example to .env and add your API keys
5. Run ingestion once: python ingest.py
6. Launch app: streamlit run app.py

## What I learned
- Building a RAG pipeline from scratch
- Vector embeddings and semantic similarity search
- ChromaDB vector database
- Integrating multiple APIs in one pipeline
- Streamlit web interface development
- Prompt engineering for personalised output

## Known limitations & planned improvements
- Language filtering (v2): add original language 
  filter via TMDB API
- Multi-turn conversation (v2): allow users to 
  refine recommendations using Streamlit session state
- Chunking (v3): extend to long-form content like 
  reviews and articles