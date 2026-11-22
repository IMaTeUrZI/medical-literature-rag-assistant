# Medical Literature Assistant (RAG for Stroke & miRNA Research)

A Retrieval-Augmented Generation (RAG) web application built for querying, indexing, and synthesizing biomedical literature from **NCBI PubMed**, with a focus on **microRNA (miRNA) biomarkers in ischemic and hemorrhagic stroke**.

## Key Features

- **Two-Stage Retrieval with Cross-Encoder Reranking:** Combines initial vector similarity search in **ChromaDB** with a local Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to accurately distinguish closely related molecular identifiers (e.g., `miR-9-3p` vs. `miR-9-5p`).
- **Dynamic PubMed Ingestion Pipeline:** Fetches live abstracts via **Biopython (`Bio.Entrez`)**, splits text using `RecursiveCharacterTextSplitter`, and upserts embeddings with structured metadata (`PMID`, `title`, `year`, `chunk`).
- **Metadata Filtering & Source Transparency:** Filters publications by publication year (`$gte`), displays indexed articles in the sidebar, and presents exact relevance scores alongside source abstracts.
- **Grounded Answer Generation:** Uses **Gemini (`gemini-3.5-flash-lite`)** via the OpenAI-compatible SDK with strict prompt grounding and mandatory inline `(PMID, year)` citations to minimize hallucinations.

## Architecture

```text
User Query
   │
   ▼  Stage 1: Vector Retrieval + Year Metadata Filter (ChromaDB)
[ Top-K Candidate Chunks ]
   │
   ▼  Stage 2: Pairwise Relevance Scoring (Cross-Encoder MiniLM)
[ Reranked Top-N Chunks ] ──▶ Gemini LLM ──▶ Grounded Answer + PMID Citations
```

## Tech Stack

- **Frontend:** Streamlit
- **Vector Database:** ChromaDB (persistent local storage)
- **Reranking Model:** `sentence-transformers` (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **LLM:** Google Gemini (`models/gemini-3.5-flash-lite`)
- **Biomedical Data Source:** NCBI Entrez / PubMed API (`biopython`)
- **Text Processing:** `langchain-text-splitters`

## Setup & Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Copy `.env.example` to `.env` and add your credentials:
   ```bash
   cp .env.example .env
   ```
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ENTREZ_EMAIL=your_email@example.com
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```