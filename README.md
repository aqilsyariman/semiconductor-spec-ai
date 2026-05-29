# Semiconductor AI Intelligence Platform

A production-grade RAG (Retrieval-Augmented Generation) pipeline for querying semiconductor technical documents. Upload any datasheet or product brief and get precise AI-powered answers with exact source and page citation.

🔗 **[Live Demo](https://semiconductor-ai-intelligence.streamlit.app/)** | **[GitHub](https://github.com/aqilsyariman/semiconductor-spec-ai)**

## What It Does

Upload Intel, Micron, or any semiconductor technical document and ask questions in natural language. The system retrieves the most relevant information and returns a precise answer — citing the exact document and page number.

**Example:**
```
Question: What client-optimized LLMs does Intel Core Ultra support?

Answer: Intel Core Ultra supports the following client-optimized LLMs:
• Microsoft Phi-2
• LLaMa2-7B / LLaMa3-8B
• Mistral-7B
• Qwen 7B

Source: Product-Brief-Intel-Core-Ultra-200S.pdf, Page 5
```

## How It Works

```
PDF Upload → Text Extraction → Chunking → Embedding → ChromaDB
     ↓
User Question → Vector Search → Re-ranking → LLaMA 3.3 70B → Answer + Citation
```

1. Extracts text from every page with page number tracking
2. Splits text into 1000-character chunks with metadata (source + page)
3. Converts chunks to vectors using Sentence Transformers
4. Stores vectors and metadata in ChromaDB
5. Converts user question to vector and searches ChromaDB
6. Re-ranks results using FlashRank for higher relevance accuracy
7. Sends top chunks to LLaMA 3.3 70B with semiconductor expert prompt
8. Returns answer with exact source document and page citation

## Key Features

- Multi-document upload — query across multiple PDFs simultaneously
- Source citation — every answer cites the document and page number
- Re-ranking pipeline — FlashRank re-scores retrieved chunks for higher accuracy
- Conversation memory — remembers previous questions in the session
- Sidebar UI — clean enterprise-grade chat interface
- Hallucination prevention — refuses to answer if relevant context is not found
- Free to run — Groq free tier, no credit card required

## Tech Stack

- **Python** — Core language
- **PyPDF2** — PDF text extraction with page tracking
- **Sentence Transformers** — Text to vector embedding (all-MiniLM-L6-v2)
- **ChromaDB** — Vector database with metadata storage
- **FlashRank** — Re-ranking for improved retrieval accuracy
- **LangChain** — LLM integration
- **Groq API** — Free LLaMA 3.3 70B inference
- **Streamlit** — Chat UI with sidebar layout
- **Streamlit Cloud** — Free cloud deployment

## Why Not Just Use ChatGPT?

ChatGPT sends your documents to OpenAI's servers — not acceptable for confidential semiconductor datasheets. This pipeline keeps everything in-house, returns exact page citations, and costs nothing at scale.

## Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/aqilsyariman/semiconductor-spec-ai.git
cd semiconductor-spec-ai
```

### Step 2 — Install dependencies
```bash
pip3 install PyPDF2 sentence-transformers chromadb langchain langchain-groq python-dotenv streamlit flashrank pycryptodome
```

### Step 3 — Get a free Groq API key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free — no credit card needed
3. Create an API key

### Step 4 — Set your API key
Create a `.env` file:
```
GROQ_API_KEY=your-api-key-here
```

### Step 5 — Run
```bash
streamlit run streamlit_app.py
```

## Usage

1. Upload one or more semiconductor PDF datasheets in the sidebar
2. Click **Process Documents**
3. Type your technical question in the chat input
4. Get a precise answer with source and page citation

## Project Structure

```
semiconductor-spec-ai/
├── app.py               # RAG pipeline — ingestion, search, reranking, generation
├── streamlit_app.py     # Chat UI with sidebar layout
├── requirements.txt     # Dependencies
├── .env                 # API key (not committed)
└── README.md
```

## Known Limitations

- Works best with documents covering different topics — overlapping documents may cause chunk imbalance (solvable with advanced reranking)
- PyPDF2 extracts text only — answers embedded in images or graphics cannot be retrieved
- Encrypted PDFs require pycryptodome to read

## Built By

Aqil Syariman
