# Agentic RAG System v1 — Multilingual + Local LLM + Reasoning

A production-ready, modular Retrieval-Augmented Generation system with:
- **Local LLM support** via [Ollama](https://ollama.com) (fully offline)
- **Arabic document support** with automatic encoding detection and normalization
- **Transparent reasoning** with step-by-step chain-of-thought traces
- Hybrid retrieval (semantic + BM25), cross-encoder reranking, and knowledge graph integration

---


| Feature | Details |
|---|---|
|  **Ollama local LLMs** | Run llama3, mistral, deepseek-r1, phi3, qwen2 — no API key needed |
|  **Reasoning chain** | Every answer includes CoT steps, confidence score, and plain-language explanation |
|  **Arabic support** | Auto-detects encoding (UTF-8, Windows-1256, ISO-8859-6), normalizes tashkeel/alef variants |
|  **Multi-provider LLM** | Switch between Ollama / Groq / OpenAI with a single env variable |
|  **Multilingual embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` supports 50+ languages including Arabic |
|  **Arabic PDF/DOCX** | Dedicated loaders for Arabic PDFs and Word documents |

---

## Architecture

```
agentic_rag/
├── data/
│   ├── raw/               # Place source documents here (.txt, .pdf, .csv, .docx)
│   ├── processed/         # Reserved for pre-processed artefacts
│   └── chroma_db/         # Auto-created: ChromaDB persistent storage
├── src/
│   ├── config.py          # Pydantic settings loaded from .env
│   ├── logging_config.py  # Centralized logging setup
│   ├── llm_factory.py     # Multi-provider LLM: Ollama / Groq / OpenAI
│   ├── arabic_utils.py    # Arabic encoding detection, normalization, loaders
│   ├── reasoning_chain.py # Chain-of-thought reasoning + explanation generation
│   ├── ingest.py          # Load → chunk → embed → index (Arabic-aware)
│   ├── retriever.py       # Hybrid retrieval, RRF fusion, reranking, compression
│   ├── tools.py           # 5 agent tools (RAG, web, Wikipedia, calc, graph)
│   └── agent.py           # ReAct agent with reasoning integration
├── api/
│   └── app.py             # FastAPI REST interface (+ Ollama health check endpoint)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up local Ollama (recommended, no API key needed)

```bash
# Install Ollama: https://ollama.com/download
ollama serve                        # Start the Ollama server

# Pull models (choose based on your hardware):
ollama pull llama3                  # General Q&A (8B, ~5GB)
ollama pull deepseek-r1:8b          # Reasoning-optimized (~5GB)
ollama pull qwen2:7b                # Great for Arabic + multilingual
# OR for lower RAM:
ollama pull phi3:mini               # 3.8B, very fast
ollama pull deepseek-r1:1.5b        # Tiny reasoning model
```

### 3. Configure environment

```bash
cp .env.example .env
# Minimum config for Ollama (no API keys needed!):
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3
# OLLAMA_REASONING_MODEL=deepseek-r1:8b
```

### 4. Ingest documents (Arabic or English)

Drop your `.txt`, `.pdf`, `.docx`, or `.csv` files into `data/raw/`, then:

```bash
python -c "
from src.logging_config import setup_logging; setup_logging()
from src.ingest import run_ingestion_pipeline
run_ingestion_pipeline('data/raw')
"
```

Arabic files are automatically detected and processed with the correct encoding.

### 5. Query the agent (Python)

```python
from src.logging_config import setup_logging; setup_logging()
from src.agent import run_agent

# English query
answer = run_agent("What are the main themes in the documents?")
print(answer)

# Arabic query — auto-detected, reasoning in Arabic
answer = run_agent("ما هي أهم النقاط في الوثائق؟")
print(answer)

# Force reasoning off for a quick raw answer
answer = run_agent("summarize the documents", with_reasoning=False)
print(answer)
```

### 6. Run the REST API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Example requests:

```bash
# English query with reasoning
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What topics are covered in the knowledge base?", "with_reasoning": true}'

# Arabic query (auto-detected, reasoning in Arabic)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي المعلومات المتاحة في قاعدة البيانات؟"}'

# Check Ollama server status
curl http://localhost:8000/ollama/check

# View current config
curl http://localhost:8000/config

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"data_dir": "./data/raw"}'
```

---

## Ollama Model Recommendations

| Use Case | Model | RAM Needed |
|---|---|---|
| General Q&A (English) | `llama3` | ~5 GB |
| General Q&A (Arabic/Multilingual) | `qwen2:7b` | ~5 GB |
| Reasoning + Explanation | `deepseek-r1:8b` | ~5 GB |
| Low memory / fast | `phi3:mini` | ~2 GB |
| Tiny reasoning | `deepseek-r1:1.5b` | ~1 GB |
| High quality | `llama3:70b` (if you have 40+ GB) | ~40 GB |

Set in `.env`:
```
OLLAMA_MODEL=qwen2:7b
OLLAMA_REASONING_MODEL=deepseek-r1:8b
```

---

## Arabic Support Details

The system handles Arabic documents across the full pipeline:

**Encoding detection** (`arabic_utils.py`):
- Tries UTF-8 → Windows-1256 → ISO-8859-6 → CP1256 in order
- Handles mixed Arabic/English documents

**Text normalization** (when `NORMALIZE_ARABIC=true`):
- Removes tashkeel (diacritics: ّ ً ٌ ٍ َ ُ ِ ْ)
- Normalizes alef variants (أ إ آ → ا)
- Normalizes alef maqsura (ى → ي)

**Document loading**:
- `.txt` → encoding-aware TextLoader
- `.pdf` → pypdf with per-page normalization
- `.docx` → python-docx preserving paragraph order

**Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages)

**Reasoning**: When an Arabic question is detected, the chain-of-thought prompt, reasoning steps, answer, and explanation are all generated in Arabic.

---

## Reasoning Output Example

```
────────────────────────────────────────────────────────────
 Reasoning Steps:
  1. Analyze what the question is asking about the documents.
  2. Review retrieved context for relevant sections.
  3. Identify key themes: machine learning, RAG systems, and NLP.
  4. Synthesize a comprehensive answer from the evidence.

 Answer:
The documents cover three main areas: retrieval-augmented generation 
architectures, Arabic NLP preprocessing techniques, and agent-based 
reasoning systems using local LLMs.

 Explanation:
RAG is a technique that combines document search with language model 
generation to produce grounded answers. The documents describe how 
to build such systems in a modular, production-ready way.

 Confidence: 87%

 Sources Referenced:
  • "Retrieval-Augmented Generation connects a knowledge base to an LLM..."
  • "Arabic text normalization involves removing tashkeel and standardizing..."
────────────────────────────────────────────────────────────
```

---

## Environment Variables Reference

See `.env.example` for the full list. **Minimum required for Ollama** (no cloud keys):

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
OLLAMA_REASONING_MODEL=deepseek-r1:8b
```

---

## Suggested Further Improvements

1. **Streaming responses**: Add SSE/WebSocket endpoints for real-time token streaming from Ollama.
2. **Conversation memory**: Add `ConversationBufferWindowMemory` for multi-turn Arabic dialogues.
3. **camel-tools integration**: Uncomment the `camel-tools` dependency for morphological analysis of Arabic text.
4. **Arabic OCR**: Integrate `pytesseract` with an Arabic language pack for scanned Arabic PDF support.
5. **LLM-based query routing**: Replace the keyword heuristic in `retriever.py` with an LLM classifier.
