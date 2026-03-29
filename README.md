# Autonomous Market Research Aggregator

An agentic AI system that compiles deep-dive research reports on any company or industry. Input a topic and receive a comprehensive, source-cited markdown report — fully automated.

> **Two interfaces:** Streamlit web app (`app.py`) and CLI (`main.py`). No web hosting required — everything runs locally.

## Architecture Overview

```
USER INPUT: "Electric Vehicle startups 2026"
                    │
                    ▼
    ┌────────────────────────────┐
    │   1. PLANNER AGENT         │  Groq LLM (llama-3.3-70b)
    │   Decomposes topic into    │  → Search queries
    │   search queries & report  │  → Report section outline
    │   structure                │  → Research dimensions
    └─────────────┬──────────────┘
                  ▼
    ┌────────────────────────────┐
    │   2. SEARCH AGENT          │  Serper.dev API
    │   Loops through queries    │  → Google web search
    │   to find articles, news,  │  → Google news search
    │   press releases           │  → Article scraping
    └─────────────┬──────────────┘  (trafilatura + BeautifulSoup)
                  ▼
    ┌────────────────────────────┐
    │   3. RAG PIPELINE          │  sentence-transformers
    │   Chunks text → embeds →   │  → all-MiniLM-L6-v2 embeddings
    │   stores in vector DB      │  → ChromaDB vector store
    │   for semantic retrieval   │  → Overlapping chunk strategy
    └─────────────┬──────────────┘
                  ▼
    ┌────────────────────────────┐
    │   4. WRITER AGENT          │  Groq LLM (llama-3.3-70b)
    │   Retrieves relevant       │  → RAG multi-query retrieval
    │   chunks per section →     │  → Section-by-section writing
    │   synthesizes full report  │  → Executive summary generation
    └─────────────┬──────────────┘
                  ▼
    ┌────────────────────────────┐
    │   5. MCP SERVER            │  Model Context Protocol
    │   Saves markdown report    │  → stdio transport
    │   to local documents       │  → VS Code / Claude Desktop
    └────────────────────────────┘
```

## Tech Stack & Values at Each Step

### Step 1: Planner Agent
| Component | Technology | Value/Purpose |
|-----------|-----------|---------------|
| LLM | Groq API (`llama-3.3-70b-versatile`) | Fast inference for structured planning |
| Temperature | `0.3` | Low randomness for consistent plans |
| Output Format | `json_object` | Forced JSON for reliable parsing |
| Max Queries | `8` (configurable) | Limits API cost while ensuring coverage |
| Output | `search_queries[]`, `report_sections[]`, `research_dimensions[]` | Structured plan for downstream agents |

### Step 2: Search Agent
| Component | Technology | Value/Purpose |
|-----------|-----------|---------------|
| Search API | Serper.dev (`/search` + `/news`) | Google search results via API |
| Results per query | `5` (configurable) | Balance between coverage and cost |
| Web Scraper | trafilatura + BeautifulSoup fallback | Robust article text extraction |
| Concurrency | `asyncio.Semaphore(5)` | 5 parallel article scrapes |
| Deduplication | URL-based dedup + domain filtering | Removes social media, duplicates |
| Rate Limiting | `0.3s` delay between search queries | Respects Serper.dev rate limits |
| Output | `articles[]` with `{url, title, text, word_count}` | Clean article data for RAG |

### Step 3: RAG Pipeline
| Component | Technology | Value/Purpose |
|-----------|-----------|---------------|
| Chunking | Custom paragraph-aware splitter | Preserves semantic boundaries |
| Chunk Size | `1000` characters | Optimal for embedding quality |
| Chunk Overlap | `200` characters | Maintains context across chunks |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Fast, high-quality 384-dim embeddings |
| Vector Store | ChromaDB (in-memory) | Zero-config vector database |
| Metadata | `{source_url, source_title, chunk_index}` | Citation tracking per chunk |
| Output | Indexed vector collection ready for queries | Semantic search over all content |

### Step 4: Writer Agent
| Component | Technology | Value/Purpose |
|-----------|-----------|---------------|
| LLM | Groq API (`llama-3.3-70b-versatile`) | Fast, high-quality text synthesis |
| Retrieval | Multi-query RAG (`top_k=20`) | Diverse, relevant context per section |
| Temperature | `0.4` (sections), `0.3` (summary) | Balanced creativity vs accuracy |
| Max Tokens | `1500` per section, `1000` for summary | Detailed but focused sections |
| Sections | Written sequentially with RAG context | Each section gets targeted retrieval |
| Executive Summary | Generated from all sections combined | Coherent top-level overview |
| Citations | `[Source: title]` format in text | Traceable claims |
| Output | Complete markdown report with YAML frontmatter | Publication-ready format |

### Step 5: MCP Server
| Component | Technology | Value/Purpose |
|-----------|-----------|---------------|
| Protocol | MCP (Model Context Protocol) | Standard LLM tool integration |
| Transport | stdio | Works with VS Code, Claude Desktop |
| Tools | `save_report`, `list_reports`, `read_report` | Full report lifecycle management |
| Security | Path traversal prevention | Safe file operations |
| Output | `.md` file in `./reports/` directory | Local file system storage |

## Project Structure

```
market research aggregator/
├── .env.example          # API keys template (copy to .env)
├── .env                  # Your actual API keys (git-ignored)
├── .gitignore            # Git ignore rules
├── requirements.txt      # Pinned Python dependencies (22 packages)
├── README.md             # This file
├── app.py                # Streamlit web UI
├── main.py               # CLI entry point (Typer)
├── config.py             # Pydantic settings loader (.env → typed config)
├── agents/
│   ├── __init__.py
│   ├── planner.py        # Planner Agent (topic → search queries + outline)
│   ├── searcher.py       # Search Agent (Serper.dev → scraped articles)
│   └── writer.py         # Writer Agent (RAG retrieval + Groq → report)
├── rag/
│   ├── __init__.py
│   ├── chunker.py        # Paragraph-aware text chunking with overlap
│   └── vector_store.py   # ChromaDB vector storage + semantic retrieval
├── orchestrator/
│   ├── __init__.py
│   └── pipeline.py       # 5-step agent coordination pipeline
├── mcp_server/
│   ├── __init__.py
│   └── server.py         # MCP server (save/list/read reports via stdio)
├── utils/
│   ├── __init__.py
│   ├── logger.py          # Rich colored logging per agent
│   └── web_scraper.py     # trafilatura + BS4 article extraction
└── reports/              # Generated reports saved here (git-ignored)
```

## Setup & Installation

### 1. Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.11 or higher |
| **Groq API key** | Free tier — [console.groq.com/keys](https://console.groq.com/keys) |
| **Serper.dev API key** | Free 2,500 searches — [serper.dev/api-key](https://serper.dev/api-key) |
| **Disk space** | ~500 MB (embedding model is downloaded on first run) |
| **OS** | Windows, macOS, or Linux |

### 2. Clone the Repository
```bash
git clone https://github.com/Vibhaaaaraooo/Autonomous-Market-Research-Aggregator.git
cd Autonomous-Market-Research-Aggregator
```

### 3. Create a Virtual Environment (recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs 22 packages:

| Package | Version | Purpose |
|---|---|---|
| `groq` | 1.1.2 | Groq LLM API client (Planner + Writer agents) |
| `httpx` | 0.28.1 | Async HTTP client for Serper.dev API & scraping |
| `python-dotenv` | 1.2.2 | Loads `.env` file into environment variables |
| `pydantic` | 2.12.5 | Data validation & typed models |
| `pydantic-settings` | 2.13.1 | BaseSettings class — auto-reads `.env` |
| `chromadb` | 1.5.5 | In-memory vector database for RAG retrieval |
| `sentence-transformers` | 5.3.0 | Embedding model (`all-MiniLM-L6-v2`, 384-dim) |
| `tiktoken` | 0.12.0 | BPE tokenizer for token-aware chunking |
| `beautifulsoup4` | 4.14.3 | HTML parser — fallback article extraction |
| `trafilatura` | 2.0.0 | Primary web article body extractor |
| `newspaper3k` | 0.2.8 | Article metadata extraction fallback |
| `mcp` | 1.26.0 | Model Context Protocol SDK (tool server) |
| `uvicorn` | 0.42.0 | ASGI server for MCP HTTP transport |
| `starlette` | 1.0.0 | ASGI framework (MCP dependency) |
| `streamlit` | 1.55.0 | Web UI for the app (`app.py`) |
| `rich` | 14.3.3 | Colored, structured terminal output |
| `typer` | 0.24.1 | CLI framework for `main.py` commands |
| `tenacity` | 9.1.4 | Retry decorator for flaky API calls |
| `aiofiles` | 25.1.0 | Async file I/O |

### 5. Configure API Keys
```bash
# Copy the template
copy .env.example .env        # Windows
cp .env.example .env           # macOS / Linux
```

Then edit `.env` and paste your keys:
```env
GROQ_API_KEY=gsk_your_key_here
SERPER_API_KEY=your_key_here
```

### 6. Run the App

**Option A — Streamlit Web UI (recommended):**
```bash
streamlit run app.py
```
Opens at http://localhost:8501 with a full visual interface.

**Option B — Command-line:**
```bash
python main.py research "Electric Vehicle startups 2026"
```

---

## Usage

### Streamlit Web App
```bash
streamlit run app.py
```

Features:
- **Sidebar:** API key inputs, model selector, search/RAG tuning sliders
- **New Research tab:** Enter topic → watch live 5-step progress → view rendered report → download
- **Past Reports tab:** Browse & download all previously generated reports
- **How It Works tab:** Architecture diagram & tech stack reference

### CLI Commands

```bash
# Generate a report
python main.py research "AI in Healthcare 2026"

# Custom output directory
python main.py research "Quantum Computing market" --output ./my-reports

# List previous reports
python main.py list-reports

# Start MCP server (for VS Code / Claude Desktop integration)
python main.py mcp-server
```

### MCP Configuration for VS Code / Claude Desktop
Add to your MCP settings:
```json
{
    "mcpServers": {
        "market-research": {
            "command": "python",
            "args": ["main.py", "mcp-server"],
            "cwd": "path/to/Autonomous-Market-Research-Aggregator"
        }
    }
}
```

## Configuration Reference

All settings are in `.env` with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | *required* | Groq API key for LLM calls |
| `SERPER_API_KEY` | *required* | Serper.dev API key for web search |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model ID |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `MAX_SEARCH_QUERIES` | `8` | Max queries planner generates |
| `SEARCH_RESULTS_PER_QUERY` | `5` | Results per Serper.dev query |
| `MAX_ARTICLES_TO_SCRAPE` | `15` | Max articles to scrape |
| `CHUNK_SIZE` | `1000` | Characters per text chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_CHUNKS` | `20` | Chunks retrieved for writing |
| `OUTPUT_DIR` | `./reports` | Report output directory |
| `MCP_SERVER_PORT` | `3001` | MCP server port |

## How the Agents Communicate

```
┌──────────┐    search_queries[]     ┌──────────┐
│  PLANNER ├────────────────────────►│ SEARCHER │
│  AGENT   │    report_sections[]    │  AGENT   │
└──────────┘                         └────┬─────┘
                                          │ articles[]
                                          ▼
                                    ┌──────────┐
                                    │   RAG    │
                                    │ PIPELINE │
                                    └────┬─────┘
                                         │ vector_store (ChromaDB)
                                         ▼
┌──────────┐    retrieval_queries    ┌──────────┐
│  WRITER  │◄───────────────────────│  VECTOR  │
│  AGENT   │    relevant_chunks[]   │  STORE   │
└────┬─────┘                        └──────────┘
     │ report.md
     ▼
┌──────────┐
│   MCP    │
│  SERVER  │ → saves to ./reports/
└──────────┘
```

## Sample Output

Running `python main.py research "Electric Vehicle startups 2026"` produces:
- A markdown file like `Electric_Vehicle_startups_2026_20260329_143022.md`
- Contains: Executive Summary, Market Landscape, Key Players, Funding & Investment, Technology Trends, Regulatory Environment, Outlook & Predictions
- Each section cites sources with `[Source: article title]`
- Full source list at the bottom with clickable links

## Troubleshooting

| Problem | Solution |
|---|---|
| `Import could not be resolved` | Run `pip install -r requirements.txt` inside the venv |
| `Missing API keys` error | Check `.env` has real keys, not the placeholder values |
| `No articles found` | Try a broader topic; Serper key may be expired |
| Embedding model download slow | First run downloads ~90 MB model; subsequent runs use cache |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| `KeyError` in Planner | Already fixed — JSON braces escaped in prompt template |

## License

MIT
