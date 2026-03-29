"""
Streamlit Web UI for the Autonomous Market Research Aggregator.

Run with:
    streamlit run app.py
"""

import asyncio
import time
from pathlib import Path

import streamlit as st

from config import get_settings, validate_api_keys
from orchestrator.pipeline import ResearchPipeline
from mcp_server.server import save_report_to_disk, list_reports

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Research Aggregator",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 { font-size: 2.2rem; }
    .step-box {
        background: #1e1e2e;
        border-left: 4px solid #6c63ff;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #262637;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 1.8rem; color: #6c63ff; }
    .metric-card p  { margin: 0; font-size: 0.85rem; color: #aaa; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────
def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("⚙️ Settings")

    settings = get_settings()

    # Pre-fill from .env or Streamlit secrets
    default_groq = ""
    default_serper = ""
    if settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here":
        default_groq = settings.groq_api_key
    elif "GROQ_API_KEY" in st.secrets:
        default_groq = st.secrets["GROQ_API_KEY"]

    if settings.serper_api_key and settings.serper_api_key != "your_serper_api_key_here":
        default_serper = settings.serper_api_key
    elif "SERPER_API_KEY" in st.secrets:
        default_serper = st.secrets["SERPER_API_KEY"]

    groq_key = st.text_input(
        "Groq API Key",
        value=default_groq,
        type="password",
        help="Get yours at https://console.groq.com/keys",
    )
    serper_key = st.text_input(
        "Serper.dev API Key",
        value=default_serper,
        type="password",
        help="Get yours at https://serper.dev/api-key",
    )

    st.divider()
    st.subheader("Model")
    groq_model = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
    )

    st.divider()
    st.subheader("Search")
    max_queries = st.slider("Max search queries", 3, 15, settings.max_search_queries)
    results_per_query = st.slider("Results per query", 2, 10, settings.search_results_per_query)
    max_articles = st.slider("Max articles to scrape", 5, 30, settings.max_articles_to_scrape)

    st.divider()
    st.subheader("RAG")
    chunk_size = st.slider("Chunk size (chars)", 300, 3000, settings.chunk_size, step=100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, settings.chunk_overlap, step=50)
    top_k = st.slider("Top-K chunks for writer", 5, 40, settings.top_k_chunks)


# ── Main area ────────────────────────────────────────────────────
st.markdown('<div class="main-header"><h1>📊 Autonomous Market Research Aggregator</h1>'
            '<p>AI-powered deep-dive research reports on any company or industry</p></div>',
            unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────
tab_research, tab_reports, tab_about = st.tabs(["🔍 New Research", "📁 Past Reports", "ℹ️ How It Works"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 – Run new research
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_research:
    col_input, col_examples = st.columns([3, 2])

    with col_input:
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g. Electric Vehicle startups 2026",
            label_visibility="collapsed",
        )

    with col_examples:
        st.caption("Try these examples:")
        example_cols = st.columns(3)
        examples = [
            "Electric Vehicle startups 2026",
            "AI in Healthcare market",
            "Quantum Computing industry",
        ]
        for i, ex in enumerate(examples):
            if example_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
                topic = ex

    run_btn = st.button("🚀 Generate Research Report", type="primary", use_container_width=True, disabled=not topic)

    if run_btn and topic:
        # Validate keys
        if not groq_key or not serper_key:
            st.error("Please enter both API keys in the sidebar.")
            st.stop()

        # Build settings from sidebar values
        settings.groq_api_key = groq_key
        settings.serper_api_key = serper_key
        settings.groq_model = groq_model
        settings.max_search_queries = max_queries
        settings.search_results_per_query = results_per_query
        settings.max_articles_to_scrape = max_articles
        settings.chunk_size = chunk_size
        settings.chunk_overlap = chunk_overlap
        settings.top_k_chunks = top_k

        # Progress UI
        progress_bar = st.progress(0, text="Initializing pipeline…")
        status_area = st.container()

        with status_area:
            step_status = st.empty()

            # ── STEP 1: Planner ──────────────────────────────
            step_status.info("🧠 **Step 1/5 — Planner Agent:** Decomposing topic into search queries…")
            progress_bar.progress(5, text="Step 1/5 — Planning…")

            pipeline = ResearchPipeline(settings)

            start_time = time.time()

            # Run planner
            plan = run_async(pipeline.planner.create_research_plan(topic))
            queries = plan.get("search_queries", [])
            sections = plan.get("report_sections", [])

            progress_bar.progress(15, text="Step 1/5 — Planning complete")
            step_status.success(f"✅ **Planner:** Generated {len(queries)} search queries & {len(sections)} report sections")

            with st.expander("📋 Research Plan", expanded=False):
                st.write(f"**Summary:** {plan.get('topic_summary', topic)}")
                st.write("**Search Queries:**")
                for i, q in enumerate(queries, 1):
                    st.write(f"  {i}. {q}")
                st.write("**Report Sections:**")
                for s in sections:
                    st.write(f"  • **{s['title']}** — {s.get('description', '')}")

            # ── STEP 2: Search ───────────────────────────────
            step_status_2 = st.empty()
            step_status_2.info("🔎 **Step 2/5 — Search Agent:** Searching the web & scraping articles…")
            progress_bar.progress(25, text="Step 2/5 — Searching…")

            articles = run_async(pipeline.searcher.search_and_extract(queries))

            progress_bar.progress(45, text="Step 2/5 — Search complete")
            total_words = sum(a.get("word_count", 0) for a in articles)
            step_status_2.success(f"✅ **Searcher:** Extracted {len(articles)} articles ({total_words:,} words)")

            if not articles:
                st.error("No articles found. Try a broader topic.")
                st.stop()

            with st.expander("📰 Extracted Articles", expanded=False):
                for a in articles:
                    st.write(f"• **{a.get('title', 'Untitled')}** ({a.get('word_count', 0)} words) — [{a.get('url', '')}]({a.get('url', '')})")

            # ── STEP 3: RAG ──────────────────────────────────
            step_status_3 = st.empty()
            step_status_3.info("🧩 **Step 3/5 — RAG Pipeline:** Chunking & embedding into vector store…")
            progress_bar.progress(55, text="Step 3/5 — Indexing…")

            from rag.chunker import chunk_articles as do_chunk
            pipeline.vector_store.create_collection(topic)
            chunks = do_chunk(articles, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
            chunks_stored = pipeline.vector_store.add_chunks(chunks)

            progress_bar.progress(65, text="Step 3/5 — Indexing complete")
            step_status_3.success(f"✅ **RAG:** Stored {chunks_stored} chunks in ChromaDB")

            # ── STEP 4: Writer ───────────────────────────────
            step_status_4 = st.empty()
            step_status_4.info("✍️ **Step 4/5 — Writer Agent:** Synthesizing report sections via Groq…")
            progress_bar.progress(70, text="Step 4/5 — Writing…")

            report = run_async(pipeline.writer.write_report(topic, plan))

            progress_bar.progress(90, text="Step 4/5 — Report written")
            step_status_4.success(f"✅ **Writer:** Report synthesized — {len(report.split()):,} words")

            # ── STEP 5: Save ─────────────────────────────────
            progress_bar.progress(95, text="Step 5/5 — Saving…")
            file_path = save_report_to_disk(report, topic, settings.output_dir)
            total_time = round(time.time() - start_time, 1)
            progress_bar.progress(100, text="✅ Complete!")

        # ── Results ──────────────────────────────────────────
        st.divider()
        st.subheader("📈 Pipeline Results")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Search Queries", len(queries))
        m2.metric("Articles Found", len(articles))
        m3.metric("RAG Chunks", chunks_stored)
        m4.metric("Report Words", f"{len(report.split()):,}")
        m5.metric("Total Time", f"{total_time}s")

        st.divider()
        st.subheader("📄 Generated Report")
        st.markdown(report)

        st.divider()
        col_dl, col_path = st.columns([1, 2])
        with col_dl:
            st.download_button(
                "⬇️ Download Report (.md)",
                data=report,
                file_name=Path(file_path).name,
                mime="text/markdown",
                use_container_width=True,
            )
        with col_path:
            st.info(f"Also saved to: `{file_path}`")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 – Past reports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_reports:
    reports = list_reports(settings.output_dir)

    if not reports:
        st.info("No reports generated yet. Run a research query first!")
    else:
        st.write(f"**{len(reports)} report(s) found**")
        for r in reports:
            with st.expander(f"📄 {r['filename']}  ({r['size_kb']} KB)"):
                fpath = Path(r["path"])
                content = fpath.read_text(encoding="utf-8")
                st.markdown(content)
                st.download_button(
                    "⬇️ Download",
                    data=content,
                    file_name=r["filename"],
                    mime="text/markdown",
                    key=f"dl_{r['filename']}",
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 – How it works
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_about:
    st.markdown("""
### 🏗️ Architecture

```
You enter a topic
        │
        ▼
┌─────────────────────┐
│  1. PLANNER AGENT   │  Groq LLM → search queries + report outline
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  2. SEARCH AGENT    │  Serper.dev API → web + news search → article scraping
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  3. RAG PIPELINE    │  Chunking → sentence-transformers embeddings → ChromaDB
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  4. WRITER AGENT    │  RAG retrieval → Groq LLM → structured markdown report
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  5. SAVE            │  Markdown file → reports/ folder + download button
└─────────────────────┘
```

### 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Groq API (llama-3.3-70b-versatile) |
| **Search** | Serper.dev (Google Search + News) |
| **Scraping** | trafilatura + BeautifulSoup4 |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Store** | ChromaDB (in-memory) |
| **Web UI** | Streamlit |
| **CLI** | Typer + Rich |

### 🔑 API Keys Needed

1. **Groq** — Free at [console.groq.com/keys](https://console.groq.com/keys)
2. **Serper.dev** — Free 2,500 searches at [serper.dev/api-key](https://serper.dev/api-key)
""")
