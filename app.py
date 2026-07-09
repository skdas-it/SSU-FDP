# =============================================================================
# ResearchMind AI – Intelligent Research Companion
# =============================================================================
# Built with:
#   - Python + Flask
#   - IBM watsonx.ai Studio
#   - IBM Granite Models (ibm/granite-13b-instruct-v2)
#   - Lightweight RAG System (in-memory, no database)
#   - Agentic AI Architecture (5 specialized agents + orchestrator)
#
# Agents:
#   1. Research Retrieval Agent   – Extract and organize research knowledge
#   2. Literature Review Agent    – Generate structured literature reviews
#   3. Research Gap Analysis Agent– Identify unexplored research opportunities
#   4. Trend Forecasting Agent    – Predict future research directions
#   5. Research Advisor Agent     – Provide strategic research guidance
#   6. Master Orchestrator Agent  – Coordinate all agents
#
# Suitable for: IBM SkillsBuild, hackathons, academic projects, AI showcases
# =============================================================================

import os
import re
import json
import math
import textwrap
import hashlib
from datetime import datetime
from io import BytesIO

from flask import Flask, request, jsonify, render_template_string, session
from dotenv import load_dotenv

# PDF text extraction (PyMuPDF or pdfminer fallback)
try:
    import fitz  # PyMuPDF
    PDF_BACKEND = "pymupdf"
except ImportError:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        PDF_BACKEND = "pdfminer"
    except ImportError:
        PDF_BACKEND = "none"

# IBM watsonx.ai SDK
try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
    WATSONX_AVAILABLE = True
except ImportError:
    WATSONX_AVAILABLE = False

# =============================================================================
# Load environment variables from .env
# =============================================================================
load_dotenv()

WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
SECRET_KEY         = os.getenv("SECRET_KEY", "researchmind-secret-2025")

# =============================================================================
# Flask Application Setup
# =============================================================================
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# =============================================================================
# In-Memory RAG Knowledge Store
# Each document is stored as a list of chunks with simple TF-IDF style scoring
# No external vector database required.
# =============================================================================
rag_store = {
    "documents": [],   # list of {id, title, content, chunks}
    "chunks":    [],   # flat list of {doc_id, title, chunk_text, chunk_id}
}

# =============================================================================
# IBM watsonx.ai – Model Initialization
# This block initializes the IBM Granite model for all agents.
# =============================================================================
def get_watsonx_model():
    """
    Initialize and return an IBM watsonx.ai ModelInference instance.
    Uses IBM Granite 13B Instruct model as the primary reasoning engine.
    Returns None when credentials are missing (demo/fallback mode activates).
    """
    if not WATSONX_AVAILABLE:
        return None
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return None
    try:
        credentials = Credentials(
            url=WATSONX_URL,
            api_key=WATSONX_API_KEY,
        )
        model = ModelInference(
            model_id="ibm/granite-13b-instruct-v2",   # IBM Granite Model
            credentials=credentials,
            project_id=WATSONX_PROJECT_ID,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS:  1200,
                GenParams.MIN_NEW_TOKENS:  50,
                GenParams.STOP_SEQUENCES:  ["<|endoftext|>"],
                GenParams.REPETITION_PENALTY: 1.1,
            },
        )
        return model
    except Exception as e:
        app.logger.warning(f"[watsonx.ai] Model init failed: {e}")
        return None


# =============================================================================
# Core IBM watsonx.ai Helper
# =============================================================================
def generate_response(prompt: str, model=None) -> str:
    """
    Send a prompt to IBM Granite via watsonx.ai and return the generated text.
    Falls back to a structured demo response when watsonx.ai is unavailable.

    IBM watsonx.ai Integration Point ← PRIMARY ENTRY POINT
    """
    if model is None:
        model = get_watsonx_model()

    if model:
        try:
            result = model.generate_text(prompt=prompt)
            return result.strip() if isinstance(result, str) else str(result).strip()
        except Exception as e:
            app.logger.warning(f"[watsonx.ai] Generation error: {e}")

    # ── Demo / Fallback Mode ──────────────────────────────────────────────────
    # Returns structured placeholder content when API is unavailable.
    return _demo_fallback(prompt)


def _demo_fallback(prompt: str) -> str:
    """
    Intelligent demo fallback that returns contextually appropriate content
    based on the detected agent type in the prompt.
    """
    lower = prompt.lower()

    if "literature review" in lower:
        return (
            "**Literature Review Summary**\n\n"
            "The selected body of work spans foundational and recent advances across the research domain. "
            "Key themes include methodology innovation, scalability concerns, and real-world applicability. "
            "Authors consistently highlight the trade-off between model complexity and interpretability. "
            "Early studies established baseline frameworks (Smith et al., 2018; Lee et al., 2019), while "
            "recent work challenges those assumptions with empirical evidence from large-scale experiments. "
            "\n\n**Key Contributions:**\n"
            "- Systematic benchmarking across diverse datasets\n"
            "- Novel hybrid architectures combining symbolic and neural approaches\n"
            "- Cross-domain transfer learning demonstrations\n\n"
            "**Research Landscape:** The field is maturing from proof-of-concept to production-grade systems, "
            "with growing emphasis on reproducibility and open benchmarks."
        )

    if "research gap" in lower or "gap analysis" in lower:
        return (
            "**Research Gap Analysis**\n\n"
            "**Identified Gaps:**\n"
            "1. Limited longitudinal studies examining long-term performance degradation\n"
            "2. Insufficient attention to low-resource and multilingual settings\n"
            "3. Lack of standardized evaluation protocols across competing methods\n"
            "4. Under-explored intersection with privacy-preserving techniques\n"
            "5. Few studies addressing energy efficiency and carbon footprint\n\n"
            "**Novel Opportunities:**\n"
            "- Federated approaches for sensitive domain data\n"
            "- Neuro-symbolic integration for explainability\n"
            "- Benchmarks for real-time deployment scenarios\n\n"
            "**Improvement Suggestions:**\n"
            "Future work should prioritize reproducible experimental pipelines and "
            "community-shared datasets to accelerate peer validation."
        )

    if "trend" in lower or "forecast" in lower:
        return (
            "**Emerging Research Trends**\n\n"
            "**High-Growth Areas (2025–2028):**\n"
            "1. Multimodal foundation models – integrating text, vision, audio\n"
            "2. AI-augmented scientific discovery in drug design and materials science\n"
            "3. Sustainable and green AI – energy-aware training and inference\n"
            "4. Agentic AI systems with tool-use and long-horizon reasoning\n"
            "5. Quantum-classical hybrid algorithms for optimization\n\n"
            "**Technological Catalysts:**\n"
            "- Advances in neuromorphic hardware reducing inference costs\n"
            "- Open-source model ecosystems democratizing research access\n"
            "- Regulatory frameworks driving responsible AI adoption\n\n"
            "**Growth Potential:** The convergence of AI with domain sciences "
            "(biomedicine, climate, education) is projected to define the next decade of publications."
        )

    if "research plan" in lower or "advisor" in lower or "methodology" in lower:
        return (
            "**Strategic Research Plan**\n\n"
            "**Suggested Research Questions:**\n"
            "1. How do architectural choices affect generalization in low-data regimes?\n"
            "2. Can self-supervised pre-training replace labeled data in specialized domains?\n"
            "3. What evaluation metrics best capture real-world utility?\n\n"
            "**Recommended Methodologies:**\n"
            "- Controlled ablation studies with statistical significance testing\n"
            "- Meta-learning frameworks for rapid domain adaptation\n"
            "- Mixed-methods combining quantitative benchmarks and qualitative expert review\n\n"
            "**Dataset Recommendations:** HuggingFace Hub, UCI Repository, domain-specific corpora\n\n"
            "**Publication Targets:** NeurIPS, ICML, ACL, IEEE TPAMI, Nature Machine Intelligence\n\n"
            "**Thesis/Project Ideas:**\n"
            "- 'Efficient Fine-tuning of Large Language Models for Scientific Literature Mining'\n"
            "- 'Explainable AI for Clinical Decision Support Systems'"
        )

    # Default: research retrieval / summary
    return (
        "**Research Summary**\n\n"
        "The provided content covers a significant research area with broad academic and industrial relevance. "
        "Core findings indicate progress in automating previously manual workflows, improving accuracy over "
        "prior baselines, and reducing computational overhead. Several papers converge on the importance of "
        "data quality over quantity, and advocate for rigorous ablation studies.\n\n"
        "**Key Findings:**\n"
        "- State-of-the-art performance on standard benchmarks (↑ 8–15% over prior work)\n"
        "- Transfer learning significantly reduces labeled data requirements\n"
        "- Interpretability remains an open challenge across all reviewed methods\n\n"
        "**Important References:**\n"
        "- Vaswani et al. (2017) – Attention Is All You Need\n"
        "- Brown et al. (2020) – Language Models as Few-Shot Learners\n"
        "- LeCun et al. (2015) – Deep Learning (Nature)"
    )


# =============================================================================
# RAG System – Text Processing Utilities
# =============================================================================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file using available backend."""
    if PDF_BACKEND == "pymupdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif PDF_BACKEND == "pdfminer":
        return pdfminer_extract(BytesIO(file_bytes))
    else:
        return "[PDF extraction unavailable – install PyMuPDF: pip install pymupdf]"


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks for RAG retrieval.
    chunk_size: approximate characters per chunk
    overlap:    character overlap between consecutive chunks
    """
    words = text.split()
    chunks, current, count = [], [], 0
    for word in words:
        current.append(word)
        count += len(word) + 1
        if count >= chunk_size:
            chunks.append(" ".join(current))
            # keep overlap words for next chunk
            overlap_words = current[-(overlap // 6):]
            current = overlap_words
            count = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if len(c.strip()) > 30]


def simple_tfidf_score(query: str, chunk: str) -> float:
    """
    Compute a lightweight relevance score between query and chunk
    using term frequency overlap (no external libraries required).
    """
    q_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
    c_terms  = re.findall(r'\b\w{3,}\b', chunk.lower())
    if not c_terms or not q_terms:
        return 0.0
    c_freq = {}
    for t in c_terms:
        c_freq[t] = c_freq.get(t, 0) + 1
    score = sum(math.log(1 + c_freq.get(t, 0)) for t in q_terms)
    return score / (math.log(1 + len(c_terms)) + 1e-9)


def retrieve_research(query: str, top_k: int = 5) -> str:
    """
    RAG Retrieval – find the most relevant chunks from uploaded documents.
    Returns a formatted context string for downstream agent prompts.

    IBM watsonx.ai RAG Integration Point ← CONTEXT PROVIDER
    """
    if not rag_store["chunks"]:
        return "No documents uploaded yet. Using general knowledge."

    scored = []
    for chunk in rag_store["chunks"]:
        score = simple_tfidf_score(query, chunk["chunk_text"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:top_k]

    if not top_chunks or top_chunks[0][0] < 0.01:
        return "No highly relevant passages found. Using general knowledge."

    parts = []
    for rank, (score, chunk) in enumerate(top_chunks, 1):
        parts.append(
            f"[Source {rank}: {chunk['title']}]\n{chunk['chunk_text']}"
        )
    return "\n\n---\n\n".join(parts)


def add_document_to_rag(title: str, content: str) -> dict:
    """
    Add a document to the in-memory RAG store.
    Chunks the text and indexes all chunks for retrieval.
    """
    doc_id = hashlib.md5(f"{title}{content[:100]}".encode()).hexdigest()[:8]

    # Avoid duplicate documents
    existing_ids = {d["id"] for d in rag_store["documents"]}
    if doc_id in existing_ids:
        return {"doc_id": doc_id, "chunks": 0, "duplicate": True}

    chunks = chunk_text(content)
    rag_store["documents"].append({
        "id": doc_id,
        "title": title,
        "content": content,
        "chunk_count": len(chunks),
        "uploaded_at": datetime.now().isoformat(),
    })
    for i, chunk_text_str in enumerate(chunks):
        rag_store["chunks"].append({
            "doc_id":     doc_id,
            "title":      title,
            "chunk_text": chunk_text_str,
            "chunk_id":   f"{doc_id}_{i}",
        })
    return {"doc_id": doc_id, "chunks": len(chunks), "duplicate": False}


# =============================================================================
# AGENT 1: Research Retrieval Agent
# =============================================================================
def retrieval_agent(query: str, model=None) -> dict:
    """
    Agent 1 – Research Retrieval Agent
    Retrieves relevant knowledge from the RAG store and produces a structured
    research summary using IBM Granite via watsonx.ai.

    IBM watsonx.ai Integration Point ← AGENT 1
    """
    context = retrieve_research(query, top_k=5)

    prompt = textwrap.dedent(f"""
    You are a Research Retrieval Agent powered by IBM Granite AI.
    Your role is to extract and organize the most important research knowledge.

    Research Query: {query}

    Retrieved Context from Knowledge Base:
    {context}

    Task:
    1. Provide a concise Research Summary (3-5 sentences)
    2. List 5 Key Findings from the retrieved content
    3. List 3-5 Important References or source mentions
    4. Rate the relevance of retrieved content (High / Medium / Low)

    Format your response clearly with headers.
    """).strip()

    output = generate_response(prompt, model)
    return {
        "agent": "Research Retrieval Agent",
        "agent_id": 1,
        "icon": "🔍",
        "status": "completed",
        "query": query,
        "context_used": context[:300] + "..." if len(context) > 300 else context,
        "output": output,
        "reasoning": "Activated to retrieve and organize relevant research knowledge from uploaded documents and general knowledge base.",
    }


# =============================================================================
# AGENT 2: Literature Review Agent
# =============================================================================
def literature_review_agent(query: str, retrieval_output: str, model=None) -> dict:
    """
    Agent 2 – Literature Review Agent
    Generates a structured literature review by comparing papers, identifying
    themes, and summarizing prior work.

    IBM watsonx.ai Integration Point ← AGENT 2
    """
    context = retrieve_research(query, top_k=4)

    prompt = textwrap.dedent(f"""
    You are a Literature Review Agent powered by IBM Granite AI.
    Your role is to generate a comprehensive, structured literature review.

    Research Topic: {query}

    Research Retrieval Summary:
    {retrieval_output[:600]}

    Additional Context from Knowledge Base:
    {context}

    Task:
    1. Write a structured Literature Review (4-6 sentences covering evolution of the field)
    2. List Key Contributions from the literature (at least 4 bullet points)
    3. Describe the Research Landscape Overview
    4. Identify Common Themes across works
    5. Note any Agreements or Disagreements between studies

    Format with clear section headers. Be academic and precise.
    """).strip()

    output = generate_response(prompt, model)
    return {
        "agent": "Literature Review Agent",
        "agent_id": 2,
        "icon": "📚",
        "status": "completed",
        "query": query,
        "output": output,
        "reasoning": "Activated after retrieval to synthesize multiple sources into a coherent literature review.",
    }


def generate_literature_review(query: str, model=None) -> str:
    """Standalone helper for literature review generation (used in API routes)."""
    result = literature_review_agent(query, "", model)
    return result["output"]


# =============================================================================
# AGENT 3: Research Gap Analysis Agent
# =============================================================================
def gap_analysis_agent(query: str, literature_output: str, model=None) -> dict:
    """
    Agent 3 – Research Gap Analysis Agent
    Identifies unexplored areas, unanswered questions, and novel opportunities
    in the current body of research.

    IBM watsonx.ai Integration Point ← AGENT 3
    """
    context = retrieve_research(f"limitations gaps future work {query}", top_k=3)

    prompt = textwrap.dedent(f"""
    You are a Research Gap Analysis Agent powered by IBM Granite AI.
    Your role is to critically analyze the literature and identify missing opportunities.

    Research Topic: {query}

    Literature Review Summary:
    {literature_output[:600]}

    Additional Context:
    {context}

    Task:
    1. Identify 4-6 specific Research Gaps (unanswered questions, unexplored areas)
    2. Highlight 3-4 Novel Research Opportunities
    3. Point out Limitations in existing studies
    4. Suggest specific areas for Improvement
    5. Rate each gap by priority (High / Medium / Low)

    Be specific, actionable, and academic in your analysis.
    """).strip()

    output = generate_response(prompt, model)
    return {
        "agent": "Research Gap Analysis Agent",
        "agent_id": 3,
        "icon": "🔬",
        "status": "completed",
        "query": query,
        "output": output,
        "reasoning": "Activated to identify unexplored research areas and novel opportunities beyond current literature.",
    }


def identify_research_gaps(query: str, model=None) -> str:
    """Standalone helper for gap identification."""
    result = gap_analysis_agent(query, "", model)
    return result["output"]


# =============================================================================
# AGENT 4: Trend Forecasting Agent
# =============================================================================
def trend_forecasting_agent(query: str, gap_output: str, model=None) -> dict:
    """
    Agent 4 – Trend Forecasting Agent
    Predicts future research directions by analyzing emerging topics,
    publication patterns, and technological developments.

    IBM watsonx.ai Integration Point ← AGENT 4
    """
    prompt = textwrap.dedent(f"""
    You are a Trend Forecasting Agent powered by IBM Granite AI.
    Your role is to predict future research directions and emerging trends.

    Research Domain: {query}

    Gap Analysis Insights:
    {gap_output[:500]}

    Task:
    1. Identify 5 Future Research Areas with high growth potential
    2. List Emerging Technologies or methods relevant to this domain
    3. Predict Research Hotspots for the next 3-5 years
    4. Highlight convergence trends (fields merging or overlapping)
    5. Provide Growth Potential estimate for each trend (High / Medium / Nascent)

    Examples of trend categories to consider:
    - AI-powered domain-specific applications
    - Sustainable / green approaches
    - Quantum or neuromorphic computing intersections
    - Foundation model adaptations
    - Human-AI collaborative frameworks

    Be forward-looking and evidence-based.
    """).strip()

    output = generate_response(prompt, model)
    return {
        "agent": "Trend Forecasting Agent",
        "agent_id": 4,
        "icon": "📈",
        "status": "completed",
        "query": query,
        "output": output,
        "reasoning": "Activated to predict emerging trends and future research directions based on gap analysis and domain knowledge.",
    }


def forecast_research_trends(query: str, model=None) -> str:
    """Standalone helper for trend forecasting."""
    result = trend_forecasting_agent(query, "", model)
    return result["output"]


# =============================================================================
# AGENT 5: Research Advisor Agent
# =============================================================================
def research_advisor_agent(query: str, trend_output: str, gap_output: str, model=None) -> dict:
    """
    Agent 5 – Research Advisor Agent
    Provides strategic research guidance, suggesting questions, methodologies,
    datasets, and publication venues.

    IBM watsonx.ai Integration Point ← AGENT 5
    """
    prompt = textwrap.dedent(f"""
    You are a Research Advisor Agent powered by IBM Granite AI.
    Your role is to provide strategic, actionable research guidance.

    Research Topic: {query}

    Identified Trends:
    {trend_output[:400]}

    Identified Gaps:
    {gap_output[:400]}

    Task:
    1. Suggest 3-5 specific Research Questions (ready for thesis/paper framing)
    2. Recommend suitable Research Methodologies
    3. Suggest relevant Datasets or Data Sources
    4. Recommend Publication Venues (journals, conferences)
    5. Propose 2-3 Thesis or Project Ideas
    6. Outline an Actionable Research Roadmap (short-term and long-term steps)

    Make your advice concrete, practical, and motivating for researchers and students.
    """).strip()

    output = generate_response(prompt, model)
    return {
        "agent": "Research Advisor Agent",
        "agent_id": 5,
        "icon": "🎯",
        "status": "completed",
        "query": query,
        "output": output,
        "reasoning": "Activated to synthesize all prior agent outputs into a concrete, actionable research plan.",
    }


def generate_research_plan(query: str, model=None) -> str:
    """Standalone helper for research plan generation."""
    result = research_advisor_agent(query, "", "", model)
    return result["output"]


# =============================================================================
# MASTER ORCHESTRATOR AGENT
# =============================================================================
def orchestrate_agents(query: str, model=None) -> dict:
    """
    Master Orchestrator Agent – The Brain of ResearchMind AI
    Coordinates all five specialized agents in sequence, passing outputs
    between agents and generating a comprehensive final research report.

    IBM watsonx.ai Integration Point ← ORCHESTRATOR (uses shared model instance)

    Workflow:
    Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5 → Final Report
    """
    # Share a single model instance across all agents for efficiency
    if model is None:
        model = get_watsonx_model()

    workflow_log = []
    agents_results = []

    # ── Step 1: Research Retrieval ────────────────────────────────────────────
    workflow_log.append({"step": 1, "agent": "Research Retrieval Agent", "status": "running"})
    r1 = retrieval_agent(query, model)
    agents_results.append(r1)
    workflow_log[-1]["status"] = "completed"

    # ── Step 2: Literature Review ─────────────────────────────────────────────
    workflow_log.append({"step": 2, "agent": "Literature Review Agent", "status": "running"})
    r2 = literature_review_agent(query, r1["output"], model)
    agents_results.append(r2)
    workflow_log[-1]["status"] = "completed"

    # ── Step 3: Gap Analysis ──────────────────────────────────────────────────
    workflow_log.append({"step": 3, "agent": "Research Gap Analysis Agent", "status": "running"})
    r3 = gap_analysis_agent(query, r2["output"], model)
    agents_results.append(r3)
    workflow_log[-1]["status"] = "completed"

    # ── Step 4: Trend Forecasting ─────────────────────────────────────────────
    workflow_log.append({"step": 4, "agent": "Trend Forecasting Agent", "status": "running"})
    r4 = trend_forecasting_agent(query, r3["output"], model)
    agents_results.append(r4)
    workflow_log[-1]["status"] = "completed"

    # ── Step 5: Research Advisor ──────────────────────────────────────────────
    workflow_log.append({"step": 5, "agent": "Research Advisor Agent", "status": "running"})
    r5 = research_advisor_agent(query, r4["output"], r3["output"], model)
    agents_results.append(r5)
    workflow_log[-1]["status"] = "completed"

    # ── Final Synthesis ───────────────────────────────────────────────────────
    final_prompt = textwrap.dedent(f"""
    You are the Master Orchestrator of ResearchMind AI powered by IBM Granite.
    Synthesize the following multi-agent research analysis into a polished final report.

    Topic: {query}

    Agent Summaries:
    - Retrieval: {r1["output"][:300]}
    - Literature Review: {r2["output"][:300]}
    - Gap Analysis: {r3["output"][:300]}
    - Trends: {r4["output"][:300]}
    - Advisory: {r5["output"][:300]}

    Generate:
    1. Executive Summary (3-4 sentences)
    2. Top 3 Research Insights
    3. Most Promising Research Direction
    4. Recommended Next Step for a researcher

    Keep the final report concise, impactful, and professional.
    """).strip()

    final_report = generate_response(final_prompt, model)

    return {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "agents": agents_results,
        "workflow_log": workflow_log,
        "final_report": final_report,
        "rag_documents": len(rag_store["documents"]),
        "rag_chunks": len(rag_store["chunks"]),
        "watsonx_model": "ibm/granite-13b-instruct-v2",
        "mode": "live" if (WATSONX_AVAILABLE and WATSONX_API_KEY) else "demo",
    }


# =============================================================================
# Flask Routes
# =============================================================================

@app.route("/")
def index():
    """Serve the main single-page application dashboard."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Main research analysis endpoint.
    Accepts a research query and runs the full multi-agent pipeline.
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Research query is required."}), 400
    if len(query) < 5:
        return jsonify({"error": "Query too short. Please provide a meaningful research topic."}), 400

    result = orchestrate_agents(query)
    return jsonify(result)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Document upload endpoint for the RAG system.
    Accepts PDF and TXT files, extracts text, and indexes into the RAG store.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    filename = file.filename or "untitled"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ("pdf", "txt"):
        return jsonify({"error": "Only PDF and TXT files are supported."}), 400

    file_bytes = file.read()

    if ext == "pdf":
        content = extract_text_from_pdf(file_bytes)
    else:
        content = file_bytes.decode("utf-8", errors="replace")

    if len(content.strip()) < 50:
        return jsonify({"error": "File appears empty or unreadable."}), 400

    title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").title()
    result = add_document_to_rag(title, content)

    return jsonify({
        "success": True,
        "title": title,
        "doc_id": result["doc_id"],
        "chunks_indexed": result["chunks"],
        "duplicate": result["duplicate"],
        "total_documents": len(rag_store["documents"]),
        "total_chunks": len(rag_store["chunks"]),
    })


@app.route("/api/upload-text", methods=["POST"])
def api_upload_text():
    """
    Text-based knowledge upload.
    Allows users to paste abstracts, notes, or article text directly.
    """
    data = request.get_json(silent=True) or {}
    title   = data.get("title", "User Note").strip()
    content = data.get("content", "").strip()

    if not content or len(content) < 50:
        return jsonify({"error": "Content must be at least 50 characters."}), 400

    result = add_document_to_rag(title or "Untitled Note", content)
    return jsonify({
        "success": True,
        "title": title,
        "doc_id": result["doc_id"],
        "chunks_indexed": result["chunks"],
        "duplicate": result["duplicate"],
        "total_documents": len(rag_store["documents"]),
        "total_chunks": len(rag_store["chunks"]),
    })


@app.route("/api/rag-status", methods=["GET"])
def api_rag_status():
    """Return current RAG knowledge store status."""
    return jsonify({
        "documents": [
            {
                "id":          d["id"],
                "title":       d["title"],
                "chunk_count": d["chunk_count"],
                "uploaded_at": d["uploaded_at"],
            }
            for d in rag_store["documents"]
        ],
        "total_documents": len(rag_store["documents"]),
        "total_chunks":    len(rag_store["chunks"]),
        "watsonx_status":  "configured" if (WATSONX_AVAILABLE and WATSONX_API_KEY) else "demo_mode",
        "model":           "ibm/granite-13b-instruct-v2",
        "pdf_backend":     PDF_BACKEND,
    })


@app.route("/api/clear-rag", methods=["POST"])
def api_clear_rag():
    """Clear all documents from the in-memory RAG store."""
    rag_store["documents"].clear()
    rag_store["chunks"].clear()
    return jsonify({"success": True, "message": "RAG store cleared."})


# =============================================================================
# HTML Template – ResearchMind AI Dashboard
# Single-page application using Bootstrap 5 + vanilla JS
# =============================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ResearchMind AI – Intelligent Research Companion</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"/>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css"/>
  <style>
    :root {
      --ibm-blue:#0062ff; --ibm-dark:#001141; --ibm-teal:#009d9a;
      --ibm-purple:#8a3ffc; --ibm-green:#24a148; --ibm-red:#da1e28;
      --ibm-orange:#ff832b; --ibm-gray:#f4f4f4; --ibm-card:#ffffff;
    }
    body { background:#f0f4ff; font-family:'IBM Plex Sans','Segoe UI',sans-serif; }
    .navbar-brand span { color:var(--ibm-blue); }
    .hero {
      background:linear-gradient(135deg,var(--ibm-dark) 0%,#003a8c 60%,var(--ibm-teal) 100%);
      color:#fff; padding:3rem 0 2rem;
    }
    .hero h1 { font-size:2.4rem; font-weight:700; }
    .hero p  { opacity:.85; font-size:1.1rem; }
    .badge-ibm { background:var(--ibm-blue); font-size:.7rem; }
    .badge-granite { background:var(--ibm-purple); font-size:.7rem; }
    .badge-rag  { background:var(--ibm-teal); font-size:.7rem; }
    .card { border:none; border-radius:12px; box-shadow:0 2px 12px rgba(0,0,0,.08); }
    .card-header { border-radius:12px 12px 0 0!important; font-weight:600; }
    /* Agent Cards */
    .agent-card { border-left:4px solid var(--ibm-blue); transition:all .2s; }
    .agent-card:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,.12); }
    .agent-card.a1 { border-color:var(--ibm-blue); }
    .agent-card.a2 { border-color:var(--ibm-purple); }
    .agent-card.a3 { border-color:var(--ibm-teal); }
    .agent-card.a4 { border-color:var(--ibm-orange); }
    .agent-card.a5 { border-color:var(--ibm-green); }
    .agent-icon { font-size:2rem; }
    .agent-badge { font-size:.65rem; letter-spacing:.5px; text-transform:uppercase; }
    /* Status indicators */
    .status-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
    .status-idle    { background:#ccc; }
    .status-running { background:#ff832b; animation:pulse .8s infinite; }
    .status-done    { background:var(--ibm-green); }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
    /* RAG Status */
    .rag-bar { height:6px; border-radius:3px; background:var(--ibm-teal); }
    /* Output area */
    .output-text { white-space:pre-wrap; font-size:.88rem; line-height:1.7;
                   background:#f8faff; border-radius:8px; padding:1rem;
                   max-height:320px; overflow-y:auto; }
    /* Knowledge Graph */
    #kgCanvas { background:#fff; border-radius:12px; }
    /* Workflow steps */
    .wf-step { display:flex; align-items:center; gap:.7rem; padding:.5rem .8rem;
               border-radius:8px; margin-bottom:.4rem; background:#f0f4ff; }
    .wf-step.done { background:#e6f9ed; }
    .wf-step.running { background:#fff7ee; }
    /* Spinner overlay */
    #loadingOverlay {
      position:fixed; inset:0; background:rgba(0,0,17,.55);
      display:none; z-index:9999; flex-direction:column;
      align-items:center; justify-content:center; color:#fff;
    }
    #loadingOverlay .spinner-border { width:3.5rem; height:3.5rem; }
    /* Responsive tweaks */
    @media(max-width:768px){ .hero h1{font-size:1.7rem;} }
    .final-report-box {
      background:linear-gradient(135deg,#e8f0fe,#f0f9ff);
      border-left:5px solid var(--ibm-blue);
      border-radius:10px; padding:1.5rem;
    }
    .tab-content { padding-top:1rem; }
    select, textarea, input[type=text] { border-radius:8px!important; }
    .btn-ibm { background:var(--ibm-blue); color:#fff; border:none; }
    .btn-ibm:hover { background:#0050d0; color:#fff; }
  </style>
</head>
<body>

<!-- Loading Overlay -->
<div id="loadingOverlay">
  <div class="spinner-border text-light mb-3"></div>
  <h5>ResearchMind AI is thinking...</h5>
  <small class="text-light opacity-75">IBM Granite agents are collaborating</small>
</div>

<!-- Navbar -->
<nav class="navbar navbar-dark" style="background:var(--ibm-dark);">
  <div class="container">
    <a class="navbar-brand fw-bold" href="#">
      <i class="bi bi-journal-richtext me-2" style="color:var(--ibm-teal)"></i>
      <span>Research</span>Mind AI
    </a>
    <div class="d-flex gap-2 align-items-center">
      <span class="badge badge-ibm rounded-pill">IBM watsonx.ai</span>
      <span class="badge badge-granite rounded-pill">Granite Models</span>
      <span class="badge badge-rag rounded-pill">RAG System</span>
      <span id="modeTag" class="badge bg-secondary rounded-pill">Loading...</span>
    </div>
  </div>
</nav>

<!-- Hero Section -->
<div class="hero">
  <div class="container text-center">
    <h1><i class="bi bi-cpu me-2"></i>ResearchMind AI</h1>
    <p class="lead mb-3">Intelligent Research Companion powered by IBM Granite &amp; Multi-Agent AI</p>
    <div class="d-flex justify-content-center gap-3 flex-wrap">
      <span class="badge bg-light text-dark"><i class="bi bi-robot me-1"></i>5 Specialized Agents</span>
      <span class="badge bg-light text-dark"><i class="bi bi-database me-1"></i>RAG Knowledge Base</span>
      <span class="badge bg-light text-dark"><i class="bi bi-diagram-3 me-1"></i>Agentic AI Architecture</span>
      <span class="badge bg-light text-dark"><i class="bi bi-graph-up me-1"></i>Trend Forecasting</span>
    </div>
  </div>
</div>

<div class="container my-4">
  <div class="row g-4">

    <!-- LEFT COLUMN: Input + RAG -->
    <div class="col-lg-4">

      <!-- Research Query Card -->
      <div class="card mb-4">
        <div class="card-header text-white" style="background:var(--ibm-blue);">
          <i class="bi bi-search me-2"></i>Research Query
        </div>
        <div class="card-body">
          <div class="mb-3">
            <label class="form-label fw-semibold">Research Topic or Question</label>
            <textarea id="queryInput" class="form-control" rows="4"
              placeholder="e.g., Large Language Models in Healthcare&#10;Climate Change Prediction using AI&#10;Quantum Computing for Cryptography"></textarea>
          </div>
          <button class="btn btn-ibm w-100" onclick="runFullAnalysis()">
            <i class="bi bi-play-fill me-2"></i>Launch Multi-Agent Analysis
          </button>
        </div>
      </div>

      <!-- Quick Examples -->
      <div class="card mb-4">
        <div class="card-header bg-light"><i class="bi bi-lightning me-2"></i>Quick Examples</div>
        <div class="card-body p-2">
          <div class="d-flex flex-wrap gap-2">
            <button class="btn btn-sm btn-outline-primary" onclick="setQuery('AI in Medical Diagnosis')">AI Healthcare</button>
            <button class="btn btn-sm btn-outline-primary" onclick="setQuery('Transformer Models for NLP')">Transformers NLP</button>
            <button class="btn btn-sm btn-outline-primary" onclick="setQuery('Federated Learning Privacy')">Federated Learning</button>
            <button class="btn btn-sm btn-outline-primary" onclick="setQuery('Quantum Machine Learning')">Quantum ML</button>
            <button class="btn btn-sm btn-outline-primary" onclick="setQuery('Sustainable AI Green Computing')">Green AI</button>
            <button class="btn btn-sm btn-outline-primary" onclick="setQuery('Explainable AI for Finance')">XAI Finance</button>
          </div>
        </div>
      </div>

      <!-- RAG Upload Card -->
      <div class="card mb-4">
        <div class="card-header text-white" style="background:var(--ibm-teal);">
          <i class="bi bi-cloud-upload me-2"></i>Knowledge Upload (RAG)
        </div>
        <div class="card-body">
          <ul class="nav nav-tabs mb-3" id="uploadTab">
            <li class="nav-item">
              <a class="nav-link active" data-bs-toggle="tab" href="#tabFile">
                <i class="bi bi-file-earmark-pdf me-1"></i>File
              </a>
            </li>
            <li class="nav-item">
              <a class="nav-link" data-bs-toggle="tab" href="#tabText">
                <i class="bi bi-text-paragraph me-1"></i>Text
              </a>
            </li>
          </ul>
          <div class="tab-content">
            <div class="tab-pane fade show active" id="tabFile">
              <input type="file" id="fileUpload" class="form-control mb-2" accept=".pdf,.txt"/>
              <small class="text-muted">PDF or TXT – max 16 MB</small>
              <button class="btn btn-sm mt-2 w-100" style="background:var(--ibm-teal);color:#fff;"
                      onclick="uploadFile()">
                <i class="bi bi-upload me-1"></i>Upload &amp; Index
              </button>
            </div>
            <div class="tab-pane fade" id="tabText">
              <input type="text" id="textTitle" class="form-control mb-2"
                     placeholder="Document title"/>
              <textarea id="textContent" class="form-control mb-2" rows="4"
                        placeholder="Paste abstract, notes, or article text here..."></textarea>
              <button class="btn btn-sm w-100" style="background:var(--ibm-teal);color:#fff;"
                      onclick="uploadText()">
                <i class="bi bi-plus-circle me-1"></i>Add to Knowledge Base
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- RAG Status -->
      <div class="card mb-4">
        <div class="card-header bg-light"><i class="bi bi-database me-2"></i>RAG Knowledge Base</div>
        <div class="card-body">
          <div class="d-flex justify-content-between small mb-1">
            <span>Documents</span><strong id="ragDocs">0</strong>
          </div>
          <div class="d-flex justify-content-between small mb-1">
            <span>Text Chunks</span><strong id="ragChunks">0</strong>
          </div>
          <div class="d-flex justify-content-between small mb-2">
            <span>PDF Backend</span><strong id="ragPdf">—</strong>
          </div>
          <div id="ragDocList" class="mt-2"></div>
          <button class="btn btn-sm btn-outline-danger w-100 mt-2" onclick="clearRag()">
            <i class="bi bi-trash me-1"></i>Clear Knowledge Base
          </button>
        </div>
      </div>

    </div><!-- /left col -->

    <!-- RIGHT COLUMN: Dashboard -->
    <div class="col-lg-8">

      <!-- Agent Workflow Visualization -->
      <div class="card mb-4">
        <div class="card-header text-white" style="background:var(--ibm-dark);">
          <i class="bi bi-diagram-3 me-2"></i>Agentic AI Workflow
          <small class="ms-2 opacity-75">IBM watsonx.ai · Granite Models</small>
        </div>
        <div class="card-body">
          <div class="row g-2" id="agentPanels">
            <!-- Populated by JS -->
          </div>
        </div>
      </div>

      <!-- Results Tabs -->
      <div class="card mb-4" id="resultsCard" style="display:none;">
        <div class="card-header" style="background:#e8f0fe;">
          <i class="bi bi-clipboard-data me-2"></i>Research Intelligence Dashboard
        </div>
        <div class="card-body">
          <ul class="nav nav-pills mb-3" id="resultsTabs">
            <li class="nav-item"><a class="nav-link active" data-bs-toggle="tab" href="#tabFinal"><i class="bi bi-star me-1"></i>Final Report</a></li>
            <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#tabRetrieval"><i class="bi bi-search me-1"></i>Retrieval</a></li>
            <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#tabLitReview"><i class="bi bi-book me-1"></i>Lit. Review</a></li>
            <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#tabGaps"><i class="bi bi-patch-question me-1"></i>Gaps</a></li>
            <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#tabTrends"><i class="bi bi-graph-up me-1"></i>Trends</a></li>
            <li class="nav-item"><a class="nav-link" data-bs-toggle="tab" href="#tabAdvisor"><i class="bi bi-compass me-1"></i>Advisor</a></li>
          </ul>
          <div class="tab-content">
            <div class="tab-pane fade show active" id="tabFinal">
              <div id="finalReportContent"></div>
            </div>
            <div class="tab-pane fade" id="tabRetrieval">
              <div id="retrievalContent" class="output-text"></div>
            </div>
            <div class="tab-pane fade" id="tabLitReview">
              <div id="litReviewContent" class="output-text"></div>
            </div>
            <div class="tab-pane fade" id="tabGaps">
              <div id="gapsContent" class="output-text"></div>
            </div>
            <div class="tab-pane fade" id="tabTrends">
              <div id="trendsContent" class="output-text"></div>
            </div>
            <div class="tab-pane fade" id="tabAdvisor">
              <div id="advisorContent" class="output-text"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Knowledge Graph -->
      <div class="card mb-4">
        <div class="card-header bg-light">
          <i class="bi bi-share me-2"></i>Knowledge Graph
          <small class="text-muted ms-2">Topics · Concepts · Relationships</small>
        </div>
        <div class="card-body text-center">
          <canvas id="kgCanvas" width="680" height="280"></canvas>
          <small class="text-muted d-block mt-1">
            Run analysis to populate the knowledge graph
          </small>
        </div>
      </div>

    </div><!-- /right col -->
  </div><!-- /row -->
</div><!-- /container -->

<!-- Footer -->
<footer class="text-center py-3 mt-2" style="background:var(--ibm-dark);color:#aaa;font-size:.8rem;">
  ResearchMind AI &nbsp;|&nbsp; IBM watsonx.ai Studio &nbsp;|&nbsp;
  IBM Granite Models &nbsp;|&nbsp; Agentic AI Architecture &nbsp;|&nbsp;
  Built for IBM SkillsBuild &amp; Hackathons
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
<script>
// ============================================================
// ResearchMind AI – Frontend JavaScript
// ============================================================

const AGENT_META = [
  { id:1, name:"Research Retrieval Agent",    icon:"🔍", color:"#0062ff", cls:"a1",
    desc:"Retrieves & organizes research knowledge from documents and knowledge base." },
  { id:2, name:"Literature Review Agent",     icon:"📚", color:"#8a3ffc", cls:"a2",
    desc:"Synthesizes multiple sources into a structured literature review." },
  { id:3, name:"Research Gap Analysis Agent", icon:"🔬", color:"#009d9a", cls:"a3",
    desc:"Identifies unexplored areas and novel research opportunities." },
  { id:4, name:"Trend Forecasting Agent",     icon:"📈", color:"#ff832b", cls:"a4",
    desc:"Predicts emerging research directions and future hotspots." },
  { id:5, name:"Research Advisor Agent",      icon:"🎯", color:"#24a148", cls:"a5",
    desc:"Provides strategic guidance and actionable research plans." },
];

// Render initial agent panels
function renderAgentPanels(states) {
  const container = document.getElementById("agentPanels");
  container.innerHTML = "";
  AGENT_META.forEach(ag => {
    const state = (states || {})[ag.id] || "idle";
    const dotCls = state === "idle" ? "status-idle" :
                   state === "running" ? "status-running" : "status-done";
    const statusLabel = state === "idle" ? "Idle" :
                        state === "running" ? "Running…" : "✓ Done";
    container.innerHTML += `
      <div class="col-md-6 col-lg-4">
        <div class="card agent-card ${ag.cls} h-100" id="agentCard${ag.id}">
          <div class="card-body py-3">
            <div class="d-flex align-items-start gap-2">
              <span class="agent-icon">${ag.icon}</span>
              <div class="flex-grow-1">
                <div class="fw-semibold small" style="color:${ag.color}">${ag.name}</div>
                <div class="text-muted" style="font-size:.75rem;">${ag.desc}</div>
              </div>
            </div>
            <div class="mt-2 d-flex align-items-center gap-2">
              <span class="status-dot ${dotCls}" id="dot${ag.id}"></span>
              <span class="agent-badge text-muted" id="status${ag.id}">${statusLabel}</span>
            </div>
            <div class="mt-1" id="reasoning${ag.id}" style="font-size:.72rem;color:#666;display:none;"></div>
          </div>
        </div>
      </div>`;
  });
  // Orchestrator card
  container.innerHTML += `
    <div class="col-md-6 col-lg-4">
      <div class="card agent-card h-100" style="border-color:#da1e28;">
        <div class="card-body py-3">
          <div class="d-flex align-items-start gap-2">
            <span class="agent-icon">🧠</span>
            <div>
              <div class="fw-semibold small" style="color:#da1e28;">Master Orchestrator</div>
              <div class="text-muted" style="font-size:.75rem;">
                Coordinates all agents &amp; generates the final research report.
              </div>
            </div>
          </div>
          <div class="mt-2 d-flex align-items-center gap-2">
            <span class="status-dot" id="dotOrch" style="background:#ccc;"></span>
            <span class="agent-badge text-muted" id="statusOrch">Idle</span>
          </div>
        </div>
      </div>
    </div>`;
}

function setAgentState(agentId, state, reasoning) {
  const dot = document.getElementById(`dot${agentId}`);
  const statusEl = document.getElementById(`status${agentId}`);
  const reasonEl = document.getElementById(`reasoning${agentId}`);
  if (!dot) return;
  dot.className = `status-dot status-${state === "completed" ? "done" : state}`;
  statusEl.textContent = state === "idle" ? "Idle" :
                         state === "running" ? "Running…" : "✓ Completed";
  if (reasoning && reasonEl) {
    reasonEl.textContent = "Why: " + reasoning;
    reasonEl.style.display = "block";
  }
}

function setQuery(q) {
  document.getElementById("queryInput").value = q;
}

async function runFullAnalysis() {
  const query = document.getElementById("queryInput").value.trim();
  if (!query) { alert("Please enter a research topic or question."); return; }

  // Reset UI
  renderAgentPanels({});
  document.getElementById("resultsCard").style.display = "none";
  document.getElementById("loadingOverlay").style.display = "flex";

  // Animate orchestrator start
  const dotOrch = document.getElementById("dotOrch");
  const statusOrch = document.getElementById("statusOrch");
  dotOrch.className = "status-dot status-running";
  statusOrch.textContent = "Orchestrating…";

  // Animate agents sequentially (visual only – actual call is async)
  let delay = 0;
  AGENT_META.forEach(ag => {
    delay += 600;
    setTimeout(() => setAgentState(ag.id, "running"), delay);
  });

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query })
    });
    const data = await res.json();

    if (data.error) { alert("Error: " + data.error); return; }

    // Update mode badge
    document.getElementById("modeTag").textContent =
      data.mode === "live" ? "🟢 Live (watsonx.ai)" : "🟡 Demo Mode";
    document.getElementById("modeTag").className =
      data.mode === "live" ? "badge bg-success rounded-pill" : "badge bg-warning text-dark rounded-pill";

    // Update agent states
    (data.agents || []).forEach(ag => {
      setAgentState(ag.agent_id, "completed", ag.reasoning);
    });
    dotOrch.className = "status-dot status-done";
    statusOrch.textContent = "✓ Report Generated";

    // Populate results
    populateResults(data);

    // Draw knowledge graph
    drawKnowledgeGraph(query, data.agents || []);

    document.getElementById("resultsCard").style.display = "block";
    document.getElementById("resultsCard").scrollIntoView({ behavior:"smooth", block:"start" });

  } catch (e) {
    alert("Network error: " + e.message);
  } finally {
    document.getElementById("loadingOverlay").style.display = "none";
  }
}

function populateResults(data) {
  const agents = data.agents || [];
  const get = id => (agents.find(a => a.agent_id === id) || {}).output || "—";

  // Final report
  document.getElementById("finalReportContent").innerHTML = `
    <div class="final-report-box mb-3">
      <h6 class="fw-bold mb-2"><i class="bi bi-star-fill text-warning me-2"></i>Executive Research Report</h6>
      <div style="white-space:pre-wrap;font-size:.9rem;line-height:1.8;">${escHtml(data.final_report)}</div>
    </div>
    <div class="row g-2 mt-1">
      <div class="col-sm-4">
        <div class="p-2 rounded text-center" style="background:#e8f0fe;">
          <div class="fw-bold" style="color:var(--ibm-blue)">${agents.length}</div>
          <small class="text-muted">Agents Used</small>
        </div>
      </div>
      <div class="col-sm-4">
        <div class="p-2 rounded text-center" style="background:#f0f9ff;">
          <div class="fw-bold" style="color:var(--ibm-teal)">${data.rag_documents}</div>
          <small class="text-muted">Documents</small>
        </div>
      </div>
      <div class="col-sm-4">
        <div class="p-2 rounded text-center" style="background:#f3eeff;">
          <div class="fw-bold" style="color:var(--ibm-purple)">${data.rag_chunks}</div>
          <small class="text-muted">RAG Chunks</small>
        </div>
      </div>
    </div>`;

  document.getElementById("retrievalContent").textContent = get(1);
  document.getElementById("litReviewContent").textContent = get(2);
  document.getElementById("gapsContent").textContent      = get(3);
  document.getElementById("trendsContent").textContent    = get(4);
  document.getElementById("advisorContent").textContent   = get(5);
}

function escHtml(str) {
  return (str || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ── Knowledge Graph (Canvas) ──────────────────────────────────────────────────
function drawKnowledgeGraph(query, agents) {
  const canvas = document.getElementById("kgCanvas");
  const ctx    = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Nodes: center = query, surrounding = agents
  const cx = W / 2, cy = H / 2;
  const nodes = [
    { label: query.length > 20 ? query.substring(0,18)+"…" : query,
      x: cx, y: cy, color:"#0062ff", r:28, bold:true },
  ];
  const colors = ["#0062ff","#8a3ffc","#009d9a","#ff832b","#24a148"];
  const radius = 100;
  agents.forEach((ag, i) => {
    const angle = (2 * Math.PI * i) / agents.length - Math.PI / 2;
    nodes.push({
      label: ag.icon + " " + (ag.agent || "").split(" ")[0],
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
      color: colors[i % colors.length],
      r: 22,
    });
  });

  // Draw edges
  ctx.lineWidth = 1.5;
  nodes.slice(1).forEach(n => {
    ctx.beginPath();
    ctx.setLineDash([4,3]);
    ctx.strokeStyle = n.color + "88";
    ctx.moveTo(nodes[0].x, nodes[0].y);
    ctx.lineTo(n.x, n.y);
    ctx.stroke();
  });
  ctx.setLineDash([]);

  // Draw nodes
  nodes.forEach(n => {
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, 2 * Math.PI);
    ctx.fillStyle = n.color + "22";
    ctx.fill();
    ctx.strokeStyle = n.color;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#1f2328";
    ctx.font = (n.bold ? "bold " : "") + "11px 'IBM Plex Sans',sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(n.label, n.x, n.y);
  });
}

// ── File Upload ───────────────────────────────────────────────────────────────
async function uploadFile() {
  const fileInput = document.getElementById("fileUpload");
  if (!fileInput.files.length) { alert("Please select a file first."); return; }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const res  = await fetch("/api/upload", { method:"POST", body:formData });
    const data = await res.json();
    if (data.error) { alert("Upload error: " + data.error); return; }
    alert(`✅ "${data.title}" indexed!\n${data.chunks_indexed} chunks added.`);
    fileInput.value = "";
    refreshRagStatus();
  } catch(e) { alert("Upload failed: " + e.message); }
}

async function uploadText() {
  const title   = document.getElementById("textTitle").value.trim() || "User Note";
  const content = document.getElementById("textContent").value.trim();
  if (!content) { alert("Please enter some text content."); return; }

  const res  = await fetch("/api/upload-text", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({ title, content })
  });
  const data = await res.json();
  if (data.error) { alert("Error: " + data.error); return; }
  alert(`✅ "${data.title}" indexed!\n${data.chunks_indexed} chunks added.`);
  document.getElementById("textContent").value = "";
  refreshRagStatus();
}

async function clearRag() {
  if (!confirm("Clear all documents from the knowledge base?")) return;
  await fetch("/api/clear-rag", { method:"POST" });
  refreshRagStatus();
}

async function refreshRagStatus() {
  try {
    const res  = await fetch("/api/rag-status");
    const data = await res.json();
    document.getElementById("ragDocs").textContent   = data.total_documents;
    document.getElementById("ragChunks").textContent = data.total_chunks;
    document.getElementById("ragPdf").textContent    = data.pdf_backend;
    document.getElementById("modeTag").textContent   =
      data.watsonx_status === "configured" ? "🟢 Live (watsonx.ai)" : "🟡 Demo Mode";
    document.getElementById("modeTag").className =
      data.watsonx_status === "configured"
        ? "badge bg-success rounded-pill"
        : "badge bg-warning text-dark rounded-pill";

    const listEl = document.getElementById("ragDocList");
    if (data.documents.length === 0) {
      listEl.innerHTML = `<small class="text-muted">No documents yet.</small>`;
    } else {
      listEl.innerHTML = data.documents.map(d =>
        `<div class="d-flex justify-content-between small py-1 border-bottom">
          <span class="text-truncate me-2" style="max-width:140px" title="${escHtml(d.title)}">${escHtml(d.title)}</span>
          <span class="badge bg-light text-dark">${d.chunk_count} chunks</span>
        </div>`
      ).join("");
    }
  } catch(e) { console.warn("RAG status fetch failed", e); }
}

// Initialize on load
document.addEventListener("DOMContentLoaded", () => {
  renderAgentPanels({});
  drawKnowledgeGraph("Research Topic", []);
  refreshRagStatus();
});
</script>
</body>
</html>
"""


# =============================================================================
# Application Entry Point
# =============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  ResearchMind AI – Intelligent Research Companion")
    print("  IBM watsonx.ai Studio + IBM Granite Models")
    print("  Agentic AI Architecture | RAG System")
    print("=" * 65)
    print(f"  watsonx.ai SDK : {'✅ Available' if WATSONX_AVAILABLE else '❌ Not installed (pip install ibm-watsonx-ai)'}")
    print(f"  API Key        : {'✅ Set' if WATSONX_API_KEY else '⚠️  Not set  – running in Demo Mode'}")
    print(f"  Project ID     : {'✅ Set' if WATSONX_PROJECT_ID else '⚠️  Not set  – running in Demo Mode'}")
    print(f"  watsonx URL    : {WATSONX_URL}")
    print(f"  PDF Backend    : {PDF_BACKEND}")
    print("=" * 65)
    print("  Dashboard → http://127.0.0.1:5000")
    print("=" * 65)
    app.run(debug=True, host="0.0.0.0", port=5000)
