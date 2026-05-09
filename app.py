import time
import random
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.graph import complaint_graph
from metrics.evaluator import evaluate, get_aggregate_metrics, init_metrics_db
from innovations.adaptive_router import init_db as init_routing_db
from innovations.self_improving import ensure_collection as init_examples_collection
from innovations.swarm_watcher import ensure_collection as init_complaints_collection, detect_swarm
from agents.ingestor import fetch_complaints
from agents.classifier import classify_complaint
from agents.router import route_complaint
from graph.state import ComplaintState
from main import build_initial_state

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="CFPB Complaint Resolution",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace !important; color: #f0f4ff !important; }

    .metric-card {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-top: 4px;
    }
    .complaint-card {
        background: #111827;
        border: 1px solid #1e2d45;
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .complaint-card.critical { border-left-color: #ef4444; }
    .complaint-card.high     { border-left-color: #f97316; }
    .complaint-card.medium   { border-left-color: #eab308; }
    .complaint-card.low      { border-left-color: #22c55e; }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 6px;
    }
    .badge-critical { background: #450a0a; color: #fca5a5; }
    .badge-high     { background: #431407; color: #fdba74; }
    .badge-medium   { background: #422006; color: #fde047; }
    .badge-low      { background: #052e16; color: #86efac; }
    .badge-team     { background: #0f172a; color: #7dd3fc; border: 1px solid #1e40af; }
    .badge-swarm    { background: #3b0764; color: #e879f9; border: 1px solid #7e22ce; }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #475569;
        border-bottom: 1px solid #1e2d45;
        padding-bottom: 6px;
        margin: 16px 0 10px 0;
    }
    .resolution-text {
        background: #0d1117;
        border: 1px solid #1e2d45;
        border-radius: 6px;
        padding: 16px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #94a3b8;
        white-space: pre-wrap;
        line-height: 1.6;
        max-height: 320px;
        overflow-y: auto;
    }
    .narrative-box {
        background: #0d1117;
        border: 1px solid #1e2d45;
        border-radius: 6px;
        padding: 16px;
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.7;
        max-height: 180px;
        overflow-y: auto;
        margin-bottom: 16px;
    }
    .swarm-alert {
        background: #1a0533;
        border: 1px solid #7e22ce;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.85rem;
        color: #e879f9;
    }
    .quality-bar-container {
        background: #1e2d45;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin-top: 6px;
    }
    .quality-bar {
        height: 6px;
        border-radius: 4px;
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
    }
    .report-section {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .report-section-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #3b82f6;
        margin-bottom: 12px;
    }
    div[data-testid="stSidebar"] {
        background-color: #080c17;
        border-right: 1px solid #1e2d45;
    }
    div[data-testid="stSidebar"] * { color: #94a3b8 !important; }
    .stButton > button {
        background: #1d4ed8;
        color: white !important;
        border: none;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 10px 24px;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #2563eb; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] { color: #60a5fa !important; }
    hr { border-color: #1e2d45; }

    /* ── Admin styles ── */
    .admin-banner {
        background: linear-gradient(135deg, #0f172a 0%, #0c1a2e 100%);
        border: 1px solid #1e3a5f;
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 28px;
    }
    .admin-section-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #334155;
        border-bottom: 1px solid #111827;
        padding-bottom: 8px;
        margin: 28px 0 16px 0;
    }
    .preview-card {
        background: #0d1117;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .preview-field-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #38bdf8;
        margin-bottom: 3px;
    }
    .preview-field-value {
        font-size: 0.88rem;
        color: #cbd5e1;
        margin-bottom: 14px;
    }
    .preview-narrative {
        background: #060a12;
        border: 1px solid #1e2d45;
        border-radius: 6px;
        padding: 16px;
        font-size: 0.84rem;
        color: #94a3b8;
        line-height: 1.75;
        max-height: 240px;
        overflow-y: auto;
        white-space: pre-wrap;
        margin-top: 6px;
    }
    .routing-outcome-card {
        background: linear-gradient(135deg, #0f2744 0%, #0d1b2e 100%);
        border: 1px solid #1e40af;
        border-radius: 12px;
        padding: 28px;
        margin: 8px 0 24px 0;
        position: relative;
        overflow: hidden;
    }
    .routing-outcome-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #38bdf8, #818cf8);
    }
    .routing-team-name {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa;
        letter-spacing: -0.02em;
        margin-bottom: 2px;
    }
    .routing-sublabel {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: #334155;
        margin-bottom: 20px;
    }
    .confidence-track {
        background: #0a1628;
        border: 1px solid #1e2d45;
        border-radius: 4px;
        height: 10px;
        width: 100%;
        margin: 6px 0 4px 0;
    }
    .confidence-fill {
        height: 10px;
        border-radius: 4px;
        background: linear-gradient(90deg, #1d4ed8, #38bdf8);
    }
    .path-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-right: 8px;
        margin-top: 14px;
    }
    .path-standard   { background: #052e16; color: #86efac; border: 1px solid #166534; }
    .path-compliance { background: #1c1306; color: #fbbf24; border: 1px solid #92400e; }
    .path-escalate   { background: #450a0a; color: #fca5a5; border: 1px solid #991b1b; }

    .innovation-card {
        background: #0d1117;
        border: 1px solid #1e2d45;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }
    .innovation-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        color: #7dd3fc;
        margin-bottom: 6px;
    }
    .innovation-body {
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.65;
    }
    .swarm-card {
        background: #120820;
        border: 1px solid #7e22ce;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }
    .swarm-card-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        color: #e879f9;
        margin-bottom: 6px;
    }
    .swarm-card-body {
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.65;
        margin-bottom: 14px;
    }
    .step-pill {
        display: inline-block;
        background: #0f172a;
        border: 1px solid #1e40af;
        border-radius: 999px;
        padding: 3px 12px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 18px;
    }
    .full-divider {
        border: none;
        border-top: 1px solid #1e2d45;
        margin: 28px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Init storage once ─────────────────────────────────────────────
@st.cache_resource
def init_storage():
    init_metrics_db()
    init_routing_db()
    init_examples_collection()
    init_complaints_collection()

init_storage()


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ CFPB Resolution")
    st.markdown("---")
    st.markdown("### Stack")
    st.markdown("""
- **LLM** — Groq / Llama 3.3 70B
- **Embeddings** — Voyage Finance-2
- **Vector DB** — Qdrant Cloud
- **Orchestration** — LangGraph
    """)


# ── Header ────────────────────────────────────────────────────────
st.markdown("# CFPB Complaint Resolution System")
st.markdown(
    "<span style='font-family:IBM Plex Mono;font-size:0.93rem;color:#94a3b8'>"
    "AI-powered complaint triage · root cause · compliance · remediation"
    "</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

tab_admin, tab_dashboard = st.tabs([
    "🛡️  Admin Panel",
    "📊  Dashboard",
])


# ══════════════════════════════════════════════════════════════════
# TAB 0 — Admin Panel  (full pipeline per product category)
# ══════════════════════════════════════════════════════════════════
with tab_admin:

    st.markdown(
        "<div class='admin-banner'>"
        "<div style='font-family:IBM Plex Mono;font-size:1rem;font-weight:600;"
        "color:#f0f4ff;margin-bottom:6px'>🛡️ Admin Panel</div>"
        "<div style='font-size:0.83rem;color:#94a3b8;font-size:0.95rem;line-height:1.6'>"
        "Select a product category, sample a live complaint, and run the complete "
        "resolution pipeline — classification → routing → compliance → "
        "root-cause → remediation → response letter → explainability."
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Product filter + Sample button ───────────────────────────
    col_filter, col_btn, _ = st.columns([2, 1, 2])
    with col_filter:
        preview_product = st.selectbox(
            "Product category",
            [
                "Select a product...", "Credit card", "Mortgage", "Debt collection",
                "Checking or savings account", "Credit reporting", "Money transfer",
                "Student loan", "Personal loan",
            ],
            key="admin_preview_filter",
        )
        preview_product_val = None if preview_product == "Select a product..." else preview_product

    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        sample_and_run_btn = st.button("▶  Sample & Run Full Pipeline", key="sample_run_btn")

    # ── Session state ─────────────────────────────────────────────
    for key in ("admin_raw", "admin_result", "admin_metrics", "admin_swarm_result"):
        if key not in st.session_state:
            st.session_state[key] = None

    # ── On button click: fetch → full pipeline ────────────────────
    if sample_and_run_btn:
        if preview_product == "Select a product...":
            st.warning("⚠ Please select a specific product category before running the pipeline.")
        else:
            st.session_state.admin_raw          = None
            st.session_state.admin_result       = None
            st.session_state.admin_metrics      = None
            st.session_state.admin_swarm_result = None

            with st.spinner("Fetching from CFPB API..."):
                pool = fetch_complaints(limit=20, product=preview_product_val)

            if not pool:
                st.error("No complaints returned. Try a different product filter.")
            else:
                c = random.choice(pool)
                st.session_state.admin_raw = c

                with st.spinner(f"Running full pipeline on complaint {c['complaint_id']} …"):
                    try:
                        initial_state = build_initial_state(c)
                        start   = time.time()
                        result  = complaint_graph.invoke(initial_state)
                        elapsed = int((time.time() - start) * 1000)
                        result["processing_time_ms"] = elapsed
                        metrics = evaluate(result)
                        st.session_state.admin_result  = result
                        st.session_state.admin_metrics = metrics
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")

    # ══════════════════════════════════════════════════════════════
    # RESULTS — only shown after pipeline completes
    # ══════════════════════════════════════════════════════════════
    if st.session_state.admin_result:
        result  = st.session_state.admin_result
        metrics = st.session_state.admin_metrics
        raw     = st.session_state.admin_raw

        severity  = result.get("severity", "medium").lower()
        team      = result.get("assigned_team", "—")
        score     = result.get("routing_score", 0.0)
        score_pct = int(score * 100)
        risk_l    = result.get("compliance_risk", "").lower()

        if severity == "critical":
            path_label, path_class = "ESCALATE",        "path-escalate"
        elif risk_l in ("high", "regulatory") or team == "legal":
            path_label, path_class = "COMPLIANCE PATH", "path-compliance"
        else:
            path_label, path_class = "STANDARD PATH",   "path-standard"

        # ── ① Identity card ──────────────────────────────────────
        st.markdown(
            "<div class='admin-section-label'>① Complaint Identity</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='complaint-card {severity}'>"
            f"<div style='font-family:IBM Plex Mono;font-size:1rem;font-weight:600;"
            f"color:#f0f4ff;margin-bottom:8px'>Complaint #{result['complaint_id']}</div>"
            f"<span class='badge badge-{severity}'>{severity}</span>"
            f"<span class='badge badge-team'>{team}</span>"
            f"<span class='badge badge-team'>{result.get('classified_product','—')}</span>"
            f"<span class='badge badge-team'>{result.get('issue_type','—')}</span>"
            + (f"<span class='badge badge-swarm'>⚡ SWARM</span>" if result.get('swarm_alert') else "")
            + f"<div style='margin-top:12px;font-size:0.93rem;color:#94a3b8'>"
            f"Company: <span style='color:#94a3b8'>{result.get('company','—')}</span>"
            f" &nbsp;·&nbsp; State: <span style='color:#94a3b8'>{raw.get('state','—')}</span>"
            f" &nbsp;·&nbsp; Via: <span style='color:#94a3b8'>{raw.get('submitted_via','—')}</span>"
            f" &nbsp;·&nbsp; Date: <span style='color:#94a3b8'>{raw.get('date_received','—')}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── ② Raw complaint details ───────────────────────────────
        st.markdown(
            "<div class='admin-section-label'>② Raw Complaint Details</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='preview-card'>"
            f"<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:0 28px'>"
            f"<div><div class='preview-field-label'>Company</div>"
            f"<div class='preview-field-value'>{raw.get('company','—')}</div></div>"
            f"<div><div class='preview-field-label'>Product</div>"
            f"<div class='preview-field-value'>{raw.get('product','—')}</div></div>"
            f"<div><div class='preview-field-label'>Sub-Product</div>"
            f"<div class='preview-field-value'>{raw.get('sub_product','—') or '—'}</div></div>"
            f"<div><div class='preview-field-label'>Issue</div>"
            f"<div class='preview-field-value'>{raw.get('issue','—')}</div></div>"
            f"<div><div class='preview-field-label'>State</div>"
            f"<div class='preview-field-value'>{raw.get('state','—')}</div></div>"
            f"<div><div class='preview-field-label'>Submitted Via</div>"
            f"<div class='preview-field-value'>{raw.get('submitted_via','—')}</div></div>"
            f"<div><div class='preview-field-label'>Date Received</div>"
            f"<div class='preview-field-value'>{raw.get('date_received','—')}</div></div>"
            f"<div><div class='preview-field-label'>Tags</div>"
            f"<div class='preview-field-value'>{raw.get('tags','—') or '—'}</div></div>"
            f"</div>"
            f"<div class='preview-field-label' style='margin-top:8px'>Complaint Narrative</div>"
            f"<div class='preview-narrative'>{raw.get('narrative','—')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── ③ Routing outcome ─────────────────────────────────────
        st.markdown(
            "<div class='admin-section-label'>③ Classification &amp; Routing</div>",
            unsafe_allow_html=True,
        )

        team_desc = {
            "credit-ops":       "Handles credit card billing, limit changes, and statement disputes.",
            "fraud":            "Investigates unauthorized transactions, identity theft, and account takeover.",
            "legal":            "Manages regulatory violations, litigation threats, and CFPB escalations.",
            "customer-service": "Resolves general inquiries, account access issues, and minor complaints.",
            "compliance":       "Reviews FCRA, FDCPA, ECOA, UDAP violations and broader regulatory risk.",
            "collections":      "Handles debt validation, collection disputes, and payment plans.",
            "mortgage-ops":     "Manages mortgage servicing, escrow issues, foreclosure, and loan modifications.",
            "tech-support":     "Resolves app issues, online banking errors, and payment processing failures.",
        }.get(team, "")

        left_col, right_col = st.columns([3, 2])
        with left_col:
            st.markdown(
                f"<div class='routing-outcome-card'>"
                f"<div class='routing-sublabel'>Assigned Team</div>"
                f"<div class='routing-team-name'>{team}</div>"
                f"<div style='font-size:0.93rem;color:#94a3b8;margin-bottom:20px;line-height:1.5'>"
                f"{team_desc}</div>"
                f"<div style='font-family:IBM Plex Mono;font-size:0.7rem;"
                f"text-transform:uppercase;letter-spacing:0.12em;color:#334155;"
                f"margin-bottom:4px'>Routing Confidence</div>"
                f"<div class='confidence-track'>"
                f"<div class='confidence-fill' style='width:{score_pct}%'></div></div>"
                f"<div style='font-family:IBM Plex Mono;font-size:0.78rem;color:#60a5fa'>"
                f"{score:.2f} ({score_pct}%)</div>"
                f"<span class='path-badge {path_class}'>{path_label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with right_col:
            product_cls = result.get("classified_product", "—")
            issue_type  = result.get("issue_type", "—")
            risk        = result.get("compliance_risk", "—")
            for label, val, color in [
                ("Classified Product", product_cls, "#7dd3fc"),
                ("Issue Type",         issue_type,  "#7dd3fc"),
                ("Severity",           severity,    {
                    "critical": "#fca5a5", "high": "#fdba74",
                    "medium":   "#fde047", "low":  "#86efac"
                }.get(severity, "#94a3b8")),
                ("Compliance Risk",    risk,        "#94a3b8"),
            ]:
                st.markdown(
                    f"<div style='background:#0d1117;border:1px solid #1e2d45;"
                    f"border-radius:6px;padding:14px 16px;margin-bottom:10px'>"
                    f"<div style='font-family:IBM Plex Mono;font-size:0.62rem;"
                    f"text-transform:uppercase;letter-spacing:0.15em;color:#334155;"
                    f"margin-bottom:4px'>{label}</div>"
                    f"<div style='font-family:IBM Plex Mono;font-size:0.9rem;"
                    f"font-weight:600;color:{color}'>{val}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── ④ Quality metrics ─────────────────────────────────────
        st.markdown(
            "<div class='admin-section-label'>④ Quality Metrics</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        for col, val, label in [
            (m1, metrics['quality_score'],                        "Quality Score"),
            (m2, f"{metrics['predicted_csat']} / 5.0",           "Predicted CSAT"),
            (m3, metrics['churn_risk_label'],                     "Churn Risk"),
            (m4, "⚠ YES" if metrics['fairness_flag'] else "✓ OK","Fairness Flag"),
            (m5, f"{result.get('processing_time_ms','—')}ms",    "Process Time"),
        ]:
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div style='font-family:IBM Plex Mono;font-size:1.1rem;"
                    f"font-weight:600;color:#60a5fa'>{val}</div>"
                    f"<div class='metric-label'>{label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr class='full-divider'>", unsafe_allow_html=True)

        # ── ⑤ Resolution reports (2-col grid) ────────────────────
        st.markdown(
            "<div class='admin-section-label'>⑤ Resolution Reports</div>",
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        with left:
            for title, key in [
                ("Resolution Plan",     "resolution_plan"),
                ("Compliance Findings", "compliance_findings"),
            ]:
                st.markdown(
                    f"<div class='report-section'>"
                    f"<div class='report-section-title'>{title}</div>"
                    f"<div class='resolution-text'>{result.get(key,'—')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        with right:
            for title, key in [
                ("Customer Response Letter", "customer_response"),
                ("Explainability Report",    "explainability_report"),
            ]:
                st.markdown(
                    f"<div class='report-section'>"
                    f"<div class='report-section-title'>{title}</div>"
                    f"<div class='resolution-text'>{result.get(key,'—')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr class='full-divider'>", unsafe_allow_html=True)

        # ── ⑥ Swarm Watch (manual trigger) ───────────────────────
        st.markdown(
            "<div class='admin-section-label'>⑥ Swarm Watch</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='swarm-card'>"
            "<div class='swarm-card-title'>⚡ Swarm Watch</div>"
            "<div class='swarm-card-body'>"
            "Manually triggered after the pipeline completes. Embeds this complaint's narrative "
            "using Voyage Finance-2 and searches Qdrant for past complaints with a cosine similarity "
            "above 0.82. If 5 or more similar past complaints are found, a systemic cluster alert "
            "is generated — covering the pattern detected, the company affected, the regulatory "
            "risk the cluster represents, and the recommended immediate action. Each processed "
            "complaint is stored in Qdrant so it can be matched against future complaints, "
            "building a growing memory of complaint clusters over time."
            "</div></div>",
            unsafe_allow_html=True,
        )

        col_swarm_btn, _ = st.columns([1, 3])
        with col_swarm_btn:
            swarm_btn = st.button("🔍  Run Swarm Watch", key="swarm_btn")

        if swarm_btn:
            st.session_state.admin_swarm_result = None
            with st.spinner("Scanning vector database for complaint clusters..."):
                try:
                    swarm_state = detect_swarm(st.session_state.admin_result)
                    st.session_state.admin_swarm_result = swarm_state
                except Exception as e:
                    st.error(f"Swarm watch error: {e}")

        if st.session_state.admin_swarm_result:
            sw = st.session_state.admin_swarm_result
            if sw.get("swarm_alert"):
                st.markdown(
                    f"<div class='swarm-alert'>"
                    f"⚡ <strong>SWARM DETECTED</strong> — Cluster: {sw.get('cluster_id','—')}"
                    f"<br><br>{sw['swarm_alert']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='background:#0d1117;border:1px solid #1e2d45;border-radius:8px;"
                    "padding:14px 18px;font-size:0.93rem;color:#94a3b8'>"
                    "✓ No significant complaint cluster detected for this type. "
                    "Fewer than 5 similar complaints found in the database."
                    "</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<hr class='full-divider'>", unsafe_allow_html=True)

        # ── ⑦ Innovation notes ────────────────────────────────────
        st.markdown(
            "<div class='admin-section-label'>⑦ Active Innovations</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='innovation-card'>"
            "<div class='innovation-title'>🧠 Self-Improving Loop</div>"
            "<div class='innovation-body'>"
            "Every complaint that gets fully resolved is stored as a vector embedding in Qdrant. "
            "The next time a similar complaint comes in, the top 3 most similar past resolutions "
            "are retrieved and injected into every agent's prompt as few-shot examples. "
            "This means each agent — classifier, compliance, remediation — has real precedents "
            "to work from, and the system gets measurably better the more complaints it processes. "
            "<br><br>"
            "<strong style='color:#7dd3fc'>Requires:</strong> "
            "<span style='color:#94a3b8;font-size:0.93rem'>Voyage AI API key + Qdrant Cloud collection.</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='innovation-card'>"
            "<div class='innovation-title'>📈 Adaptive Router</div>"
            "<div class='innovation-body'>"
            "Every time a complaint is resolved, the outcome — which team handled it and how "
            "long it took — is stored in the local SQLite database. The router then queries "
            "this history when assigning future complaints, preferring teams that have resolved "
            "similar complaint types faster in the past. Over time this creates a feedback loop "
            "where routing decisions are grounded in actual team performance data, not just "
            "the LLM's best guess."
            "<br><br>"
            "<strong style='color:#7dd3fc'>Requires:</strong> "
            "<span style='color:#94a3b8;font-size:0.93rem'>Only local SQLite — no external dependencies.</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            "<div style='text-align:center;padding:80px 0;color:#1e2d45;"
            "font-family:IBM Plex Mono;font-size:0.9rem'>"
            "Select a product category and click ▶ Sample &amp; Run Full Pipeline"
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════
# TAB 1 — Dashboard  (always fresh on app start)
# ══════════════════════════════════════════════════════════════════
with tab_dashboard:
    # Auto-load fresh data on every app start via a run-once flag
    if "dashboard_loaded" not in st.session_state:
        st.session_state.dashboard_loaded = True
        st.rerun()

    if st.button("↻  Refresh Dashboard"):
        st.rerun()

    agg = get_aggregate_metrics()

    if "error" in agg:
        st.markdown(
            "<div style='text-align:center;padding:80px 0;color:#1e2d45;"
            "font-family:IBM Plex Mono;font-size:0.9rem'>"
            "No data yet — process some complaints first"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, label in [
            (c1, agg.get("total_complaints", 0),         "Total Processed"),
            (c2, f"{agg.get('avg_processing_ms', 0)}ms", "Avg Process Time"),
            (c3, agg.get("avg_quality_score", 0),        "Avg Quality Score"),
            (c4, agg.get("fairness_flags", 0),           "Fairness Flags"),
            (c5, agg.get("swarm_detections", 0),         "Swarm Detections"),
        ]:
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value'>{val}</div>"
                    f"<div class='metric-label'>{label}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        col_sev, col_team = st.columns(2)

        with col_sev:
            st.markdown("### Severity Distribution")
            severity_dist   = agg.get("severity_distribution", {})
            severity_colors = {
                "critical": "#ef4444", "high": "#f97316",
                "medium": "#eab308",   "low":  "#22c55e",
            }
            total = sum(severity_dist.values()) or 1
            for sev in ["critical", "high", "medium", "low"]:
                count = severity_dist.get(sev, 0)
                pct   = int((count / total) * 100)
                color = severity_colors[sev]
                st.markdown(
                    f"<div style='margin-bottom:12px'>"
                    f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
                    f"<span style='font-family:IBM Plex Mono;font-size:0.8rem;"
                    f"text-transform:uppercase;color:#94a3b8'>{sev}</span>"
                    f"<span style='font-family:IBM Plex Mono;font-size:0.93rem;color:#94a3b8'>"
                    f"{count} ({pct}%)</span></div>"
                    f"<div style='background:#1e2d45;border-radius:4px;height:8px'>"
                    f"<div style='width:{pct}%;height:8px;border-radius:4px;background:{color}'>"
                    f"</div></div></div>",
                    unsafe_allow_html=True,
                )

        with col_team:
            st.markdown("### Team Avg Resolution Time")
            team_perf = agg.get("team_avg_resolution_ms", {})
            if team_perf:
                max_ms = max(team_perf.values()) or 1
                for team, avg_ms in sorted(team_perf.items(), key=lambda x: x[1]):
                    pct = int((avg_ms / max_ms) * 100)
                    st.markdown(
                        f"<div style='margin-bottom:12px'>"
                        f"<div style='display:flex;justify-content:space-between;margin-bottom:4px'>"
                        f"<span style='font-family:IBM Plex Mono;font-size:0.8rem;color:#94a3b8'>{team}</span>"
                        f"<span style='font-family:IBM Plex Mono;font-size:0.93rem;color:#94a3b8'>{int(avg_ms)}ms</span>"
                        f"</div>"
                        f"<div style='background:#1e2d45;border-radius:4px;height:8px'>"
                        f"<div style='width:{pct}%;height:8px;border-radius:4px;background:#3b82f6'></div>"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        st.markdown(
            f"<span style='font-family:IBM Plex Mono;font-size:0.75rem;color:#1e2d45'>"
            f"Avg few-shot examples used per complaint: {agg.get('avg_few_shots_used', 0)}"
            f"</span>",
            unsafe_allow_html=True,
        )