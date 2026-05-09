# ⚖️ CFPB Complaint Resolution System

An AI-powered multi-agent pipeline that automatically triages, classifies, routes, and resolves consumer financial complaints from the CFPB (Consumer Financial Protection Bureau) public database.

Built with **LangGraph**, **Groq (LLaMA 3.3 70B)**, **Voyage AI**, and **Qdrant** — deployed on **AWS EC2** via Docker.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1.10-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3%2070B-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![AWS](https://img.shields.io/badge/AWS-EC2-yellow)

---

## 🧠 What It Does

Takes a raw consumer complaint narrative and runs it through a full resolution pipeline:

```
Complaint Ingested
       ↓
Few-Shot Injection (past similar resolutions retrieved from Qdrant)
       ↓
Classification (product, issue type, severity, compliance risk)
       ↓
Routing (assigned to specialist team based on historical performance)
       ↓
        ├── Critical/Regulatory → Compliance Agent
        └── Standard → Root Cause Agent
                            ↓
                     Remediation Agent
                            ↓
                   Response Draft Agent
                            ↓
                       Synthesiser
                            ↓
              Resolution Plan + Explainability Report
                            ↓
                 Metrics Stored → Routing Memory Updated
```

---

## 📊 Evaluation Results

Measured across **14 real CFPB complaints** spanning Credit Card, Mortgage, and Debt Collection categories using live API data.

### Aggregate Metrics

| Metric | Value |
|---|---|
| Avg Resolution Quality Score | **0.827 / 1.0** |
| Avg Pipeline Latency | **~8.8 seconds** |
| Fairness Flags | **0 / 14** |
| Swarm Detections | 0 (requires volume) |
| Avg Processing Time | 8,934ms |

### Quality Score Breakdown

The resolution quality score is computed by an LLM-as-judge rubric across 4 dimensions (each 0.0–0.25):

| Dimension | What It Measures |
|---|---|
| Completeness | Does the resolution address all aspects of the complaint? |
| Regulatory Compliance | Are all applicable regulations addressed? |
| Remediation Quality | Are steps specific, actionable, and time-bound? |
| Customer Response | Is the letter empathetic, clear, and professional? |

Scores ranged from **0.75 to 0.95** across all complaints, with the highest score (0.95) on a Debt Collection/FDCPA violation — a well-defined regulatory case.

### Severity Distribution

| Severity | Count | % |
|---|---|---|
| High | 9 | 64% |
| Medium | 5 | 36% |
| Critical | 0 | 0% |
| Low | 0 | 0% |

High severity dominance is expected — CFPB complaints by nature tend to involve significant financial harm.

### Team Routing Accuracy

Each product category was correctly routed to its domain specialist with no cross-contamination:

| Product Category | Teams Assigned |
|---|---|
| Credit Card | `fraud`, `credit-ops` |
| Mortgage | `mortgage-ops`, `compliance` |
| Debt Collection | `compliance`, `legal` |

### Team Avg Resolution Time

| Team | Avg Time |
|---|---|
| compliance | 8,450ms |
| mortgage-ops | 8,621ms |
| fraud | 8,854ms |
| credit-ops | 9,678ms |

---

## 🚀 Key Features

### 1. Multi-Agent LangGraph Pipeline
7 specialist agents orchestrated by LangGraph with conditional routing — critical complaints escalate directly to compliance, standard complaints go through root cause analysis first.

### 2. Self-Improving Loop
Every resolved complaint is embedded using **Voyage Finance-2** and stored in **Qdrant**. The next similar complaint retrieves the top 3 most similar past resolutions and injects them as few-shot examples into every agent's prompt — the system gets measurably better the more complaints it processes.

### 3. Adaptive Router
Routing decisions are informed by a **SQLite memory** that tracks which team resolves each complaint type fastest. Over time the router prefers historically faster teams, creating a data-driven feedback loop.

### 4. Swarm Detection
Embeds each complaint and searches Qdrant for past complaints with cosine similarity above **0.82**. If 5 or more similar complaints are found, a systemic cluster alert is generated — covering the pattern, affected company, regulatory risk, and recommended action.

### 5. Regulatory Compliance Coverage
Automatically detects potential violations across **8 regulations**:
- FCRA — Fair Credit Reporting Act
- FDCPA — Fair Debt Collection Practices Act
- ECOA — Equal Credit Opportunity Act
- UDAP — Unfair, Deceptive, or Abusive Acts or Practices
- TILA — Truth in Lending Act
- RESPA — Real Estate Settlement Procedures Act
- EFTA — Electronic Fund Transfer Act
- GLBA — Gramm-Leach-Bliley Act

### 6. Explainability Reports
Every resolution generates a full **CFPB-auditable explainability report** justifying every AI decision — classification rationale, routing rationale, root cause, compliance assessment, and decision confidence.

### 7. Fairness Monitoring
Flags complaints where similar cases across different states or companies received inconsistent severity scores — a critical responsible AI consideration for fintech.

---

## 🗂️ Project Structure

```
├── agents/
│   ├── classifier.py        # Product, severity, compliance risk classification
│   ├── compliance.py        # Regulatory violation detection
│   ├── ingestor.py          # CFPB API data fetcher
│   ├── remediation.py       # Remediation plan generator
│   ├── response_draft.py    # Customer response letter writer
│   ├── root_cause.py        # Root cause analyzer
│   ├── router.py            # Team assignment with routing memory
│   └── synthesiser.py       # Resolution plan + explainability report
├── graph/
│   ├── edges.py             # Conditional routing logic
│   ├── graph.py             # LangGraph pipeline builder
│   └── state.py             # ComplaintState TypedDict
├── innovations/
│   ├── adaptive_router.py   # SQLite-backed routing memory
│   ├── self_improving.py    # Qdrant few-shot retrieval and storage
│   └── swarm_watcher.py     # Complaint cluster detection
├── metrics/
│   └── evaluator.py         # Quality scoring, fairness, CSAT prediction
├── app.py                   # Streamlit UI
├── main.py                  # CLI entry point
├── Dockerfile
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Embeddings | Voyage AI — voyage-finance-2 |
| Vector DB | Qdrant Cloud |
| Orchestration | LangGraph |
| UI | Streamlit |
| Routing Memory | SQLite |
| Data Source | CFPB Public Complaints API |
| Deployment | AWS EC2 (t3.small) via Docker |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- Docker
- API keys for Groq, Voyage AI, and Qdrant

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cfpb-complaint-resolution.git
cd cfpb-complaint-resolution
```

### 2. Create your `.env` file
```bash
GROQ_API_KEY=your_groq_api_key
VOYAGE_API_KEY=your_voyage_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_EXAMPLES_COLLECTION=resolution_examples
QDRANT_COMPLAINTS_COLLECTION=complaints
```

### 3. Build and run with Docker
```bash
docker build -t cfpb-app .
docker run -d --name cfpb-app --env-file .env -p 8501:8501 cfpb-app
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 💻 CLI Usage

```bash
# Run pipeline on 5 complaints
python main.py --limit 5

# Filter by product category
python main.py --limit 10 --product "Credit card"
python main.py --limit 10 --product "Mortgage"
python main.py --limit 10 --product "Debt collection"
```

---

## 🖥️ UI Overview

### Admin Panel
- Select a product category
- Sample a live complaint from the CFPB API
- Run the full 7-agent pipeline
- View classification, routing, compliance findings, remediation steps, customer response letter, and explainability report
- Manually trigger Swarm Watch to detect complaint clusters

### Dashboard
- Aggregate metrics across all processed complaints
- Severity distribution
- Team average resolution times
- Fairness flags and swarm detection counts

---

## 📝 License
MIT
