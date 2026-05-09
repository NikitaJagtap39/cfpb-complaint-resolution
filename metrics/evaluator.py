# metrics/evaluator.py
import os
import json
import sqlite3
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from graph.state import ComplaintState

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

DB_PATH = "memory/routing_memory.db"


# ── DB setup ──────────────────────────────────────────────────────
def init_metrics_db():
    os.makedirs("memory", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS complaint_metrics (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id            TEXT NOT NULL,
            classified_product      TEXT,
            issue_type              TEXT,
            severity                TEXT,
            compliance_risk         TEXT,
            assigned_team           TEXT,
            resolution_status       TEXT,
            processing_time_ms      INTEGER,
            resolution_quality_score REAL,
            fairness_flag           INTEGER DEFAULT 0,
            swarm_detected          INTEGER DEFAULT 0,
            few_shots_used          INTEGER DEFAULT 0,
            created_at              DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def store_metrics(state: ComplaintState, quality_score: float, fairness_flag: bool):
    init_metrics_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO complaint_metrics (
            complaint_id, classified_product, issue_type,
            severity, compliance_risk, assigned_team,
            resolution_status, processing_time_ms,
            resolution_quality_score, fairness_flag,
            swarm_detected, few_shots_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        state.get("complaint_id", ""),
        state.get("classified_product", ""),
        state.get("issue_type", ""),
        state.get("severity", ""),
        state.get("compliance_risk", ""),
        state.get("assigned_team", ""),
        "resolved",
        state.get("processing_time_ms", 0),
        quality_score,
        int(fairness_flag),
        int(bool(state.get("swarm_alert"))),
        len(state.get("few_shot_examples", [])),
    ))
    conn.commit()
    conn.close()


# ── Metric 1: Classification accuracy ────────────────────────────
def compute_classification_f1(
    predictions: list[dict],
    ground_truth: list[dict],
) -> dict:
    """
    Computes per-category F1 for classified_product and severity.
    predictions and ground_truth are lists of dicts with those fields.
    """
    from collections import defaultdict

    def f1_per_class(preds, truths, field):
        classes = set(truths)
        results = {}
        for cls in classes:
            tp = sum(1 for p, t in zip(preds, truths) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(preds, truths) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(preds, truths) if p != cls and t == cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )
            results[cls] = {
                "precision": round(precision, 3),
                "recall":    round(recall, 3),
                "f1":        round(f1, 3),
            }
        macro_f1 = sum(v["f1"] for v in results.values()) / len(results) if results else 0.0
        return results, round(macro_f1, 3)

    pred_products  = [p.get("classified_product", "") for p in predictions]
    true_products  = [t.get("classified_product", "") for t in ground_truth]
    pred_severities = [p.get("severity", "") for p in predictions]
    true_severities = [t.get("severity", "") for t in ground_truth]

    product_f1,  product_macro  = f1_per_class(pred_products,  true_products,  "product")
    severity_f1, severity_macro = f1_per_class(pred_severities, true_severities, "severity")

    return {
        "product_f1":       product_f1,
        "product_macro_f1": product_macro,
        "severity_f1":      severity_f1,
        "severity_macro_f1": severity_macro,
    }


# ── Metric 2: Resolution quality ─────────────────────────────────
def score_resolution_quality(state: ComplaintState) -> float:
    """
    Uses Groq as a rubric judge to score the resolution 0.0-1.0.
    Evaluates resolution plan, customer response, and remediation steps.
    """
    remediation_summary = "\n".join(
        f"  - {step}" for step in state.get("remediation_steps", [])[:5]
    )

    prompt = f"""You are a quality assurance auditor evaluating a complaint resolution.
Score this resolution on a scale of 0.0 to 1.0 based on these criteria:

1. Completeness — does the resolution address all aspects of the complaint? (0.0-0.25)
2. Regulatory compliance — are all applicable regulations addressed? (0.0-0.25)
3. Remediation quality — are the steps specific, actionable, and time-bound? (0.0-0.25)
4. Customer response quality — is it empathetic, clear, and professional? (0.0-0.25)

Respond in this exact format — no extra text:
completeness_score: <float 0.0-0.25>
compliance_score: <float 0.0-0.25>
remediation_score: <float 0.0-0.25>
response_score: <float 0.0-0.25>
total_score: <float 0.0-1.0>
one_line_feedback: <single sentence>

### Resolution to evaluate:
Complaint narrative:   {state['narrative'][:500]}
Classified product:    {state.get('classified_product', 'N/A')}
Severity:              {state.get('severity', 'N/A')}
Compliance findings:   {state.get('compliance_findings', 'N/A')[:300]}
Remediation steps:
{remediation_summary}
Customer response:     {state.get('customer_response', 'N/A')[:400]}

Score now:"""

    response = llm.invoke(prompt)
    total    = 0.5  # default if parsing fails

    for line in response.content.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        if key.strip().lower() == "total_score":
            try:
                total = max(0.0, min(1.0, float(value.strip())))
            except ValueError:
                pass

    return round(total, 3)


# ── Metric 3: Fairness ────────────────────────────────────────────
def check_fairness(state: ComplaintState) -> bool:
    """
    Flags a complaint if similar complaints from different states
    received significantly different severity scores.
    Returns True if a fairness concern is detected.
    """
    init_metrics_db()
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT severity, COUNT(*) as cnt
        FROM complaint_metrics
        WHERE classified_product = ?
        AND issue_type = ?
        GROUP BY severity
    """, (
        state.get("classified_product", ""),
        state.get("issue_type", ""),
    ))
    rows = cursor.fetchall()
    conn.close()

    if not rows or len(rows) < 2:
        return False

    severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    counts = {row[0]: row[1] for row in rows}
    total  = sum(counts.values())

    # flag if one severity bucket holds more than 80% of similar complaints
    # but current complaint is in a minority bucket — possible inconsistency
    current_severity = state.get("severity", "medium")
    current_count    = counts.get(current_severity, 0)
    dominant_share   = max(counts.values()) / total if total > 0 else 0

    if dominant_share > 0.8 and current_count / total < 0.1:
        return True

    return False


# ── Metric 4: CSAT impact ─────────────────────────────────────────
def predict_csat_impact(state: ComplaintState) -> dict:
    """
    Predicts customer satisfaction impact and churn risk
    based on severity, resolution quality, and complaint type.
    """
    severity_weights = {
        "low":      0.1,
        "medium":   0.3,
        "high":     0.6,
        "critical": 0.9,
    }

    base_churn_risk = severity_weights.get(
        state.get("severity", "medium"), 0.3
    )

    # compliance issues increase churn risk
    compliance_risk = state.get("compliance_risk", "none")
    if compliance_risk == "regulatory":
        base_churn_risk = min(1.0, base_churn_risk + 0.2)
    elif compliance_risk == "high":
        base_churn_risk = min(1.0, base_churn_risk + 0.1)

    # swarm detection means systemic issue — higher churn risk
    if state.get("swarm_alert"):
        base_churn_risk = min(1.0, base_churn_risk + 0.15)

    # few-shot examples used means better resolution — lower churn risk
    if state.get("few_shot_examples"):
        base_churn_risk = max(0.0, base_churn_risk - 0.05)

    predicted_csat = round(max(1.0, 5.0 - (base_churn_risk * 4.0)), 2)

    return {
        "predicted_churn_risk":   round(base_churn_risk, 3),
        "predicted_csat_score":   predicted_csat,
        "churn_risk_label":       (
            "critical" if base_churn_risk > 0.7 else
            "high"     if base_churn_risk > 0.5 else
            "medium"   if base_churn_risk > 0.3 else
            "low"
        ),
    }


# ── Aggregate metrics dashboard ───────────────────────────────────
def get_aggregate_metrics() -> dict:
    init_metrics_db()
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*)                            as total_complaints,
            AVG(processing_time_ms)             as avg_processing_ms,
            AVG(resolution_quality_score)       as avg_quality_score,
            SUM(fairness_flag)                  as fairness_flags,
            SUM(swarm_detected)                 as swarm_detections,
            AVG(few_shots_used)                 as avg_few_shots
        FROM complaint_metrics
    """)
    row = cursor.fetchone()

    cursor.execute("""
        SELECT severity, COUNT(*) as cnt
        FROM complaint_metrics
        GROUP BY severity
    """)
    severity_dist = {r[0]: r[1] for r in cursor.fetchall()}

    cursor.execute("""
        SELECT assigned_team, AVG(processing_time_ms) as avg_ms
        FROM complaint_metrics
        GROUP BY assigned_team
        ORDER BY avg_ms ASC
    """)
    team_performance = {r[0]: round(r[1], 0) for r in cursor.fetchall()}

    conn.close()

    if not row or row[0] == 0:
        return {"error": "No metrics data yet — process some complaints first"}

    return {
        "total_complaints":    row[0],
        "avg_processing_ms":   round(row[1] or 0, 0),
        "avg_quality_score":   round(row[2] or 0, 3),
        "fairness_flags":      row[3],
        "swarm_detections":    row[4],
        "avg_few_shots_used":  round(row[5] or 0, 2),
        "severity_distribution": severity_dist,
        "team_avg_resolution_ms": team_performance,
    }


# ── Main evaluator node ───────────────────────────────────────────
def evaluate(state: ComplaintState) -> dict:
    quality_score = score_resolution_quality(state)
    fairness_flag = check_fairness(state)
    csat          = predict_csat_impact(state)

    store_metrics(state, quality_score, fairness_flag)

    return {
        "quality_score":      quality_score,
        "fairness_flag":      fairness_flag,
        "predicted_csat":     csat["predicted_csat_score"],
        "churn_risk":         csat["predicted_churn_risk"],
        "churn_risk_label":   csat["churn_risk_label"],
    }


if __name__ == "__main__":
    from agents.ingestor import fetch_complaints
    from agents.classifier import classify_complaint
    from agents.router import route_complaint
    from agents.root_cause import analyze_root_cause
    from agents.compliance import check_compliance
    from agents.remediation import generate_remediation
    from agents.response_draft import draft_response
    from agents.synthesiser import synthesise
    import time

    init_metrics_db()
    complaints = fetch_complaints(limit=3)

    for c in complaints:
        state: ComplaintState = {
            "complaint_id":        c["complaint_id"],
            "narrative":           c["narrative"],
            "product":             c["product"],
            "sub_product":         c["sub_product"],
            "issue":               c["issue"],
            "sub_issue":           c["sub_issue"],
            "company":             c["company"],
            "state":               c["state"],
            "submitted_via":       c["submitted_via"],
            "date_received":       c["date_received"],
            "tags":                c["tags"],
            "classified_product":  "",
            "issue_type":          "",
            "severity":            "",
            "compliance_risk":     "",
            "assigned_team":       "",
            "routing_score":       0.0,
            "root_cause":          "",
            "compliance_findings": "",
            "remediation_steps":   [],
            "customer_response":   "",
            "resolution_plan":     "",
            "explainability_report": "",
            "swarm_alert":         None,
            "cluster_id":          None,
            "few_shot_examples":   [],
            "processing_time_ms":  0,
            "error":               None,
        }

        start  = time.time()
        state  = classify_complaint(state)
        state  = route_complaint(state)
        state  = analyze_root_cause(state)
        state  = check_compliance(state)
        state  = generate_remediation(state)
        state  = draft_response(state)
        state  = synthesise(state)
        elapsed = int((time.time() - start) * 1000)
        state["processing_time_ms"] = elapsed

        metrics = evaluate(state)

        print(f"ID            : {state['complaint_id']}")
        print(f"Quality score : {metrics['quality_score']}")
        print(f"Fairness flag : {metrics['fairness_flag']}")
        print(f"CSAT predicted: {metrics['predicted_csat']}")
        print(f"Churn risk    : {metrics['churn_risk_label']} ({metrics['churn_risk']})")
        print()

    print("=== Aggregate Metrics ===")
    agg = get_aggregate_metrics()
    for k, v in agg.items():
        print(f"{k}: {v}")

