# agents/router.py
import os
import sqlite3
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

VALID_TEAMS = [
    "credit-ops",
    "fraud",
    "legal",
    "customer-service",
    "compliance",
    "collections",
    "mortgage-ops",
    "tech-support",
]


# ── Routing memory helpers ────────────────────────────────────────
def get_routing_scores(complaint_type: str) -> dict:
    """
    Pull average resolution times per team for this complaint type.
    Returns dict of {team: avg_resolution_ms} or empty if no history.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT assigned_team, AVG(resolution_time_ms)
            FROM routing_outcomes
            WHERE complaint_type = ?
            GROUP BY assigned_team
            ORDER BY AVG(resolution_time_ms) ASC
        """, (complaint_type,))
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except Exception:
        return {}


def build_prompt(state: ComplaintState, scores: dict) -> str:
    history_context = ""
    if scores:
        history_context = "\n### Historical routing performance for this complaint type:\n"
        for team, avg_ms in scores.items():
            history_context += f"- {team}: avg resolution {int(avg_ms)}ms\n"
        history_context += "Prefer faster teams when capability is equal.\n"

    return f"""You are a complaint routing specialist at a fintech company.

Assign this complaint to exactly one internal team.
Respond in this exact format — no extra text, no markdown:

assigned_team: <one of {VALID_TEAMS}>
routing_score: <confidence float between 0.0 and 1.0>
routing_reason: <one sentence explaining why>

Team responsibilities:
- credit-ops: credit card billing, limit changes, statement disputes
- fraud: unauthorized transactions, identity theft, account takeover
- legal: regulatory violations, litigation threats, CFPB escalations
- customer-service: general inquiries, account access, minor complaints
- compliance: FCRA, FDCPA, ECOA, UDAP violations, regulatory risk
- collections: debt validation, collection disputes, payment plans
- mortgage-ops: mortgage servicing, escrow, foreclosure, loan modifications
- tech-support: app issues, online banking errors, payment processing failures
{history_context}
### Complaint details:
Product:          {state['classified_product']}
Issue type:       {state['issue_type']}
Severity:         {state['severity']}
Compliance risk:  {state['compliance_risk']}
Company:          {state['company']}
Narrative:        {state['narrative'][:800]}

Assign now:"""


def parse_response(text: str) -> dict:
    result = {
        "assigned_team":  "customer-service",
        "routing_score":  0.5,
        "routing_reason": "",
    }

    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key   = key.strip().lower()
        value = value.strip()

        if key == "assigned_team" and value.lower() in VALID_TEAMS:
            result["assigned_team"] = value.lower()
        elif key == "routing_score":
            try:
                score = float(value)
                result["routing_score"] = max(0.0, min(1.0, score))
            except ValueError:
                pass
        elif key == "routing_reason":
            result["routing_reason"] = value

    return result


def route_complaint(state: ComplaintState) -> ComplaintState:
    complaint_type = f"{state['classified_product']}_{state['issue_type']}"
    scores         = get_routing_scores(complaint_type)
    prompt         = build_prompt(state, scores)
    response       = llm.invoke(prompt)
    parsed         = parse_response(response.content)

    return {
        **state,
        "assigned_team":  parsed["assigned_team"],
        "routing_score":  parsed["routing_score"],
    }


if __name__ == "__main__":
    from agents.ingestor import fetch_complaints
    from agents.classifier import classify_complaint

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

        state  = classify_complaint(state)
        result = route_complaint(state)

        print(f"ID       : {result['complaint_id']}")
        print(f"Product  : {result['classified_product']}")
        print(f"Issue    : {result['issue_type']}")
        print(f"Severity : {result['severity']}")
        print(f"Team     : {result['assigned_team']}")
        print(f"Score    : {result['routing_score']}")
        print()

