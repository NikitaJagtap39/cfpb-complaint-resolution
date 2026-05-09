# agents/classifier.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from graph.state import ComplaintState

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# ── Valid output values ───────────────────────────────────────────
VALID_PRODUCTS = [
    "credit_card",
    "mortgage",
    "debt_collection",
    "checking_savings",
    "personal_loan",
    "credit_reporting",
    "money_transfer",
    "student_loan",
    "other",
]

VALID_SEVERITIES = ["low", "medium", "high", "critical"]

VALID_RISKS = ["none", "low", "high", "regulatory"]


def build_prompt(state: ComplaintState) -> str:
    few_shots = ""
    if state.get("few_shot_examples"):
        examples = state["few_shot_examples"][:3]  # cap at 3
        few_shots = "\n\n### Past resolved complaints for reference:\n"
        for ex in examples:
            few_shots += f"""
Narrative: {ex.get('narrative', '')[:300]}
Classified product: {ex.get('classified_product', '')}
Issue type: {ex.get('issue_type', '')}
Severity: {ex.get('severity', '')}
Compliance risk: {ex.get('compliance_risk', '')}
---"""

    return f"""You are a financial complaint classification expert working for a fintech regulator.

Classify the complaint below into exactly these fields.
Respond in this exact format — no extra text, no markdown:

classified_product: <one of {VALID_PRODUCTS}>
issue_type: <short snake_case label e.g. unauthorized_charge, identity_theft, incorrect_reporting>
severity: <one of {VALID_SEVERITIES}>
compliance_risk: <one of {VALID_RISKS}>

Severity guide:
- critical: potential fraud, identity theft, illegal activity, vulnerable consumer (elderly/disabled)
- high: significant financial harm, repeated violations, UDAP concerns
- medium: billing errors, incorrect reporting, unresolved disputes
- low: general inquiries, minor inconveniences, already resolved issues

Compliance risk guide:
- regulatory: potential CFPB, FCRA, FDCPA, ECOA, or UDAP violation
- high: likely policy violation, escalation probable
- low: minor procedural issue
- none: no regulatory concern
{few_shots}

### Complaint to classify:
Product (raw): {state['product']}
Sub-product: {state.get('sub_product', 'N/A')}
Issue (raw): {state.get('issue', 'N/A')}
Company: {state['company']}
Narrative: {state['narrative'][:1500]}

Classify now:"""


def parse_response(text: str) -> dict:
    result = {
        "classified_product": "other",
        "issue_type":         "unknown",
        "severity":           "medium",
        "compliance_risk":    "low",
    }

    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key   = key.strip().lower()
        value = value.strip().lower()

        if key == "classified_product" and value in VALID_PRODUCTS:
            result["classified_product"] = value
        elif key == "issue_type" and value:
            result["issue_type"] = value.replace(" ", "_")
        elif key == "severity" and value in VALID_SEVERITIES:
            result["severity"] = value
        elif key == "compliance_risk" and value in VALID_RISKS:
            result["compliance_risk"] = value

    return result


def classify_complaint(state: ComplaintState) -> ComplaintState:
    prompt   = build_prompt(state)
    response = llm.invoke(prompt)
    parsed   = parse_response(response.content)

    return {
        **state,
        "classified_product": parsed["classified_product"],
        "issue_type":         parsed["issue_type"],
        "severity":           parsed["severity"],
        "compliance_risk":    parsed["compliance_risk"],
    }


if __name__ == "__main__":
    from agents.ingestor import fetch_complaints

    complaints = fetch_complaints(limit=3)

    for c in complaints:
        test_state: ComplaintState = {
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

        result = classify_complaint(test_state)

        print(f"ID       : {result['complaint_id']}")
        print(f"Product  : {result['classified_product']}")
        print(f"Issue    : {result['issue_type']}")
        print(f"Severity : {result['severity']}")
        print(f"Risk     : {result['compliance_risk']}")
        print()

