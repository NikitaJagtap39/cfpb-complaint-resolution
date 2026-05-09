# agents/remediation.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from graph.state import ComplaintState

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)


def build_prompt(state: ComplaintState) -> str:
    few_shots = ""
    if state.get("few_shot_examples"):
        examples = state["few_shot_examples"][:2]
        few_shots = "\n\n### Examples of past remediation plans:\n"
        for ex in examples:
            few_shots += f"""
Narrative: {ex.get('narrative', '')[:200]}
Remediation steps: {ex.get('remediation_steps', '')}
---"""

    return f"""You are a financial services operations manager responsible for resolving consumer complaints.

Generate a complete remediation plan for this complaint.

Respond in this exact format — no extra text, no markdown:

immediate_actions: <comma separated list of actions to take within 24 hours>
short_term_actions: <comma separated list of actions to take within 30 days>
long_term_actions: <comma separated list of preventive actions to take within 90 days>
compensation_recommended: <yes | no>
compensation_type: <one of: none | fee_waiver | account_credit | rate_reduction | formal_apology | monetary_refund>
estimated_resolution_days: <integer number of days>
escalation_required: <yes | no>
escalation_reason: <one sentence or NONE>
preventive_recommendations: <comma separated list of process improvements to prevent recurrence>

Remediation guidelines:
- immediate actions must address the consumer's core complaint directly
- short term actions must fix the underlying process or system failure
- long term actions must prevent recurrence at scale
- compensation should match severity — critical always gets monetary consideration
- estimated resolution must account for regulatory response timeframes
{few_shots}

### Complaint details:
Complaint ID:          {state['complaint_id']}
Product:               {state['classified_product']}
Issue type:            {state['issue_type']}
Severity:              {state['severity']}
Compliance risk:       {state['compliance_risk']}
Root cause:            {state.get('root_cause', 'N/A')}
Compliance findings:   {state.get('compliance_findings', 'N/A')}
Assigned team:         {state['assigned_team']}
Company:               {state['company']}
Narrative:             {state['narrative'][:1200]}

Generate remediation plan now:"""


def parse_response(text: str) -> dict:
    result = {
        "immediate_actions":          "",
        "short_term_actions":         "",
        "long_term_actions":          "",
        "compensation_recommended":   "no",
        "compensation_type":          "none",
        "estimated_resolution_days":  30,
        "escalation_required":        "no",
        "escalation_reason":          "NONE",
        "preventive_recommendations": "",
    }

    valid_compensation = {
        "none", "fee_waiver", "account_credit",
        "rate_reduction", "formal_apology", "monetary_refund"
    }

    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key   = key.strip().lower()
        value = value.strip()

        if key == "immediate_actions" and value:
            result["immediate_actions"] = value
        elif key == "short_term_actions" and value:
            result["short_term_actions"] = value
        elif key == "long_term_actions" and value:
            result["long_term_actions"] = value
        elif key == "compensation_recommended" and value.lower() in ("yes", "no"):
            result["compensation_recommended"] = value.lower()
        elif key == "compensation_type" and value.lower() in valid_compensation:
            result["compensation_type"] = value.lower()
        elif key == "estimated_resolution_days":
            try:
                result["estimated_resolution_days"] = int(value)
            except ValueError:
                pass
        elif key == "escalation_required" and value.lower() in ("yes", "no"):
            result["escalation_required"] = value.lower()
        elif key == "escalation_reason":
            result["escalation_reason"] = value
        elif key == "preventive_recommendations" and value:
            result["preventive_recommendations"] = value

    # build remediation_steps list for state
    steps = []
    if result["immediate_actions"]:
        for s in result["immediate_actions"].split(","):
            steps.append(f"[IMMEDIATE] {s.strip()}")
    if result["short_term_actions"]:
        for s in result["short_term_actions"].split(","):
            steps.append(f"[30 DAYS] {s.strip()}")
    if result["long_term_actions"]:
        for s in result["long_term_actions"].split(","):
            steps.append(f"[90 DAYS] {s.strip()}")
    if result["preventive_recommendations"]:
        for s in result["preventive_recommendations"].split(","):
            steps.append(f"[PREVENTIVE] {s.strip()}")

    return {**result, "remediation_steps": steps}


def generate_remediation(state: ComplaintState) -> ComplaintState:
    prompt   = build_prompt(state)
    response = llm.invoke(prompt)
    parsed   = parse_response(response.content)

    return {
        **state,
        "remediation_steps": parsed["remediation_steps"],
    }


if __name__ == "__main__":
    from agents.ingestor import fetch_complaints
    from agents.classifier import classify_complaint
    from agents.router import route_complaint
    from agents.root_cause import analyze_root_cause
    from agents.compliance import check_compliance

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
        state  = route_complaint(state)
        state  = analyze_root_cause(state)
        state  = check_compliance(state)
        result = generate_remediation(state)

        print(f"ID    : {result['complaint_id']}")
        print(f"Steps :")
        for step in result["remediation_steps"]:
            print(f"  {step}")
        print()

