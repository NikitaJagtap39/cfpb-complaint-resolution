# agents/root_cause.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from graph.state import ComplaintState

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)


def build_prompt(state: ComplaintState) -> str:
    few_shots = ""
    if state.get("few_shot_examples"):
        examples = state["few_shot_examples"][:2]
        few_shots = "\n\n### Examples of past root cause analyses:\n"
        for ex in examples:
            few_shots += f"""
Narrative: {ex.get('narrative', '')[:300]}
Root cause: {ex.get('root_cause', '')}
---"""

    return f"""You are a senior financial services operations analyst specializing in consumer complaint root cause analysis.

Analyze the complaint below and identify the precise root cause of the issue.

Respond in this exact format — no extra text, no markdown:

root_cause: <one concise sentence identifying the specific process, system, or policy failure>
root_cause_category: <one of: process_failure | system_error | policy_gap | human_error | third_party_failure | regulatory_breach>
contributing_factors: <comma separated list of up to 3 contributing factors>
internal_department: <the internal department most responsible e.g. credit-ops, fraud-prevention, loan-servicing>
recurrence_risk: <one of: low | medium | high>

Root cause category guide:
- process_failure: broken or missing internal process
- system_error: technical bug, outage, or data issue
- policy_gap: policy does not cover this scenario or is ambiguous
- human_error: agent or employee mistake
- third_party_failure: vendor, bureau, or partner failure
- regulatory_breach: direct violation of a law or regulation
{few_shots}

### Complaint details:
Complaint ID:     {state['complaint_id']}
Product:          {state['classified_product']}
Issue type:       {state['issue_type']}
Severity:         {state['severity']}
Compliance risk:  {state['compliance_risk']}
Assigned team:    {state['assigned_team']}
Company:          {state['company']}
Narrative:        {state['narrative'][:1500]}

Analyze now:"""


def parse_response(text: str) -> dict:
    result = {
        "root_cause":           "",
        "root_cause_category":  "process_failure",
        "contributing_factors": "",
        "internal_department":  "",
        "recurrence_risk":      "medium",
    }

    valid_categories = {
        "process_failure", "system_error", "policy_gap",
        "human_error", "third_party_failure", "regulatory_breach"
    }
    valid_risks = {"low", "medium", "high"}

    known_keys = {
        "root_cause", "root_cause_category",
        "contributing_factors", "internal_department", "recurrence_risk",
    }

    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        parts = line.split(":", 1)          # split on FIRST colon only so values containing colons are preserved
        key   = parts[0].strip().lower().replace(" ", "_")
        value = parts[1].strip() if len(parts) > 1 else ""

        if key not in known_keys or not value:
            continue

        if key == "root_cause":
            result["root_cause"] = value
        elif key == "root_cause_category" and value.lower() in valid_categories:
            result["root_cause_category"] = value.lower()
        elif key == "contributing_factors":
            result["contributing_factors"] = value
        elif key == "internal_department":
            result["internal_department"] = value.lower()
        elif key == "recurrence_risk" and value.lower() in valid_risks:
            result["recurrence_risk"] = value.lower()

    # combine into a structured root_cause string for the synthesiser
    full_root_cause = (
        f"{result['root_cause']} "
        f"[Category: {result['root_cause_category']}] "
        f"[Contributing factors: {result['contributing_factors']}] "
        f"[Department: {result['internal_department']}] "
        f"[Recurrence risk: {result['recurrence_risk']}]"
    )

    return {**result, "full_root_cause": full_root_cause}


def analyze_root_cause(state: ComplaintState) -> ComplaintState:
    prompt   = build_prompt(state)
    response = llm.invoke(prompt)
    raw_text = response.content.strip()
    parsed   = parse_response(raw_text)

    # If parsing got nothing (LLM ignored the format), fall back to the raw
    # response so the UI is never blank.
    root_cause = parsed["full_root_cause"] if parsed["root_cause"] else raw_text

    return {
        **state,
        "root_cause": root_cause,
    }


if __name__ == "__main__":
    from agents.ingestor import fetch_complaints
    from agents.classifier import classify_complaint
    from agents.router import route_complaint

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
        result = analyze_root_cause(state)

        print(f"ID              : {result['complaint_id']}")
        print(f"Product         : {result['classified_product']}")
        print(f"Issue           : {result['issue_type']}")
        print(f"Root cause      : {result['root_cause']}")
        print()