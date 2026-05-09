# agents/synthesiser.py
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


def build_resolution_prompt(state: ComplaintState) -> str:
    remediation_summary = "\n".join(
        f"  - {step}" for step in state.get("remediation_steps", [])
    )

    return f"""You are a chief complaints officer at a fintech company.
Synthesise all agent outputs into a single coherent resolution plan.

Respond in this exact format — no extra text, no markdown:

executive_summary: <2-3 sentence summary of the complaint and resolution>
resolution_status: <one of: resolved | in_progress | escalated | pending_review>
priority_actions: <comma separated list of the top 3 most critical actions>
team_assignments: <comma separated list of team:action pairs e.g. legal:file_regulatory_response>
estimated_closure_date: <number of days from today>
customer_impact_score: <integer 1-10 where 10 is highest impact>
systemic_issue: <yes | no>
systemic_description: <one sentence describing the systemic issue or NONE>

### All agent outputs:
Complaint ID:          {state['complaint_id']}
Product:               {state['classified_product']}
Issue type:            {state['issue_type']}
Severity:              {state['severity']}
Compliance risk:       {state['compliance_risk']}
Assigned team:         {state['assigned_team']}
Routing score:         {state['routing_score']}
Root cause:            {state.get('root_cause', 'N/A')}
Compliance findings:   {state.get('compliance_findings', 'N/A')}
Swarm alert:           {state.get('swarm_alert', 'None')}
Remediation steps:
{remediation_summary}
Customer response:     {state.get('customer_response', 'N/A')[:300]}

Synthesise now:"""


def build_explainability_prompt(state: ComplaintState, resolution: dict) -> str:
    return f"""You are a regulatory affairs officer preparing an explainability report
for CFPB regulators. This report must justify every AI decision made during
complaint processing in plain, auditable language.

Write a structured explainability report covering these sections exactly:

1. Complaint summary
2. Classification rationale — why this product/issue/severity was assigned
3. Routing rationale — why this team was selected and what historical data informed it
4. Root cause analysis — what failed and why
5. Compliance assessment — which regulations apply and why
6. Remediation justification — why these specific steps were chosen
7. Systemic risk assessment — whether this complaint indicates a broader pattern
8. Decision confidence — overall confidence in the resolution and any uncertainties

Keep each section to 2-3 sentences. Use plain language suitable for regulators.
Do not use bullet points. Write in paragraphs.

### Context:
Complaint ID:          {state['complaint_id']}
Product:               {state['classified_product']}
Issue type:            {state['issue_type']}
Severity:              {state['severity']}
Compliance risk:       {state['compliance_risk']}
Assigned team:         {state['assigned_team']}
Routing score:         {state['routing_score']}
Root cause:            {state.get('root_cause', 'N/A')}
Compliance findings:   {state.get('compliance_findings', 'N/A')}
Swarm alert:           {state.get('swarm_alert', 'None')}
Executive summary:     {resolution.get('executive_summary', 'N/A')}
Resolution status:     {resolution.get('resolution_status', 'N/A')}
Customer impact score: {resolution.get('customer_impact_score', 'N/A')}
Systemic issue:        {resolution.get('systemic_issue', 'N/A')}

Write the explainability report now:"""


def parse_resolution(text: str) -> dict:
    result = {
        "executive_summary":      "",
        "resolution_status":      "in_progress",
        "priority_actions":       "",
        "team_assignments":       "",
        "estimated_closure_date": 30,
        "customer_impact_score":  5,
        "systemic_issue":         "no",
        "systemic_description":   "NONE",
    }

    valid_statuses = {"resolved", "in_progress", "escalated", "pending_review"}

    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key   = key.strip().lower()
        value = value.strip()

        if key == "executive_summary" and value:
            result["executive_summary"] = value
        elif key == "resolution_status" and value.lower() in valid_statuses:
            result["resolution_status"] = value.lower()
        elif key == "priority_actions" and value:
            result["priority_actions"] = value
        elif key == "team_assignments" and value:
            result["team_assignments"] = value
        elif key == "estimated_closure_date":
            try:
                result["estimated_closure_date"] = int(value)
            except ValueError:
                pass
        elif key == "customer_impact_score":
            try:
                result["customer_impact_score"] = max(1, min(10, int(value)))
            except ValueError:
                pass
        elif key == "systemic_issue" and value.lower() in ("yes", "no"):
            result["systemic_issue"] = value.lower()
        elif key == "systemic_description":
            result["systemic_description"] = value

    return result


def build_resolution_plan(state: ComplaintState, resolution: dict) -> str:
    remediation_summary = "\n".join(
        f"  {i+1}. {step}"
        for i, step in enumerate(state.get("remediation_steps", []))
    )

    return f"""=== COMPLAINT RESOLUTION PLAN ===
Complaint ID       : {state['complaint_id']}
Company            : {state['company']}
Product            : {state['classified_product']}
Issue type         : {state['issue_type']}
Severity           : {state['severity']}
Compliance risk    : {state['compliance_risk']}
Assigned team      : {state['assigned_team']}
Resolution status  : {resolution['resolution_status']}
Customer impact    : {resolution['customer_impact_score']}/10
Est. closure       : {resolution['estimated_closure_date']} days
Systemic issue     : {resolution['systemic_issue']}
Swarm alert        : {state.get('swarm_alert') or 'None'}

--- Executive Summary ---
{resolution['executive_summary']}

--- Priority Actions ---
{resolution['priority_actions']}

--- Team Assignments ---
{resolution['team_assignments']}

--- Remediation Steps ---
{remediation_summary}

--- Systemic Risk ---
{resolution['systemic_description']}

--- Root Cause ---
{state.get('root_cause', 'N/A')}

--- Compliance Findings ---
{state.get('compliance_findings', 'N/A')}
"""


def synthesise(state: ComplaintState) -> ComplaintState:
    # step 1 — build resolution plan
    resolution_response = llm.invoke(build_resolution_prompt(state))
    resolution          = parse_resolution(resolution_response.content)
    resolution_plan     = build_resolution_plan(state, resolution)

    # step 2 — build explainability report
    explainability_response = llm.invoke(build_explainability_prompt(state, resolution))
    explainability_report   = explainability_response.content.strip()

    return {
        **state,
        "resolution_plan":       resolution_plan,
        "explainability_report": explainability_report,
    }


if __name__ == "__main__":
    from agents.ingestor import fetch_complaints
    from agents.classifier import classify_complaint
    from agents.router import route_complaint
    from agents.root_cause import analyze_root_cause
    from agents.compliance import check_compliance
    from agents.remediation import generate_remediation
    from agents.response_draft import draft_response

    complaints = fetch_complaints(limit=1)

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
        state  = generate_remediation(state)
        state  = draft_response(state)
        result = synthesise(state)

        print(result["resolution_plan"])
        print()
        print("=== EXPLAINABILITY REPORT ===")
        print(result["explainability_report"])

