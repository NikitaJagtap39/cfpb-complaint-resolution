# agents/response_draft.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from graph.state import ComplaintState

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4,
)


def build_prompt(state: ComplaintState) -> str:
    remediation_summary = "\n".join(
        f"  - {step}" for step in state.get("remediation_steps", [])
    )

    few_shots = ""
    if state.get("few_shot_examples"):
        examples = state["few_shot_examples"][:2]
        few_shots = "\n\n### Examples of past customer responses:\n"
        for ex in examples:
            few_shots += f"""
Narrative: {ex.get('narrative', '')[:200]}
Response: {ex.get('customer_response', '')[:400]}
---"""

    return f"""You are a regulatory compliance writer at a fintech company.
Write a formal, empathetic, and fully regulation-compliant customer response letter
for this complaint. The letter will be sent directly to the consumer.

Rules:
- Never admit legal liability
- Never promise outcomes you cannot guarantee
- Acknowledge the complaint specifically — do not use generic language
- Include the case reference number
- State the expected resolution timeframe clearly
- If a regulation applies, reference the consumer's rights without legalese
- Close with a clear next step for the consumer
- Tone: professional, empathetic, human — not robotic
- Length: 3 to 5 paragraphs
- Do not include a subject line or email headers
- Start directly with "Dear Valued Customer,"

### Complaint context:
Complaint ID:          {state['complaint_id']}
Product:               {state['classified_product']}
Issue type:            {state['issue_type']}
Severity:              {state['severity']}
Regulations implicated:{state.get('compliance_findings', 'N/A')}
Company:               {state['company']}
Assigned team:         {state['assigned_team']}
Remediation steps:
{remediation_summary}
Narrative summary:     {state['narrative'][:800]}
{few_shots}

Write the response letter now:"""


def draft_response(state: ComplaintState) -> ComplaintState:
    prompt   = build_prompt(state)
    response = llm.invoke(prompt)

    return {
        **state,
        "customer_response": response.content.strip(),
    }


if __name__ == "__main__":
    from agents.ingestor import fetch_complaints
    from agents.classifier import classify_complaint
    from agents.router import route_complaint
    from agents.root_cause import analyze_root_cause
    from agents.compliance import check_compliance
    from agents.remediation import generate_remediation

    complaints = fetch_complaints(limit=2)

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
        result = draft_response(state)

        print(f"ID       : {result['complaint_id']}")
        print(f"Product  : {result['classified_product']}")
        print(f"Severity : {result['severity']}")
        print()
        print("--- Customer Response ---")
        print(result["customer_response"])
        print()

