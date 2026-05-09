# agents/compliance.py
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

REGULATIONS = {
    "FCRA":  "Fair Credit Reporting Act — governs credit reporting accuracy, disputes, and consumer rights",
    "FDCPA": "Fair Debt Collection Practices Act — prohibits abusive, deceptive, or unfair debt collection",
    "ECOA":  "Equal Credit Opportunity Act — prohibits discrimination in credit transactions",
    "UDAP":  "Unfair, Deceptive, or Abusive Acts or Practices — broad CFPB consumer protection standard",
    "TILA":  "Truth in Lending Act — requires clear disclosure of loan terms and costs",
    "RESPA": "Real Estate Settlement Procedures Act — governs mortgage servicing and escrow",
    "EFTA":  "Electronic Fund Transfer Act — governs electronic payments and error resolution",
    "GLBA":  "Gramm-Leach-Bliley Act — governs privacy and security of consumer financial data",
}


def build_prompt(state: ComplaintState) -> str:
    reg_list = "\n".join([f"- {k}: {v}" for k, v in REGULATIONS.items()])

    few_shots = ""
    if state.get("few_shot_examples"):
        examples = state["few_shot_examples"][:2]
        few_shots = "\n\n### Examples of past compliance analyses:\n"
        for ex in examples:
            few_shots += f"""
Narrative: {ex.get('narrative', '')[:300]}
Compliance findings: {ex.get('compliance_findings', '')}
---"""

    return f"""You are a senior compliance officer at a fintech company with deep expertise in consumer financial regulations.

Analyze this complaint for potential regulatory violations and compliance risks.

Respond in this exact format — no extra text, no markdown:

regulations_implicated: <comma separated list of regulation codes from the list below, or NONE>
violation_likelihood: <one of: none | possible | probable | confirmed>
violation_description: <one sentence describing the specific alleged violation or NONE>
required_response_timeframe: <number of days the regulation requires for response, e.g. 30, 45, 60, or N/A>
mandatory_actions: <comma separated list of required regulatory actions or NONE>
escalate_to_legal: <yes | no>
regulatory_exposure: <one of: none | low | medium | high | critical>

Available regulations:
{reg_list}

Violation likelihood guide:
- confirmed: narrative explicitly describes a clear regulatory violation
- probable: strong indicators of violation present
- possible: some indicators but insufficient information to confirm
- none: no regulatory concerns identified
{few_shots}

### Complaint details:
Complaint ID:     {state['complaint_id']}
Product:          {state['classified_product']}
Issue type:       {state['issue_type']}
Severity:         {state['severity']}
Compliance risk:  {state['compliance_risk']}
Root cause:       {state.get('root_cause', 'N/A')}
Company:          {state['company']}
Narrative:        {state['narrative'][:1500]}

Analyze now:"""


def parse_response(text: str) -> dict:
    result = {
        "regulations_implicated":    "NONE",
        "violation_likelihood":      "none",
        "violation_description":     "NONE",
        "required_response_timeframe": "N/A",
        "mandatory_actions":         "NONE",
        "escalate_to_legal":         "no",
        "regulatory_exposure":       "none",
    }

    valid_likelihoods  = {"none", "possible", "probable", "confirmed"}
    valid_exposures    = {"none", "low", "medium", "high", "critical"}

    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key   = key.strip().lower()
        value = value.strip()

        if key == "regulations_implicated":
            result["regulations_implicated"] = value.upper()
        elif key == "violation_likelihood" and value.lower() in valid_likelihoods:
            result["violation_likelihood"] = value.lower()
        elif key == "violation_description":
            result["violation_description"] = value
        elif key == "required_response_timeframe":
            result["required_response_timeframe"] = value
        elif key == "mandatory_actions":
            result["mandatory_actions"] = value
        elif key == "escalate_to_legal" and value.lower() in ("yes", "no"):
            result["escalate_to_legal"] = value.lower()
        elif key == "regulatory_exposure" and value.lower() in valid_exposures:
            result["regulatory_exposure"] = value.lower()

    # bundle into structured compliance findings string
    full_findings = (
        f"Regulations implicated: {result['regulations_implicated']} | "
        f"Violation likelihood: {result['violation_likelihood']} | "
        f"Description: {result['violation_description']} | "
        f"Response timeframe: {result['required_response_timeframe']} days | "
        f"Mandatory actions: {result['mandatory_actions']} | "
        f"Escalate to legal: {result['escalate_to_legal']} | "
        f"Regulatory exposure: {result['regulatory_exposure']}"
    )

    return {**result, "full_findings": full_findings}


def check_compliance(state: ComplaintState) -> ComplaintState:
    prompt   = build_prompt(state)
    response = llm.invoke(prompt)
    parsed   = parse_response(response.content)

    return {
        **state,
        "compliance_findings": parsed["full_findings"],
    }


if __name__ == "__main__":
    from agents.ingestor import fetch_complaints
    from agents.classifier import classify_complaint
    from agents.router import route_complaint
    from agents.root_cause import analyze_root_cause

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
        result = check_compliance(state)

        print(f"ID       : {result['complaint_id']}")
        print(f"Product  : {result['classified_product']}")
        print(f"Findings : {result['compliance_findings']}")
        print()

