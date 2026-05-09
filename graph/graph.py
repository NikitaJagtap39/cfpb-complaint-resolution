# graph/graph.py
from langgraph.graph import StateGraph, END
from graph.state import ComplaintState
from agents.ingestor import fetch_complaints
from agents.classifier import classify_complaint
from agents.router import route_complaint
from agents.root_cause import analyze_root_cause
from agents.compliance import check_compliance
from agents.remediation import generate_remediation
from agents.response_draft import draft_response
from agents.synthesiser import synthesise
from innovations.adaptive_router import update_routing_memory
from innovations.self_improving import inject_few_shots, store_resolution
from graph.edges import route_to_team


def build_graph() -> StateGraph:
    graph = StateGraph(ComplaintState)

    # ── Register nodes ────────────────────────────────────────────
    # Note: swarm_watcher is intentionally excluded from the automatic
    # pipeline. It is available as a manual admin tool in the UI.
    graph.add_node("inject_few_shots",    inject_few_shots)
    graph.add_node("classifier",          classify_complaint)
    graph.add_node("router",              route_complaint)
    graph.add_node("root_cause",          analyze_root_cause)
    graph.add_node("compliance",          check_compliance)
    graph.add_node("remediation",         generate_remediation)
    graph.add_node("response_draft",      draft_response)
    graph.add_node("synthesiser",         synthesise)
    graph.add_node("store_resolution",    store_resolution)
    graph.add_node("update_routing",      update_routing_memory)

    # ── Entry point ───────────────────────────────────────────────
    graph.set_entry_point("inject_few_shots")

    # ── Main pipeline edges ───────────────────────────────────────
    graph.add_edge("inject_few_shots", "classifier")
    graph.add_edge("classifier",       "router")

    # ── Conditional routing to specialist agents ──────────────────
    graph.add_conditional_edges(
        "router",
        route_to_team,
        {
            "standard":   "root_cause",
            "compliance": "compliance",
            "escalate":   "compliance",   # escalations go to compliance first
        }
    )

    # ── Specialist agent edges ────────────────────────────────────
    # standard path: root_cause → remediation → response_draft
    graph.add_edge("root_cause",     "remediation")
    graph.add_edge("remediation",    "response_draft")

    # compliance path: compliance → remediation → response_draft
    graph.add_edge("compliance",     "remediation")

    # all paths converge at response_draft → synthesiser
    graph.add_edge("response_draft", "synthesiser")

    # ── Post-resolution: feedback loops ──────────────────────────
    graph.add_edge("synthesiser",      "store_resolution")
    graph.add_edge("store_resolution", "update_routing")
    graph.add_edge("update_routing",   END)

    return graph.compile()


# compiled graph — import this everywhere else
complaint_graph = build_graph()


if __name__ == "__main__":
    from agents.ingestor import fetch_complaints
    import time

    complaints = fetch_complaints(limit=1)
    if not complaints:
        print("No complaints fetched")
        exit()

    c = complaints[0]

    initial_state: ComplaintState = {
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
        # defaults — agents will fill these in
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

    start = time.time()
    result = complaint_graph.invoke(initial_state)
    elapsed = int((time.time() - start) * 1000)

    print("=== RESOLUTION COMPLETE ===")
    print(f"Complaint ID : {result['complaint_id']}")
    print(f"Severity     : {result['severity']}")
    print(f"Team         : {result['assigned_team']}")
    print(f"Time taken   : {elapsed}ms")
    print("\n--- Resolution Plan ---")
    print(result["resolution_plan"])
    print("\n--- Explainability Report ---")
    print(result["explainability_report"])