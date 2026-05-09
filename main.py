# main.py
import os
import time
from dotenv import load_dotenv
from graph.graph import complaint_graph
from agents.ingestor import fetch_complaints
from metrics.evaluator import evaluate, get_aggregate_metrics, init_metrics_db
from innovations.adaptive_router import init_db as init_routing_db
from innovations.self_improving import ensure_collection as init_examples_collection
from innovations.swarm_watcher import ensure_collection as init_complaints_collection
from graph.state import ComplaintState

load_dotenv()


def build_initial_state(c: dict) -> ComplaintState:
    return {
        "complaint_id":          c["complaint_id"],
        "narrative":             c["narrative"],
        "product":               c["product"],
        "sub_product":           c["sub_product"],
        "issue":                 c["issue"],
        "sub_issue":             c["sub_issue"],
        "company":               c["company"],
        "state":                 c["state"],
        "submitted_via":         c["submitted_via"],
        "date_received":         c["date_received"],
        "tags":                  c["tags"],
        "classified_product":    "",
        "issue_type":            "",
        "severity":              "",
        "compliance_risk":       "",
        "assigned_team":         "",
        "routing_score":         0.0,
        "root_cause":            "",
        "compliance_findings":   "",
        "remediation_steps":     [],
        "customer_response":     "",
        "resolution_plan":       "",
        "explainability_report": "",
        "swarm_alert":           None,
        "cluster_id":            None,
        "few_shot_examples":     [],
        "processing_time_ms":    0,
        "error":                 None,
    }


def print_separator(char="─", width=60):
    print(char * width)


def print_result(result: ComplaintState, metrics: dict, index: int, total: int):
    print_separator("═")
    print(f"  COMPLAINT {index}/{total}")
    print_separator("═")

    print(f"  ID           : {result['complaint_id']}")
    print(f"  Company      : {result['company']}")
    print(f"  Product      : {result['classified_product']}")
    print(f"  Issue        : {result['issue_type']}")
    print(f"  Severity     : {result['severity']}")
    print(f"  Risk         : {result['compliance_risk']}")
    print(f"  Team         : {result['assigned_team']}")
    print(f"  Few shots    : {len(result.get('few_shot_examples', []))} examples injected")
    print(f"  Time         : {result['processing_time_ms']}ms")

    if result.get("swarm_alert"):
        print_separator("!")
        print("  ⚠  SWARM ALERT DETECTED")
        print(f"  Cluster ID   : {result['cluster_id']}")
        print(f"  Alert        : {result['swarm_alert'][:200]}...")
        print_separator("!")

    print_separator()
    print("  METRICS")
    print_separator()
    print(f"  Quality score : {metrics['quality_score']}")
    print(f"  Fairness flag : {'YES ⚠' if metrics['fairness_flag'] else 'No'}")
    print(f"  CSAT predicted: {metrics['predicted_csat']} / 5.0")
    print(f"  Churn risk    : {metrics['churn_risk_label']} ({metrics['churn_risk']})")

    print_separator()
    print("  RESOLUTION PLAN")
    print_separator()
    print(result["resolution_plan"])

    print_separator()
    print("  CUSTOMER RESPONSE")
    print_separator()
    print(result["customer_response"])

    print_separator()
    print("  EXPLAINABILITY REPORT")
    print_separator()
    print(result["explainability_report"])
    print()


def print_dashboard(agg: dict):
    print_separator("═")
    print("  SYSTEM DASHBOARD")
    print_separator("═")
    print(f"  Total processed     : {agg.get('total_complaints', 0)}")
    print(f"  Avg processing time : {agg.get('avg_processing_ms', 0)}ms")
    print(f"  Avg quality score   : {agg.get('avg_quality_score', 0)}")
    print(f"  Fairness flags      : {agg.get('fairness_flags', 0)}")
    print(f"  Swarm detections    : {agg.get('swarm_detections', 0)}")
    print(f"  Avg few shots used  : {agg.get('avg_few_shots_used', 0)}")

    print_separator()
    print("  SEVERITY DISTRIBUTION")
    print_separator()
    for severity, count in agg.get("severity_distribution", {}).items():
        bar = "█" * count
        print(f"  {severity:<10}: {bar} ({count})")

    print_separator()
    print("  TEAM PERFORMANCE (avg resolution ms)")
    print_separator()
    for team, avg_ms in agg.get("team_avg_resolution_ms", {}).items():
        print(f"  {team:<20}: {avg_ms}ms")

    print_separator("═")


def run(limit: int = 5, product: str = None):
    print_separator("═")
    print("  CFPB COMPLAINT RESOLUTION SYSTEM")
    print("  Powered by LangGraph + Groq + Voyage + Qdrant")
    print_separator("═")
    print()

    # initialise all storage
    print("Initialising storage...")
    init_metrics_db()
    init_routing_db()
    init_examples_collection()
    init_complaints_collection()
    print("Storage ready.\n")

    # fetch complaints
    print(f"Fetching {limit} complaints from CFPB API...")
    complaints = fetch_complaints(limit=limit, product=product)
    print(f"Fetched {len(complaints)} complaints with narratives.\n")

    if not complaints:
        print("No complaints fetched. Exiting.")
        return

    total = len(complaints)

    for i, c in enumerate(complaints, start=1):
        print(f"\nProcessing complaint {i}/{total}: {c['complaint_id']}...")

        initial_state = build_initial_state(c)

        try:
            start  = time.time()
            result = complaint_graph.invoke(initial_state)
            elapsed = int((time.time() - start) * 1000)
            result["processing_time_ms"] = elapsed

            metrics = evaluate(result)
            print_result(result, metrics, i, total)

        except Exception as e:
            print(f"  ERROR processing {c['complaint_id']}: {e}")
            continue

    # print aggregate dashboard
    agg = get_aggregate_metrics()
    print_dashboard(agg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CFPB Complaint Resolution System"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of complaints to process (default: 5)"
    )
    parser.add_argument(
        "--product",
        type=str,
        default=None,
        help=(
            "Filter by CFPB product string e.g. "
            "'Credit card' 'Mortgage' 'Debt collection'"
        )
    )
    args = parser.parse_args()
    run(limit=args.limit, product=args.product)

