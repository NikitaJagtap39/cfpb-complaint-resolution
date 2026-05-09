# innovations/self_improving.py
import os
import uuid
import json
from dotenv import load_dotenv
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    PointStruct,
)
from graph.state import ComplaintState

load_dotenv()

vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
qc = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION    = os.getenv("QDRANT_EXAMPLES_COLLECTION", "resolution_examples")
EMBEDDING_DIM = 1024
TOP_K         = 3   # number of few-shot examples to inject


def ensure_collection():
    existing = [c.name for c in qc.get_collections().collections]
    if COLLECTION not in existing:
        qc.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )


def embed(text: str) -> list[float]:
    return vo.embed(
        [text],
        model="voyage-finance-2",
    ).embeddings[0]


def retrieve_similar_examples(narrative: str) -> list[dict]:
    """
    Find top-k most similar resolved complaints from the store.
    Returns list of dicts with past resolution fields.
    """
    ensure_collection()

    try:
        vector  = embed(narrative[:2000])
        results = qc.query_points(
            collection_name = COLLECTION,
            query    = vector,
            limit           = TOP_K,
            score_threshold = 0.75,
        ).points

        examples = []
        for r in results:
            payload = r.payload or {}
            examples.append({
                "narrative":           payload.get("narrative", ""),
                "classified_product":  payload.get("classified_product", ""),
                "issue_type":          payload.get("issue_type", ""),
                "severity":            payload.get("severity", ""),
                "compliance_risk":     payload.get("compliance_risk", ""),
                "root_cause":          payload.get("root_cause", ""),
                "compliance_findings": payload.get("compliance_findings", ""),
                "remediation_steps":   payload.get("remediation_steps", []),
                "customer_response":   payload.get("customer_response", ""),
                "similarity_score":    r.score,
            })
        return examples

    except Exception:
        return []


def inject_few_shots(state: ComplaintState) -> ComplaintState:
    """
    LangGraph node — runs first in the graph.
    Retrieves similar past resolutions and injects them into state.
    """
    examples = retrieve_similar_examples(state["narrative"])

    return {
        **state,
        "few_shot_examples": examples,
    }


def store_resolution(state: ComplaintState) -> ComplaintState:
    """
    LangGraph node — runs after synthesiser.
    Stores the resolved complaint as a future few-shot example.
    """
    ensure_collection()

    try:
        vector = embed(state["narrative"][:2000])

        qc.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id     = str(uuid.uuid4()),
                    vector = vector,
                    payload = {
                        "complaint_id":        state["complaint_id"],
                        "narrative":           state["narrative"][:1000],
                        "classified_product":  state.get("classified_product", ""),
                        "issue_type":          state.get("issue_type", ""),
                        "severity":            state.get("severity", ""),
                        "compliance_risk":     state.get("compliance_risk", ""),
                        "root_cause":          state.get("root_cause", ""),
                        "compliance_findings": state.get("compliance_findings", ""),
                        "remediation_steps":   state.get("remediation_steps", []),
                        "customer_response":   state.get("customer_response", "")[:500],
                        "assigned_team":       state.get("assigned_team", ""),
                        "resolution_plan":     state.get("resolution_plan", "")[:500],
                    },
                )
            ],
        )
    except Exception as e:
        print(f"Warning: could not store resolution example: {e}")

    return state


if __name__ == "__main__":
    from agents.ingestor import fetch_complaints
    from agents.classifier import classify_complaint
    from agents.router import route_complaint
    from agents.root_cause import analyze_root_cause
    from agents.compliance import check_compliance
    from agents.remediation import generate_remediation
    from agents.response_draft import draft_response
    from agents.synthesiser import synthesise

    ensure_collection()
    complaints = fetch_complaints(limit=3)

    print("=== Pass 1: processing and storing resolutions ===")
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

        state = classify_complaint(state)
        state = route_complaint(state)
        state = analyze_root_cause(state)
        state = check_compliance(state)
        state = generate_remediation(state)
        state = draft_response(state)
        state = synthesise(state)
        state = store_resolution(state)
        print(f"Stored: {state['complaint_id']}")

    print()
    print("=== Pass 2: inject few shots for a new complaint ===")
    new_complaints = fetch_complaints(limit=1)
    if new_complaints:
        c      = new_complaints[0]
        result = inject_few_shots({**c, "few_shot_examples": []})
        print(f"Injected {len(result['few_shot_examples'])} examples")
        for ex in result["few_shot_examples"]:
            print(f"  - {ex['complaint_id']} | "
                  f"{ex['classified_product']} | "
                  f"similarity {ex['similarity_score']:.2f}")

