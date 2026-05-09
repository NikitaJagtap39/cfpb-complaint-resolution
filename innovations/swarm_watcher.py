# innovations/swarm_watcher.py
import os
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
import voyageai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    PointStruct, Filter,
    FieldCondition, Range,
    SearchRequest,
)
from langchain_groq import ChatGroq
from graph.state import ComplaintState

load_dotenv()

vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
qc = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

COLLECTION        = os.getenv("QDRANT_COMPLAINTS_COLLECTION", "complaints")
EMBEDDING_DIM     = 1024
SWARM_THRESHOLD   = 5     # min complaints in cluster to fire alert
SIMILARITY_CUTOFF = 0.82  # cosine similarity threshold


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


def store_complaint_vector(state: ComplaintState, vector: list[float]):
    ensure_collection()
    qc.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id      = str(uuid.uuid4()),
                vector  = vector,
                payload = {
                    "complaint_id":       state["complaint_id"],
                    "classified_product": state.get("classified_product", ""),
                    "issue_type":         state.get("issue_type", ""),
                    "severity":           state.get("severity", ""),
                    "company":            state.get("company", ""),
                    "timestamp":          datetime.now(timezone.utc).isoformat(),
                },
            )
        ],
    )


def find_similar_complaints(vector: list[float]) -> list[dict]:
    ensure_collection()
    results = qc.query_points(
        collection_name = COLLECTION,
        query   = vector,
        limit           = 20,
        score_threshold = SIMILARITY_CUTOFF,
    ).points
    return [
        {
            "complaint_id": r.payload.get("complaint_id"),
            "issue_type":   r.payload.get("issue_type"),
            "severity":     r.payload.get("severity"),
            "company":      r.payload.get("company"),
            "score":        r.score,
        }
        for r in results
    ]


def generate_swarm_alert(
    state: ComplaintState,
    similar: list[dict],
) -> str:
    similar_summary = "\n".join([
        f"  - ID {s['complaint_id']} | {s['issue_type']} | "
        f"{s['severity']} | {s['company']} | similarity {s['score']:.2f}"
        for s in similar[:10]
    ])

    prompt = f"""You are a systemic risk analyst at a fintech regulator.
A cluster of similar complaints has been detected. Generate a concise
systemic alert for the compliance team.

Respond in one paragraph of 3-4 sentences covering:
- What the pattern is
- Which company/product is affected
- The regulatory risk this cluster represents
- Recommended immediate action

### Current complaint:
Product:    {state.get('classified_product')}
Issue:      {state.get('issue_type')}
Severity:   {state.get('severity')}
Company:    {state.get('company')}

### Similar complaints in cluster ({len(similar)} total):
{similar_summary}

Write the alert now:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def detect_swarm(state: ComplaintState) -> ComplaintState:
    # embed the narrative
    vector = embed(state["narrative"][:2000])

    # find similar complaints already in qdrant
    similar = find_similar_complaints(vector)

    # store current complaint vector for future swarm detection
    store_complaint_vector(state, vector)

    swarm_alert = None
    cluster_id  = None

    if len(similar) >= SWARM_THRESHOLD:
        cluster_id  = f"cluster_{state.get('classified_product', 'unknown')}_"  \
                      f"{state.get('issue_type', 'unknown')}"
        swarm_alert = generate_swarm_alert(state, similar)

    return {
        **state,
        "swarm_alert": swarm_alert,
        "cluster_id":  cluster_id,
    }


if __name__ == "__main__":
    from agents.ingestor import fetch_complaints
    from agents.classifier import classify_complaint

    ensure_collection()
    complaints = fetch_complaints(limit=5)

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
        result = detect_swarm(state)

        print(f"ID           : {result['complaint_id']}")
        print(f"Cluster ID   : {result['cluster_id']}")
        print(f"Swarm alert  : {result['swarm_alert'] or 'None'}")
        print()

