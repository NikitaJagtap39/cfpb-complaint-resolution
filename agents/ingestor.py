# agents/ingestor.py
import httpx
from typing import Optional

CFPB_BASE = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

def fetch_complaints(
    limit: int = 100,
    product: Optional[str] = None,
    date_received_min: Optional[str] = None,  # "YYYY-MM-DD"
    date_received_max: Optional[str] = None,
) -> list[dict]:
    params = {
        "size": limit,
        "sort": "created_date_desc",
        "no_aggs": "true",
        "has_narrative": "true",
    }
    if product:
        params["product"] = product
    if date_received_min:
        params["date_received_min"] = date_received_min
    if date_received_max:
        params["date_received_max"] = date_received_max

    response = httpx.get(CFPB_BASE, params=params, timeout=30)
    response.raise_for_status()

    hits = response.json()["hits"]["hits"]

    complaints = []
    for hit in hits:
        src = hit["_source"]
        complaints.append({
            "complaint_id":  src.get("complaint_id", ""),
            "narrative":     src.get("complaint_what_happened", ""),
            "product":       src.get("product", ""),
            "sub_product":   src.get("sub_product", ""),
            "issue":         src.get("issue", ""),
            "sub_issue":     src.get("sub_issue", ""),
            "company":       src.get("company", ""),
            "state":         src.get("state", ""),
            "submitted_via": src.get("submitted_via", ""),
            "date_received": src.get("date_received", ""),
            "tags":          src.get("tags", ""),
        })

    return [c for c in complaints if c["narrative"].strip()]


if __name__ == "__main__":
    results = fetch_complaints(limit=5)

    print(f"Fetched {len(results)} complaints with narratives\n")
    for r in results:
        print("---")
        print("ID:       ", r["complaint_id"])
        print("Product:  ", r["product"])
        print("Company:  ", r["company"])
        print("State:    ", r["state"])
        print("Narrative:", r["narrative"][:200])
        print()

