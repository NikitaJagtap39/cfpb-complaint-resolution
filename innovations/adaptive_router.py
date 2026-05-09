# innovations/adaptive_router.py
import os
import sqlite3
from graph.state import ComplaintState

DB_PATH = "memory/routing_memory.db"


def init_db():
    os.makedirs("memory", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS routing_outcomes (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_type      TEXT NOT NULL,
            assigned_team       TEXT NOT NULL,
            resolution_time_ms  INTEGER NOT NULL,
            severity            TEXT,
            compliance_risk     TEXT,
            routing_score       REAL,
            created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_complaint_type
        ON routing_outcomes (complaint_type)
    """)
    conn.commit()
    conn.close()


def store_routing_outcome(
    complaint_type: str,
    assigned_team:  str,
    resolution_time_ms: int,
    severity:       str,
    compliance_risk: str,
    routing_score:  float,
):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO routing_outcomes
            (complaint_type, assigned_team, resolution_time_ms,
             severity, compliance_risk, routing_score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (complaint_type, assigned_team, resolution_time_ms,
          severity, compliance_risk, routing_score))
    conn.commit()
    conn.close()


def get_best_team(complaint_type: str) -> dict:
    """
    Returns the team with the lowest average resolution time
    for this complaint type over the last 100 outcomes.
    """
    init_db()
    conn    = sqlite3.connect(DB_PATH)
    cursor  = conn.cursor()
    cursor.execute("""
        SELECT
            assigned_team,
            AVG(resolution_time_ms) as avg_ms,
            COUNT(*)                as sample_size,
            AVG(routing_score)      as avg_confidence
        FROM (
            SELECT assigned_team, resolution_time_ms, routing_score
            FROM routing_outcomes
            WHERE complaint_type = ?
            ORDER BY created_at DESC
            LIMIT 100
        )
        GROUP BY assigned_team
        ORDER BY avg_ms ASC
        LIMIT 1
    """, (complaint_type,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}

    return {
        "best_team":       row[0],
        "avg_ms":          row[1],
        "sample_size":     row[2],
        "avg_confidence":  row[3],
    }


def get_routing_scores(complaint_type: str) -> dict:
    """
    Returns avg resolution time per team for this complaint type.
    Used by router.py to inject historical context into the prompt.
    """
    init_db()
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT assigned_team, AVG(resolution_time_ms)
        FROM routing_outcomes
        WHERE complaint_type = ?
        GROUP BY assigned_team
        ORDER BY AVG(resolution_time_ms) ASC
    """, (complaint_type,))
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def update_routing_memory(state: ComplaintState) -> ComplaintState:
    """
    LangGraph node — called after every resolved complaint.
    Stores the routing outcome so future complaints benefit.
    """
    complaint_type = (
        f"{state.get('classified_product', 'unknown')}_"
        f"{state.get('issue_type', 'unknown')}"
    )

    store_routing_outcome(
        complaint_type     = complaint_type,
        assigned_team      = state.get("assigned_team", "customer-service"),
        resolution_time_ms = state.get("processing_time_ms", 30000),
        severity           = state.get("severity", "medium"),
        compliance_risk    = state.get("compliance_risk", "low"),
        routing_score      = state.get("routing_score", 0.5),
    )

    return state


if __name__ == "__main__":
    init_db()

    # seed some fake outcomes to test adaptive routing
    test_data = [
        ("credit_card_unauthorized_charge", "fraud",            12000, "high",   "high",  0.9),
        ("credit_card_unauthorized_charge", "fraud",            11500, "high",   "high",  0.85),
        ("credit_card_unauthorized_charge", "customer-service", 45000, "medium", "low",   0.6),
        ("mortgage_payment_processing",     "mortgage-ops",     18000, "medium", "low",   0.8),
        ("mortgage_payment_processing",     "customer-service", 60000, "low",    "none",  0.5),
        ("debt_collection_fdcpa_violation", "legal",            9000,  "high",   "regulatory", 0.95),
    ]

    for data in test_data:
        store_routing_outcome(*data)
        print(f"Stored: {data[0]} → {data[1]} ({data[2]}ms)")

    print()
    for complaint_type in [
        "credit_card_unauthorized_charge",
        "mortgage_payment_processing",
        "debt_collection_fdcpa_violation",
    ]:
        best = get_best_team(complaint_type)
        print(f"Best team for '{complaint_type}': {best}")

