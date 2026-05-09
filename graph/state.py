# graph/state.py
from typing import TypedDict, Optional, Annotated
import operator

class ComplaintState(TypedDict):
    # ── Raw input from CFPB ingestor ──────────────────────────────
    complaint_id:   str
    narrative:      str
    product:        str
    sub_product:    str
    issue:          str
    sub_issue:      str
    company:        str
    state:          str
    submitted_via:  str
    date_received:  str
    tags:           str

    # ── Classifier outputs ────────────────────────────────────────
    classified_product:  str        # normalized product bucket
    issue_type:          str        # standardized issue label
    severity:            str        # low / medium / high / critical
    compliance_risk:     str        # none / low / high / regulatory

    # ── Routing ───────────────────────────────────────────────────
    assigned_team:   str            # e.g. "credit-ops", "legal", "fraud"
    routing_score:   float          # confidence score from adaptive router

    # ── Specialist agent outputs ──────────────────────────────────
    root_cause:           str
    compliance_findings:  str
    remediation_steps:    list[str]
    customer_response:    str

    # ── Synthesiser outputs ───────────────────────────────────────
    resolution_plan:       str
    explainability_report: str

    # ── Innovation 2: swarm detection ─────────────────────────────
    swarm_alert:  Optional[str]     # None if no cluster detected
    cluster_id:   Optional[str]     # Qdrant cluster group id if detected

    # ── Innovation 3: self-improving loop ─────────────────────────
    few_shot_examples: list[dict]   # injected before classifier runs

    # ── Metrics ───────────────────────────────────────────────────
    processing_time_ms: int
    error:              Optional[str]   # set if any node fails

