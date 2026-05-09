# graph/edges.py
from graph.state import ComplaintState


def route_to_team(state: ComplaintState) -> str:
    """
    Called by the conditional edge after the router node runs.
    Returns a string key that maps to the next node in graph.py.
    
    Routing priority (highest to lowest):
      1. critical severity          → escalate
      2. high / regulatory risk     → compliance
      3. everything else            → standard
    """
    severity        = (state.get("severity")        or "").lower().strip()
    compliance_risk = (state.get("compliance_risk") or "").lower().strip()
    assigned_team   = (state.get("assigned_team")   or "").lower().strip()

    # critical complaints always escalate regardless of anything else
    if severity == "critical":
        return "escalate"

    # regulatory or high compliance risk goes to compliance agent first
    if compliance_risk in ("high", "regulatory"):
        return "compliance"

    # legal team assignment overrides standard routing
    if assigned_team == "legal":
        return "compliance"

    # everything else goes through standard root-cause path
    return "standard"

