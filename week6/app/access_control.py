"""
Week 6: Access Control, Rate Limiting & Cost Enforcement

Three guardrails for the TechCorp AI agent:
1. AccessController - role-based document/field access control with audit logging
2. RateLimiter - sliding-window rate limit per user per minute
3. CostEnforcer - per-role monthly budget enforcement
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from time import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sensitivity levels ordered from least to most restrictive.
# Roles at each level can view documents at that level or below.
_SENSITIVITY_ACCESS: Dict[str, List[str]] = {
    "public":       ["engineer", "manager", "hr", "finance", "executive"],
    "internal":     ["engineer", "manager", "hr", "finance", "executive"],
    "confidential": ["manager", "hr", "finance", "executive"],
    "restricted":   ["finance", "executive"],
}


class AccessController:
    """Enforce role-based access control for documents and fields."""

    def __init__(self, access_policy_path: str):
        with open(access_policy_path, "r", encoding="utf-8") as f:
            self.policy: Dict[str, Any] = json.load(f)
        self.audit_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Document-level access
    # ------------------------------------------------------------------

    def can_view_document(self, role: str, document: Dict[str, Any]) -> bool:
        """Return True if *role* may view *document* based on its sensitivity."""
        sensitivity = document.get("sensitivity", "internal").lower()
        allowed_roles = _SENSITIVITY_ACCESS.get(sensitivity, [])
        return role.lower() in allowed_roles

    def filter_documents(self, role: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return only the documents that *role* is permitted to view."""
        visible = []
        for doc in documents:
            allowed = self.can_view_document(role, doc)
            self.log_access(role, doc.get("id", doc.get("title", "unknown")), allowed)
            if allowed:
                visible.append(doc)
        return visible

    # ------------------------------------------------------------------
    # Field-level access
    # ------------------------------------------------------------------

    def can_view_field(self, role: str, field_name: str) -> bool:
        """Return True if *role* may view *field_name*.

        Fields not listed in sensitive_fields are visible to everyone.
        """
        sensitive_fields: Dict[str, Any] = self.policy.get("sensitive_fields", {})
        if field_name not in sensitive_fields:
            return True
        visibility: List[str] = sensitive_fields[field_name].get("visibility", [])
        return role.lower() in [v.lower() for v in visibility]

    def redact_response(self, role: str, response: str) -> str:
        """Replace values of sensitive fields the *role* cannot view with [REDACTED]."""
        sensitive_fields: Dict[str, Any] = self.policy.get("sensitive_fields", {})
        redacted = response

        for field, config in sensitive_fields.items():
            if self.can_view_field(role, field):
                continue
            # Match "field: value" or "field: $value" or JSON "field": "value"
            patterns = [
                # JSON key-value: "salary": "120000" or "salary": 120000
                rf'("{re.escape(field)}"\s*:\s*)"([^"]*)"',
                rf'("{re.escape(field)}"\s*:\s*)(\d[\d,.$]*)',
                # Plain text: salary: $120,000 or salary: 120000
                rf'(?i)({re.escape(field)}\s*:\s*)([^\n,;]+)',
            ]
            for pattern in patterns:
                redacted = re.sub(pattern, lambda m, p=pattern: m.group(1) + "[REDACTED]", redacted)

        # Always redact SSN-shaped patterns (###-##-####) regardless of field name
        if not self.can_view_field(role, "ssn"):
            redacted = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED]', redacted)

        return redacted

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def log_access(self, role: str, resource: str, allowed: bool, field: Optional[str] = None):
        """Append an access-attempt record to the in-memory audit log."""
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "resource": resource,
            "allowed": allowed,
        }
        if field is not None:
            entry["field"] = field
        self.audit_log.append(entry)
        status = "ALLOWED" if allowed else "DENIED"
        logger.debug("ACCESS %s | role=%s resource=%s", status, role, resource)

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Return all recorded audit log entries."""
        return self.audit_log


# ===========================================================================
# RateLimiter
# ===========================================================================

class RateLimiter:
    """Sliding-window rate limiter: max N queries per user per 60 seconds."""

    def __init__(self, max_queries_per_minute: int = 30):
        self.max_queries_per_minute = max_queries_per_minute
        # Maps user_id -> list of epoch timestamps (seconds) for recent queries
        self.user_query_times: Dict[str, List[float]] = {}

    def _prune(self, user_id: str, now: float) -> List[float]:
        """Return timestamps for *user_id* within the last 60 seconds."""
        window_start = now - 60.0
        times = [t for t in self.user_query_times.get(user_id, []) if t >= window_start]
        self.user_query_times[user_id] = times
        return times

    def is_allowed(self, user_id: str) -> bool:
        """Return True and record the query if the user is within their rate limit."""
        now = time()
        recent = self._prune(user_id, now)
        if len(recent) >= self.max_queries_per_minute:
            logger.warning("Rate limit hit for user %s (%d/%d)", user_id, len(recent), self.max_queries_per_minute)
            return False
        self.user_query_times[user_id] = recent + [now]
        return True

    def get_remaining_queries(self, user_id: str) -> int:
        """Return how many more queries the user may make in the current window."""
        recent = self._prune(user_id, time())
        return max(0, self.max_queries_per_minute - len(recent))


# ===========================================================================
# CostEnforcer
# ===========================================================================

class CostEnforcer:
    """Enforce per-role monthly API budget limits."""

    # Default monthly budgets in USD
    _DEFAULT_BUDGETS: Dict[str, float] = {
        "engineer":  100.0,
        "manager":   500.0,
        "hr":        200.0,
        "finance":   500.0,
        "executive": 1000.0,
    }

    def __init__(self, policy_path: Optional[str] = None):
        if policy_path:
            with open(policy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.role_budgets: Dict[str, float] = data.get("budgets", self._DEFAULT_BUDGETS)
        else:
            self.role_budgets = dict(self._DEFAULT_BUDGETS)

        # Maps user_id -> {"role": str, "total": float}
        self.user_spending: Dict[str, Dict[str, Any]] = {}

    def add_cost(self, user_id: str, role: str, cost: float):
        """Record *cost* (USD) against *user_id*'s running total."""
        if user_id not in self.user_spending:
            self.user_spending[user_id] = {"role": role, "total": 0.0}
        self.user_spending[user_id]["total"] += cost

    def _get_budget(self, user_id: str) -> float:
        """Look up the monthly budget for this user based on their role."""
        if user_id in self.user_spending:
            role = self.user_spending[user_id].get("role", "engineer")
        else:
            role = "engineer"
        return self.role_budgets.get(role.lower(), self._DEFAULT_BUDGETS["engineer"])

    def can_afford_query(self, user_id: str, estimated_cost: float) -> bool:
        """Return True if the user has enough remaining budget for *estimated_cost*."""
        spent = self.user_spending.get(user_id, {}).get("total", 0.0)
        budget = self._get_budget(user_id)
        remaining = budget - spent
        if estimated_cost > remaining:
            logger.warning(
                "Budget exceeded for user %s: remaining=%.4f estimated=%.4f",
                user_id, remaining, estimated_cost,
            )
            return False
        return True

    def get_budget_remaining(self, user_id: str) -> float:
        """Return remaining budget for *user_id* (never negative)."""
        spent = self.user_spending.get(user_id, {}).get("total", 0.0)
        budget = self._get_budget(user_id)
        return max(0.0, budget - spent)
