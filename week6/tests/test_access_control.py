"""
Week 6 Tests: AccessController, RateLimiter, CostEnforcer
"""

import json
import os
import sys
import time
import pytest

# Resolve paths so tests can be run from anywhere.
WEEK6_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WEEK6_DIR)

from app.access_control import AccessController, RateLimiter, CostEnforcer

POLICY_PATH = os.path.join(WEEK6_DIR, "data", "access_control.json")


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def controller():
    return AccessController(POLICY_PATH)


@pytest.fixture
def rate_limiter():
    return RateLimiter(max_queries_per_minute=5)


@pytest.fixture
def cost_enforcer():
    return CostEnforcer()


# ---------------------------------------------------------------------------
# Sample documents for testing
# ---------------------------------------------------------------------------

PUBLIC_DOC     = {"id": "pub1",  "title": "Mission",          "sensitivity": "Public"}
INTERNAL_DOC   = {"id": "int1",  "title": "Travel Policy",    "sensitivity": "Internal"}
CONFIDENTIAL_DOC = {"id": "con1","title": "Salary Ranges",    "sensitivity": "Confidential"}
RESTRICTED_DOC = {"id": "res1",  "title": "Executive Report", "sensitivity": "Restricted"}


# ===========================================================================
# AccessController — initialisation
# ===========================================================================

class TestAccessControllerInit:
    def test_loads_policy(self, controller):
        assert controller.policy, "Policy should not be empty after loading"

    def test_has_roles(self, controller):
        assert "roles" in controller.policy

    def test_has_sensitive_fields(self, controller):
        assert "sensitive_fields" in controller.policy

    def test_audit_log_starts_empty(self, controller):
        assert controller.audit_log == []


# ===========================================================================
# AccessController — can_view_document
# ===========================================================================

class TestCanViewDocument:
    # Public documents
    def test_engineer_can_view_public(self, controller):
        assert controller.can_view_document("engineer", PUBLIC_DOC)

    def test_hr_can_view_public(self, controller):
        assert controller.can_view_document("hr", PUBLIC_DOC)

    # Internal documents
    def test_engineer_can_view_internal(self, controller):
        assert controller.can_view_document("engineer", INTERNAL_DOC)

    def test_manager_can_view_internal(self, controller):
        assert controller.can_view_document("manager", INTERNAL_DOC)

    # Confidential documents
    def test_engineer_cannot_view_confidential(self, controller):
        assert not controller.can_view_document("engineer", CONFIDENTIAL_DOC)

    def test_manager_can_view_confidential(self, controller):
        assert controller.can_view_document("manager", CONFIDENTIAL_DOC)

    def test_hr_can_view_confidential(self, controller):
        assert controller.can_view_document("hr", CONFIDENTIAL_DOC)

    def test_executive_can_view_confidential(self, controller):
        assert controller.can_view_document("executive", CONFIDENTIAL_DOC)

    # Restricted documents
    def test_engineer_cannot_view_restricted(self, controller):
        assert not controller.can_view_document("engineer", RESTRICTED_DOC)

    def test_manager_cannot_view_restricted(self, controller):
        assert not controller.can_view_document("manager", RESTRICTED_DOC)

    def test_hr_cannot_view_restricted(self, controller):
        assert not controller.can_view_document("hr", RESTRICTED_DOC)

    def test_finance_can_view_restricted(self, controller):
        assert controller.can_view_document("finance", RESTRICTED_DOC)

    def test_executive_can_view_restricted(self, controller):
        assert controller.can_view_document("executive", RESTRICTED_DOC)

    # Case-insensitivity
    def test_case_insensitive_sensitivity(self, controller):
        doc = {"id": "x", "title": "X", "sensitivity": "CONFIDENTIAL"}
        assert not controller.can_view_document("engineer", doc)
        assert controller.can_view_document("manager", doc)


# ===========================================================================
# AccessController — can_view_field
# ===========================================================================

class TestCanViewField:
    def test_engineer_can_view_name(self, controller):
        # 'name' is not a sensitive field — everyone can view it
        assert controller.can_view_field("engineer", "name")

    def test_engineer_cannot_view_salary(self, controller):
        assert not controller.can_view_field("engineer", "salary")

    def test_manager_cannot_view_salary(self, controller):
        # Per policy: salary visible to executive, hr, finance only — not manager
        assert not controller.can_view_field("manager", "salary")

    def test_hr_can_view_salary(self, controller):
        assert controller.can_view_field("hr", "salary")

    def test_engineer_cannot_view_ssn(self, controller):
        assert not controller.can_view_field("engineer", "ssn")

    def test_hr_can_view_ssn(self, controller):
        assert controller.can_view_field("hr", "ssn")

    def test_finance_can_view_ssn(self, controller):
        assert controller.can_view_field("finance", "ssn")

    def test_engineer_cannot_view_performance_review(self, controller):
        assert not controller.can_view_field("engineer", "performance_review")

    def test_manager_can_view_performance_review(self, controller):
        assert controller.can_view_field("manager", "performance_review")

    def test_unknown_field_visible_to_all(self, controller):
        assert controller.can_view_field("engineer", "department")
        assert controller.can_view_field("hr", "department")


# ===========================================================================
# AccessController — redact_response
# ===========================================================================

class TestRedactResponse:
    def test_no_redaction_for_permitted_role(self, controller):
        text = "salary: $120,000"
        result = controller.redact_response("hr", text)
        assert "120,000" in result

    def test_redacts_salary_for_engineer(self, controller):
        text = "The employee's salary: $120,000 per year."
        result = controller.redact_response("engineer", text)
        assert "[REDACTED]" in result
        assert "120,000" not in result

    def test_redacts_ssn_pattern(self, controller):
        text = "ssn: 123-45-6789"
        result = controller.redact_response("engineer", text)
        assert "123-45-6789" not in result
        assert "[REDACTED]" in result

    def test_no_modification_when_nothing_sensitive(self, controller):
        text = "The travel policy allows $350/night in NYC."
        result = controller.redact_response("engineer", text)
        # No sensitive field names present, text should be unchanged
        assert "NYC" in result

    def test_executive_sees_salary(self, controller):
        text = "salary: $300,000"
        result = controller.redact_response("executive", text)
        assert "300,000" in result


# ===========================================================================
# AccessController — audit logging
# ===========================================================================

class TestAuditLogging:
    def test_log_access_appends_entry(self, controller):
        controller.log_access("engineer", "doc_001", True)
        assert len(controller.audit_log) == 1

    def test_log_entry_has_required_fields(self, controller):
        controller.log_access("hr", "doc_salary", False)
        entry = controller.audit_log[-1]
        assert "timestamp" in entry
        assert entry["role"] == "hr"
        assert entry["resource"] == "doc_salary"
        assert entry["allowed"] is False

    def test_log_entry_with_field(self, controller):
        controller.log_access("engineer", "employee_record", False, field="ssn")
        entry = controller.audit_log[-1]
        assert entry["field"] == "ssn"

    def test_get_audit_log_returns_all(self, controller):
        controller.log_access("engineer", "a", True)
        controller.log_access("manager",  "b", False)
        log = controller.get_audit_log()
        assert len(log) >= 2

    def test_filter_documents_logs_each(self, controller):
        docs = [PUBLIC_DOC, CONFIDENTIAL_DOC]
        controller.filter_documents("engineer", docs)
        assert len(controller.audit_log) == 2


# ===========================================================================
# AccessController — filter_documents
# ===========================================================================

class TestFilterDocuments:
    def test_engineer_filtered_to_public_and_internal(self, controller):
        docs = [PUBLIC_DOC, INTERNAL_DOC, CONFIDENTIAL_DOC, RESTRICTED_DOC]
        result = controller.filter_documents("engineer", docs)
        ids = [d["id"] for d in result]
        assert "pub1" in ids
        assert "int1" in ids
        assert "con1" not in ids
        assert "res1" not in ids

    def test_executive_sees_all(self, controller):
        docs = [PUBLIC_DOC, INTERNAL_DOC, CONFIDENTIAL_DOC, RESTRICTED_DOC]
        result = controller.filter_documents("executive", docs)
        assert len(result) == 4

    def test_empty_list_returns_empty(self, controller):
        assert controller.filter_documents("engineer", []) == []


# ===========================================================================
# RateLimiter
# ===========================================================================

class TestRateLimiter:
    def test_allows_within_limit(self, rate_limiter):
        for _ in range(5):
            assert rate_limiter.is_allowed("user1")

    def test_blocks_after_limit(self, rate_limiter):
        for _ in range(5):
            rate_limiter.is_allowed("user1")
        assert not rate_limiter.is_allowed("user1")

    def test_different_users_independent(self, rate_limiter):
        for _ in range(5):
            rate_limiter.is_allowed("userA")
        # userA is blocked but userB is not
        assert not rate_limiter.is_allowed("userA")
        assert rate_limiter.is_allowed("userB")

    def test_remaining_queries_decrements(self, rate_limiter):
        assert rate_limiter.get_remaining_queries("new_user") == 5
        rate_limiter.is_allowed("new_user")
        assert rate_limiter.get_remaining_queries("new_user") == 4

    def test_remaining_queries_never_negative(self, rate_limiter):
        for _ in range(10):
            rate_limiter.is_allowed("overflow_user")
        assert rate_limiter.get_remaining_queries("overflow_user") == 0

    def test_window_resets_after_60s(self, rate_limiter):
        # Exhaust limit
        for _ in range(5):
            rate_limiter.is_allowed("time_user")
        assert not rate_limiter.is_allowed("time_user")

        # Manually backdating timestamps simulates window expiry
        rate_limiter.user_query_times["time_user"] = [time.time() - 61] * 5
        assert rate_limiter.is_allowed("time_user")


# ===========================================================================
# CostEnforcer
# ===========================================================================

class TestCostEnforcer:
    def test_initialises_with_default_budgets(self, cost_enforcer):
        assert cost_enforcer.role_budgets["engineer"] == 100.0
        assert cost_enforcer.role_budgets["executive"] == 1000.0

    def test_can_afford_when_no_spending(self, cost_enforcer):
        assert cost_enforcer.can_afford_query("u1", 50.0)

    def test_can_afford_within_budget(self, cost_enforcer):
        cost_enforcer.add_cost("u1", "engineer", 50.0)
        assert cost_enforcer.can_afford_query("u1", 49.99)

    def test_blocks_when_budget_exceeded(self, cost_enforcer):
        cost_enforcer.add_cost("u1", "engineer", 50.0)
        assert not cost_enforcer.can_afford_query("u1", 51.0)

    def test_exact_budget_remaining_is_allowed(self, cost_enforcer):
        cost_enforcer.add_cost("u1", "engineer", 90.0)
        assert cost_enforcer.can_afford_query("u1", 10.0)

    def test_one_cent_over_budget_is_blocked(self, cost_enforcer):
        cost_enforcer.add_cost("u1", "engineer", 90.0)
        assert not cost_enforcer.can_afford_query("u1", 10.01)

    def test_add_cost_accumulates(self, cost_enforcer):
        cost_enforcer.add_cost("u2", "manager", 100.0)
        cost_enforcer.add_cost("u2", "manager", 200.0)
        assert cost_enforcer.user_spending["u2"]["total"] == 300.0

    def test_get_budget_remaining(self, cost_enforcer):
        cost_enforcer.add_cost("u3", "engineer", 30.0)
        assert cost_enforcer.get_budget_remaining("u3") == pytest.approx(70.0)

    def test_budget_remaining_never_negative(self, cost_enforcer):
        cost_enforcer.add_cost("u4", "engineer", 200.0)
        assert cost_enforcer.get_budget_remaining("u4") == 0.0

    def test_manager_has_higher_budget(self, cost_enforcer):
        cost_enforcer.add_cost("mgr1", "manager", 400.0)
        assert cost_enforcer.can_afford_query("mgr1", 99.0)

    def test_different_users_independent(self, cost_enforcer):
        cost_enforcer.add_cost("ua", "engineer", 99.0)
        cost_enforcer.add_cost("ub", "engineer", 0.0)
        assert not cost_enforcer.can_afford_query("ua", 2.0)
        assert cost_enforcer.can_afford_query("ub", 50.0)
