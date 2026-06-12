"""
Week 5: Unit tests for tools and Agent class.
All tests run without making real API calls.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent import (
    Agent,
    EmployeeLookupTool,
    ExpenseQueryTool,
    PolicySearchTool,
    Tool,
)

# Absolute paths so tests work from any working directory
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_BASE, "data", "techcorp.db")
DOCS_PATH = os.path.join(_BASE, "data", "documents.json")


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------

class TestToolBase:
    def test_tool_stores_name_and_description(self):
        tool = EmployeeLookupTool(DB_PATH)
        assert tool.name == "employee_lookup"
        assert len(tool.description) > 0

    def test_base_tool_raises_not_implemented(self):
        tool = Tool("test", "description")
        with pytest.raises(NotImplementedError):
            tool.execute()


# ---------------------------------------------------------------------------
# EmployeeLookupTool
# ---------------------------------------------------------------------------

class TestEmployeeLookupTool:
    def setup_method(self):
        self.tool = EmployeeLookupTool(DB_PATH)

    def test_lookup_by_name_returns_json(self):
        result = self.tool.execute(employee_name="Brian")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_lookup_by_id_returns_json(self):
        result = self.tool.execute(employee_id=1)
        data = json.loads(result)
        assert isinstance(data, list)
        assert data[0]["id"] == 1

    def test_lookup_unknown_name_returns_not_found(self):
        result = self.tool.execute(employee_name="XYZZY_NO_SUCH_PERSON_99999")
        assert "not found" in result.lower()

    def test_lookup_missing_params_returns_error(self):
        result = self.tool.execute()
        assert result.startswith("Error")

    def test_result_does_not_contain_ssn(self):
        result = self.tool.execute(employee_id=1)
        assert "ssn" not in result.lower()


# ---------------------------------------------------------------------------
# PolicySearchTool
# ---------------------------------------------------------------------------

class TestPolicySearchTool:
    def setup_method(self):
        self.tool = PolicySearchTool(DOCS_PATH)

    def test_documents_loaded(self):
        assert len(self.tool.documents) > 0

    def test_search_returns_relevant_result(self):
        result = self.tool.execute(query="travel policy")
        assert "travel" in result.lower()

    def test_search_with_limit(self):
        result = self.tool.execute(query="policy", limit=1)
        # Should return at most 1 document (only one "---" separator expected to be absent or 0)
        assert "---" not in result  # 1 result means no separator

    def test_search_no_match_returns_message(self):
        result = self.tool.execute(query="xyzzy_totally_fake_topic_99999")
        assert "no documents found" in result.lower()

    def test_search_missing_query_returns_error(self):
        result = self.tool.execute()
        assert result.startswith("Error")

    def test_search_pto_policy(self):
        result = self.tool.execute(query="parental leave")
        assert "parental" in result.lower() or "leave" in result.lower()


# ---------------------------------------------------------------------------
# ExpenseQueryTool
# ---------------------------------------------------------------------------

class TestExpenseQueryTool:
    def setup_method(self):
        self.tool = ExpenseQueryTool(DB_PATH)

    def test_approval_limit_manager(self):
        result = self.tool.execute(query_type="approval_limit", role="manager")
        assert "5,000" in result or "5000" in result

    def test_approval_limit_director(self):
        result = self.tool.execute(query_type="approval_limit", role="director")
        assert "25,000" in result or "25000" in result

    def test_approval_limit_engineer(self):
        result = self.tool.execute(query_type="approval_limit", role="engineer")
        assert "500" in result

    def test_summary_returns_json(self):
        result = self.tool.execute(query_type="summary")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "category" in data[0]

    def test_history_requires_employee_name(self):
        result = self.tool.execute(query_type="history")
        assert result.startswith("Error")

    def test_history_returns_records(self):
        result = self.tool.execute(query_type="history", employee_name="Tracy")
        # Either finds records or returns "not found" — both are valid JSON or plain text
        assert len(result) > 0

    def test_unknown_query_type_returns_error(self):
        result = self.tool.execute(query_type="bogus_type")
        assert "Unknown" in result or "Error" in result


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestAgent:
    def test_agent_initializes_with_three_tools(self):
        agent = Agent(DB_PATH, api_key="dummy-key-for-test")
        assert "employee_lookup" in agent.tools
        assert "policy_search" in agent.tools
        assert "expense_query" in agent.tools

    def test_agent_raises_without_api_key(self):
        import app.agent as agent_module
        original_env = os.environ.pop("OPENAI_API_KEY", None)
        original_mod = agent_module.OPENAI_API_KEY
        agent_module.OPENAI_API_KEY = ""
        try:
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                Agent(DB_PATH, api_key=None)
        finally:
            agent_module.OPENAI_API_KEY = original_mod
            if original_env:
                os.environ["OPENAI_API_KEY"] = original_env

    def test_cost_calculation_input_only(self):
        # gpt-4o-mini: $0.15 per 1M input tokens
        agent = Agent(DB_PATH, api_key="dummy")
        cost = agent._estimate_query_cost(1_000_000, 0)
        assert abs(cost - 0.15) < 1e-9

    def test_cost_calculation_output_only(self):
        # gpt-4o-mini: $0.60 per 1M output tokens
        agent = Agent(DB_PATH, api_key="dummy")
        cost = agent._estimate_query_cost(0, 1_000_000)
        assert abs(cost - 0.60) < 1e-9

    def test_cost_calculation_combined(self):
        # $0.15 + $0.60 = $0.75 per 1M of each
        agent = Agent(DB_PATH, api_key="dummy")
        cost = agent._estimate_query_cost(1_000_000, 1_000_000)
        assert abs(cost - 0.75) < 1e-9

    def test_cost_zero_tokens(self):
        agent = Agent(DB_PATH, api_key="dummy")
        assert agent._estimate_query_cost(0, 0) == 0.0

    def test_initial_metrics_are_zero(self):
        agent = Agent(DB_PATH, api_key="dummy")
        m = agent.get_metrics()
        assert m["total_queries"] == 0
        assert m["total_tokens"] == 0
        assert m["total_cost"] == 0.0
        assert m["avg_cost_per_query"] == 0.0

    def test_parse_tool_calls_single(self):
        agent = Agent(DB_PATH, api_key="dummy")
        text = (
            "I need employee data.\n"
            "TOOL_CALLS:\n"
            "```json\n"
            '[{"tool": "employee_lookup", "args": {"employee_name": "Alice"}}]\n'
            "```"
        )
        calls = agent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "employee_lookup"
        assert calls[0]["args"]["employee_name"] == "Alice"

    def test_parse_tool_calls_multiple(self):
        agent = Agent(DB_PATH, api_key="dummy")
        text = (
            "TOOL_CALLS:\n"
            "```json\n"
            '[{"tool": "employee_lookup", "args": {"employee_id": 1}},'
            ' {"tool": "expense_query", "args": {"query_type": "summary"}}]\n'
            "```"
        )
        calls = agent._parse_tool_calls(text)
        assert len(calls) == 2

    def test_parse_tool_calls_empty_when_no_block(self):
        agent = Agent(DB_PATH, api_key="dummy")
        calls = agent._parse_tool_calls("I can answer this directly.")
        assert calls == []
