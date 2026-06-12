"""
Week 5: Agent Architecture
AI agent that answers TechCorp questions using OpenAI GPT + tools.
"""

import json
import os
import re
import sqlite3
import logging
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# gpt-4o-mini: cheapest capable OpenAI model
DEFAULT_MODEL = "gpt-4o-mini"
INPUT_COST_PER_M = 0.15    # $0.15 per 1M input tokens
OUTPUT_COST_PER_M = 0.60   # $0.60 per 1M output tokens


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------

class Tool:
    """Base class for tools the agent can call."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Tool 1: Employee lookup
# ---------------------------------------------------------------------------

class EmployeeLookupTool(Tool):
    """Look up employee information from SQLite database."""

    def __init__(self, db_path: str):
        super().__init__(
            "employee_lookup",
            (
                "Find employee information by name or ID. "
                "Args: employee_name (str, optional), employee_id (int, optional). "
                "Returns employee details as JSON."
            ),
        )
        self.db_path = db_path

    def execute(self, employee_name: str = None, employee_id=None) -> str:
        if not employee_name and not employee_id:
            return "Error: provide either employee_name or employee_id"
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Redact sensitive fields (SSN, address)
            select = (
                "SELECT id, name, email, department_name, job_level, title, "
                "hire_date, bonus_eligible, stock_options FROM employees"
            )
            if employee_id:
                cursor.execute(f"{select} WHERE id = ?", (employee_id,))
            else:
                cursor.execute(f"{select} WHERE name LIKE ?", (f"%{employee_name}%",))
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                return "Employee not found"
            return json.dumps([dict(r) for r in rows], indent=2)
        except Exception as e:
            logger.error("Employee lookup error: %s", e)
            return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool 2: Policy document search
# ---------------------------------------------------------------------------

class PolicySearchTool(Tool):
    """Search policy documents by keyword."""

    def __init__(self, documents_path: str = None):
        super().__init__(
            "policy_search",
            (
                "Search TechCorp policy documents by keyword or topic. "
                "Args: query (str), limit (int, optional, default 3). "
                "Returns matching document titles and content snippets."
            ),
        )
        self.documents_path = documents_path or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'documents.json'
        )
        self.documents = self._load_documents()

    def _load_documents(self) -> List[Dict]:
        try:
            with open(self.documents_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not load documents: %s", e)
            return []

    def execute(self, query: str = None, limit: int = 3) -> str:
        if not query:
            return "Error: query parameter is required"
        try:
            query_lower = query.lower()
            words = [w for w in query_lower.split() if len(w) > 2]
            scored = []
            for doc in self.documents:
                content = doc.get('content', '').lower()
                title = doc.get('title', '').lower()
                score = content.count(query_lower) * 5 + title.count(query_lower) * 10
                for word in words:
                    score += content.count(word) + title.count(word) * 3
                if score > 0:
                    scored.append((score, doc))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:int(limit)]
            if not top:
                return f"No documents found matching '{query}'"
            parts = []
            for _, doc in top:
                snippet = doc.get('content', '')[:600].strip()
                parts.append(
                    f"**{doc['title']}** (Category: {doc['category']}, "
                    f"Sensitivity: {doc.get('sensitivity', 'Unknown')})\n{snippet}"
                )
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.error("Policy search error: %s", e)
            return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tool 3: Expense query
# ---------------------------------------------------------------------------

class ExpenseQueryTool(Tool):
    """Query expense records and approval limits."""

    # Approval limits from Budget Guidelines doc (doc_fin_002)
    _LIMITS = {
        "ic1": 500, "ic2": 500, "junior": 500,
        "ic3": 2000, "senior": 2000,
        "manager": 5000, "director": 25000,
        "vp": 100000, "executive": 100000,
        "cfo": 10_000_000,
        "engineer": 500,
    }

    def __init__(self, db_path: str):
        super().__init__(
            "expense_query",
            (
                "Query expense records and approval limits. "
                "Args: query_type (str: 'history'|'summary'|'approval_limit'), "
                "employee_name (str, optional), role (str, optional), "
                "category (str, optional). "
                "Use 'approval_limit' with role to get spending limits."
            ),
        )
        self.db_path = db_path

    def execute(
        self,
        query_type: str = "summary",
        employee_name: str = None,
        role: str = None,
        category: str = None,
    ) -> str:
        try:
            if query_type == "approval_limit":
                key = (role or "engineer").lower()
                for k, limit in self._LIMITS.items():
                    if k in key or key in k:
                        return f"Approval limit for {role or 'engineer'}: ${limit:,}"
                return f"Approval limit for {role}: $500 (default IC limit)"

            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if query_type == "history":
                if not employee_name:
                    conn.close()
                    return "Error: employee_name required for history query"
                cursor.execute(
                    "SELECT amount, category, description, date, status, vendor "
                    "FROM expenses WHERE employee_name LIKE ? "
                    "ORDER BY date DESC LIMIT 10",
                    (f"%{employee_name}%",),
                )
                rows = cursor.fetchall()
                conn.close()
                if not rows:
                    return f"No expenses found for {employee_name}"
                return json.dumps([dict(r) for r in rows], indent=2)

            elif query_type == "summary":
                if category:
                    cursor.execute(
                        "SELECT category, COUNT(*) as count, "
                        "ROUND(SUM(amount),2) as total, ROUND(AVG(amount),2) as avg "
                        "FROM expenses WHERE category = ? GROUP BY category",
                        (category,),
                    )
                else:
                    cursor.execute(
                        "SELECT category, COUNT(*) as count, "
                        "ROUND(SUM(amount),2) as total, ROUND(AVG(amount),2) as avg "
                        "FROM expenses GROUP BY category ORDER BY total DESC LIMIT 10"
                    )
                rows = cursor.fetchall()
                conn.close()
                if not rows:
                    return "No expense data found"
                return json.dumps([dict(r) for r in rows], indent=2)

            else:
                conn.close()
                return (
                    f"Unknown query_type '{query_type}'. "
                    "Use 'history', 'summary', or 'approval_limit'."
                )
        except Exception as e:
            logger.error("Expense query error: %s", e)
            return f"Error: {e}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """AI agent that answers questions using OpenAI GPT + tools."""

    def __init__(self, db_path: str, api_key: str = None, model: str = DEFAULT_MODEL):
        self.db_path = db_path
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Get a key at https://platform.openai.com/api-keys"
            )

        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

        data_dir = os.path.dirname(db_path)
        docs_path = os.path.join(data_dir, 'documents.json')

        self.tools: Dict[str, Tool] = {
            "employee_lookup": EmployeeLookupTool(db_path),
            "policy_search": PolicySearchTool(docs_path),
            "expense_query": ExpenseQueryTool(db_path),
        }

        # Metrics
        self.total_queries = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, user_query: str, user_role: str = "engineer") -> Dict[str, Any]:
        """Answer a question using LLM + tools."""
        logger.info("Processing query: %s", user_query)
        input_tokens = 0
        output_tokens = 0

        try:
            # Step 1: Ask LLM which tools to call
            system_prompt = self._build_system_prompt(user_role)
            step1_msg = (
                "Decide which tools (if any) you need to call to answer this question.\n"
                "If you need tools, respond ONLY with:\n"
                "TOOL_CALLS:\n"
                "```json\n"
                '[{"tool": "tool_name", "args": {"param": "value"}}]\n'
                "```\n"
                "You may list multiple tool calls in the array.\n"
                "If you can answer without tools, answer directly."
            )
            resp1 = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{step1_msg}\n\nUser question: {user_query}"},
                ],
            )
            resp1_text = resp1.choices[0].message.content or ""
            input_tokens += resp1.usage.prompt_tokens
            output_tokens += resp1.usage.completion_tokens

            # Step 2: Execute tool calls found in the response
            tool_calls = self._parse_tool_calls(resp1_text)
            tool_results = self._execute_tools(tool_calls)

            # Step 3: Synthesize final answer
            if tool_results:
                results_block = "\n\n".join(tool_results)
                resp2 = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": resp1_text},
                        {
                            "role": "user",
                            "content": (
                                f"Tool results:\n{results_block}\n\n"
                                "Using the tool results above, provide a clear and helpful "
                                "answer. Be concise and cite specific data where relevant."
                            ),
                        },
                    ],
                )
                final_answer = resp2.choices[0].message.content or resp1_text
                input_tokens += resp2.usage.prompt_tokens
                output_tokens += resp2.usage.completion_tokens
            else:
                final_answer = re.sub(
                    r'TOOL_CALLS:\s*```json.*?```', '', resp1_text, flags=re.DOTALL
                ).strip()

            # Step 4: Track cost
            total_tokens = input_tokens + output_tokens
            cost = self._estimate_query_cost(input_tokens, output_tokens)
            self.total_queries += 1
            self.total_tokens += total_tokens
            self.total_cost += cost

            return {
                "answer": final_answer,
                "tokens_used": total_tokens,
                "cost": cost,
                "role": user_role,
            }

        except Exception as e:
            logger.error("Query error: %s", e)
            return {
                "answer": f"Error processing query: {e}",
                "tokens_used": 0,
                "cost": 0.0,
                "role": user_role,
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Return cumulative performance metrics."""
        return {
            "total_queries": self.total_queries,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_cost_per_query": (
                self.total_cost / self.total_queries
                if self.total_queries > 0
                else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, user_role: str) -> str:
        tool_list = "\n".join(
            f"  - {name}: {tool.description}"
            for name, tool in self.tools.items()
        )
        return (
            "You are TechCorp's AI assistant. You help employees find information "
            "about company policies, colleagues, and expense records.\n"
            f"The current user has role: {user_role}\n\n"
            f"Available tools:\n{tool_list}"
        )

    def _parse_tool_calls(self, text: str) -> List[Dict]:
        """Extract JSON tool-call list from LLM response."""
        match = re.search(
            r'TOOL_CALLS:\s*```json\s*(.*?)\s*```', text, re.DOTALL
        )
        if not match:
            return []
        try:
            calls = json.loads(match.group(1))
            if isinstance(calls, list):
                return calls
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse tool calls JSON: %s", e)
        return []

    def _execute_tools(self, tool_calls: List[Dict]) -> List[str]:
        results = []
        for call in tool_calls:
            name = call.get("tool", "")
            args = call.get("args", {})
            if name in self.tools:
                result = self.tools[name].execute(**args)
                results.append(f"[{name}]\n{result}")
                logger.info("Tool %s returned: %s", name, result[:120])
            else:
                results.append(f"[{name}] Error: tool not found")
        return results

    def _estimate_query_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD based on token counts."""
        return (input_tokens / 1_000_000) * INPUT_COST_PER_M + \
               (output_tokens / 1_000_000) * OUTPUT_COST_PER_M

    # ------------------------------------------------------------------
    # Smoke test
    # ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    try:
        db = os.path.join(os.path.dirname(__file__), '..', 'data', 'techcorp.db')
        agent = Agent(db)
        print("Agent initialized. Model:", agent.model)

        result = agent.query("What is the travel policy?")
        print(f"\nAnswer: {result['answer']}")
        print(f"Tokens: {result['tokens_used']}")
        print(f"Cost:   ${result['cost']:.6f}")
        print(f"\nMetrics: {agent.get_metrics()}")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Smoke test failed")
        sys.exit(1)
