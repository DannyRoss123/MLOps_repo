"""
Week 6: TechCorp AI Agent with Access Control Guardrails

Extends the Week 5 agent architecture with:
- Role-based document filtering (AccessController)
- Per-user rate limiting (RateLimiter)
- Per-role budget enforcement (CostEnforcer)
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.access_control import AccessController, RateLimiter, CostEnforcer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini pricing (per 1M tokens)
_INPUT_COST_PER_M = 0.075
_OUTPUT_COST_PER_M = 0.30
_ESTIMATED_QUERY_COST = 0.01   # conservative pre-flight estimate


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * _INPUT_COST_PER_M + (output_tokens / 1_000_000) * _OUTPUT_COST_PER_M


class Agent:
    """AI agent with access-control, rate-limiting, and cost-enforcement guardrails."""

    def __init__(
        self,
        corpus_path: str = "data/corpus_updated.json",
        access_policy_path: str = "data/access_control.json",
        api_key: Optional[str] = None,
        max_queries_per_minute: int = 30,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY", "")

        # Load document corpus
        corpus_file = Path(corpus_path)
        if corpus_file.exists():
            with open(corpus_file, "r", encoding="utf-8") as f:
                self._corpus: List[Dict[str, Any]] = json.load(f)
        else:
            logger.warning("Corpus not found at %s; starting empty.", corpus_path)
            self._corpus = []

        # Guardrails
        self.access_controller = AccessController(access_policy_path)
        self.rate_limiter = RateLimiter(max_queries_per_minute=max_queries_per_minute)
        self.cost_enforcer = CostEnforcer()

        # Metrics
        self._total_queries = 0
        self._total_tokens = 0
        self._total_cost = 0.0

        # Optional LLM client
        self._llm_client = None
        if self.api_key:
            try:
                import google.genai as genai
                self._llm_client = genai.Client(api_key=self.api_key)
                logger.info("Gemini client initialized.")
            except ImportError:
                logger.warning("google-genai not installed; running in keyword-search mode.")

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        user_id: str = "anonymous",
        user_role: str = "engineer",
    ) -> Dict[str, Any]:
        """Answer *user_query* subject to all guardrails.

        Returns a dict with keys: answer, tokens_used, cost, role, denied_reason.
        """
        # 1. Rate limit check
        if not self.rate_limiter.is_allowed(user_id):
            remaining = self.rate_limiter.get_remaining_queries(user_id)
            return {
                "answer": None,
                "tokens_used": 0,
                "cost": 0.0,
                "role": user_role,
                "denied_reason": f"Rate limit exceeded. Remaining queries: {remaining}",
            }

        # 2. Budget pre-flight check
        if not self.cost_enforcer.can_afford_query(user_id, _ESTIMATED_QUERY_COST):
            remaining_budget = self.cost_enforcer.get_budget_remaining(user_id)
            return {
                "answer": None,
                "tokens_used": 0,
                "cost": 0.0,
                "role": user_role,
                "denied_reason": f"Budget exceeded. Remaining budget: ${remaining_budget:.4f}",
            }

        # 3. Filter corpus to role-appropriate documents
        visible_docs = self.access_controller.filter_documents(user_role, self._corpus)

        # 4. Retrieve relevant documents via keyword search
        relevant_docs = self._search(user_query, visible_docs)

        # 5. Generate answer
        if self._llm_client and relevant_docs:
            answer, input_tokens, output_tokens = self._llm_answer(user_query, user_role, relevant_docs)
        else:
            answer, input_tokens, output_tokens = self._keyword_answer(user_query, user_role, relevant_docs)

        actual_cost = _estimate_cost(input_tokens, output_tokens)

        # 6. Redact sensitive content the role cannot view
        answer = self.access_controller.redact_response(user_role, answer)

        # 7. Record cost and update metrics
        self.cost_enforcer.add_cost(user_id, user_role, actual_cost)
        self._total_queries += 1
        self._total_tokens += input_tokens + output_tokens
        self._total_cost += actual_cost

        return {
            "answer": answer,
            "tokens_used": input_tokens + output_tokens,
            "cost": actual_cost,
            "role": user_role,
            "denied_reason": None,
            "docs_retrieved": len(relevant_docs),
            "budget_remaining": self.cost_enforcer.get_budget_remaining(user_id),
        }

    # ------------------------------------------------------------------
    # Document search
    # ------------------------------------------------------------------

    def _search(self, query: str, docs: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword-overlap ranking over *docs*."""
        query_terms = set(query.lower().split())
        scored: List[tuple] = []
        for doc in docs:
            text = (doc.get("title", "") + " " + doc.get("content", "")).lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_n]]

    # ------------------------------------------------------------------
    # Answer generation (LLM path)
    # ------------------------------------------------------------------

    def _llm_answer(
        self,
        query: str,
        role: str,
        docs: List[Dict[str, Any]],
    ) -> tuple:
        """Call Gemini and return (answer_text, input_tokens, output_tokens)."""
        import google.genai as genai

        context = "\n\n".join(
            f"[{doc.get('title', 'Document')}]\n{doc.get('content', '')[:1000]}"
            for doc in docs
        )
        system_prompt = (
            f"You are TechCorp's internal assistant. The user has role '{role}'. "
            "Answer using ONLY the provided context. Be concise."
        )
        full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            response = self._llm_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=full_prompt,
            )
            answer = response.text or ""
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", len(full_prompt.split()))
            output_tokens = getattr(usage, "candidates_token_count", len(answer.split()))
            return answer, input_tokens, output_tokens
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return self._keyword_answer(query, role, docs)

    # ------------------------------------------------------------------
    # Answer generation (fallback keyword path)
    # ------------------------------------------------------------------

    def _keyword_answer(
        self,
        query: str,
        role: str,
        docs: List[Dict[str, Any]],
    ) -> tuple:
        """Build a plain-text answer from retrieved document snippets."""
        if not docs:
            answer = (
                f"No documents accessible for role '{role}' matched your query. "
                "Try a different query or contact your administrator for access."
            )
            return answer, 0, 0

        snippets = []
        for doc in docs[:3]:
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")[:500].strip()
            snippets.append(f"**{title}**\n{content}")

        answer = f"Based on accessible documents for role '{role}':\n\n" + "\n\n---\n\n".join(snippets)
        approx_tokens = len(answer.split())
        return answer, len(query.split()), approx_tokens

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return cumulative usage metrics."""
        avg = self._total_cost / self._total_queries if self._total_queries else 0.0
        return {
            "total_queries": self._total_queries,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "avg_cost_per_query": avg,
        }


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="TechCorp Agent API (Week 6)", version="1.0.0")

    # Agent is initialised once at module load.
    _agent: Optional[Agent] = None


    def get_agent() -> Agent:
        global _agent
        if _agent is None:
            _agent = Agent()
        return _agent


    class QueryRequest(BaseModel):
        query: str
        user_id: str = "anonymous"
        role: str = "engineer"


    @app.post("/query")
    def run_query(request: QueryRequest):
        agent = get_agent()
        result = agent.query(request.query, user_id=request.user_id, user_role=request.role)
        if result.get("denied_reason"):
            raise HTTPException(status_code=429, detail=result["denied_reason"])
        return result

    @app.get("/metrics")
    def metrics():
        return get_agent().get_metrics()

    @app.get("/health")
    def health():
        return {"status": "ok"}

except ImportError:
    pass  # FastAPI optional; core logic still works


# ---------------------------------------------------------------------------
# Manual smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Week 6: Access Control Demo ===\n")

    base = Path(__file__).parent.parent
    agent = Agent(
        corpus_path=str(base / "data" / "corpus_updated.json"),
        access_policy_path=str(base / "data" / "access_control.json"),
    )

    test_cases = [
        ("user_engineer", "engineer", "What is the travel policy?"),
        ("user_engineer", "engineer", "What are salary ranges?"),
        ("user_hr",       "hr",       "What are salary ranges?"),
        ("user_exec",     "executive","What are the department budgets?"),
    ]

    for user_id, role, query in test_cases:
        result = agent.query(query, user_id=user_id, user_role=role)
        print(f"[{role}] Q: {query}")
        if result["denied_reason"]:
            print(f"  DENIED: {result['denied_reason']}")
        else:
            preview = (result["answer"] or "")[:200].replace("\n", " ")
            print(f"  A: {preview}...")
            print(f"  docs={result.get('docs_retrieved',0)}  cost=${result['cost']:.6f}  budget_left=${result['budget_remaining']:.2f}")
        print()

    print("Rate limiting demo:")
    limiter_agent = Agent(
        corpus_path=str(base / "data" / "corpus_updated.json"),
        access_policy_path=str(base / "data" / "access_control.json"),
        max_queries_per_minute=3,
    )
    for i in range(1, 6):
        r = limiter_agent.query("test", user_id="tester", user_role="engineer")
        status = "BLOCKED" if r["denied_reason"] else "OK"
        print(f"  Query {i}: {status}")
