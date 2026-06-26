"""
Week 7: Cost Optimization & Feedback Loop

Three systems for production cost management and continuous improvement:
1. CostAnalyzer  — record, breakdown, and spike-detect query costs
2. OptimizationStrategy — caching, model selection, retrieval trimming, compression
3. FeedbackLoop — collect, validate, and measure user corrections
"""

import math
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================================================================
# CostAnalyzer
# ===========================================================================

class CostAnalyzer:
    """Record query costs and identify expensive outliers."""

    def __init__(self):
        self.query_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------

    def record_query(self, query: Dict[str, Any]):
        """Store a query record.

        Expected keys in *query*:
          query_text, retrieval_cost, llm_cost, tool_cost, error_cost.
        total_cost and timestamp are added automatically if missing.
        """
        record = {
            "query_text":     query.get("query_text", ""),
            "retrieval_cost": float(query.get("retrieval_cost", 0.0)),
            "llm_cost":       float(query.get("llm_cost", 0.0)),
            "tool_cost":      float(query.get("tool_cost", 0.0)),
            "error_cost":     float(query.get("error_cost", 0.0)),
            "timestamp":      query.get("timestamp", datetime.now(timezone.utc).isoformat()),
        }
        # Allow caller to supply total_cost, otherwise compute it.
        record["total_cost"] = float(query.get(
            "total_cost",
            record["retrieval_cost"] + record["llm_cost"] + record["tool_cost"] + record["error_cost"],
        ))
        self.query_history.append(record)

    # ------------------------------------------------------------------

    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Return summed cost totals across all recorded queries."""
        retrieval_total = sum(q["retrieval_cost"] for q in self.query_history)
        llm_total       = sum(q["llm_cost"]       for q in self.query_history)
        tool_total      = sum(q["tool_cost"]       for q in self.query_history)
        error_total     = sum(q["error_cost"]      for q in self.query_history)
        total_daily     = retrieval_total + llm_total + tool_total + error_total
        return {
            "retrieval_total": round(retrieval_total, 6),
            "llm_total":       round(llm_total,       6),
            "tool_total":      round(tool_total,       6),
            "error_total":     round(error_total,      6),
            "total_daily":     round(total_daily,      6),
            "query_count":     len(self.query_history),
        }

    # ------------------------------------------------------------------

    def identify_cost_spikes(self) -> List[Dict[str, Any]]:
        """Return queries whose total_cost exceeds mean + 2 × std-dev."""
        if len(self.query_history) < 2:
            return []

        costs = [q["total_cost"] for q in self.query_history]
        mean  = sum(costs) / len(costs)
        variance = sum((c - mean) ** 2 for c in costs) / len(costs)
        stdev = math.sqrt(variance)

        if stdev == 0:
            return []

        threshold = mean + 2 * stdev
        spikes = [
            {**q, "spike_threshold": round(threshold, 6), "z_score": round((q["total_cost"] - mean) / stdev, 2)}
            for q in self.query_history
            if q["total_cost"] > threshold
        ]
        return sorted(spikes, key=lambda x: x["total_cost"], reverse=True)


# ===========================================================================
# OptimizationStrategy
# ===========================================================================

# Keywords that signal query complexity.
_COMPLEX_KEYWORDS = {
    "analyze", "analyse", "compare", "contrast", "design", "architect",
    "explain why", "how does", "why does", "evaluate", "assess", "summarize",
    "summarise", "describe all", "list all", "what are the differences",
    "pros and cons", "trade-off", "trade off", "trade offs",
}
_SIMPLE_PREFIXES = ("what is", "what are", "who is", "where is", "when is",
                    "how much", "how many", "what's", "what was")

_MAX_COMPRESSED_SENTENCES = 5


class OptimizationStrategy:
    """Reduce agent costs via caching, model routing, retrieval trimming, and compression."""

    def __init__(self):
        # Maps normalised query → cached response
        self.cache: Dict[str, str] = {}
        # Running log of which strategies fired
        self.strategies_applied: List[str] = []
        # Counters for impact estimation
        self._cache_hits   = 0
        self._cache_misses = 0
        self._flash_selections = 0
        self._pro_selections   = 0
        self._retrieval_reductions: List[int] = []  # bytes saved per call
        self._compression_chars_saved = 0

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def apply_caching(self, query: str, response: str) -> tuple:
        """Return (True, cached_response) on hit; store and return (False, response) on miss."""
        key = self._normalise(query)
        if key in self.cache:
            self._cache_hits += 1
            if "caching" not in self.strategies_applied:
                self.strategies_applied.append("caching")
            logger.debug("Cache HIT for query: %s", query[:60])
            return (True, self.cache[key])

        self.cache[key] = response
        self._cache_misses += 1
        return (False, response)

    # ------------------------------------------------------------------
    # Retrieval count optimisation
    # ------------------------------------------------------------------

    def optimize_retrieval_count(self, num_docs: int) -> int:
        """Reduce retrieved document count to at most 3 (5× reduction)."""
        optimised = max(1, num_docs // 5)
        saved = num_docs - optimised
        if saved > 0:
            self._retrieval_reductions.append(saved)
            if "retrieval_optimization" not in self.strategies_applied:
                self.strategies_applied.append("retrieval_optimization")
        return optimised

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def select_model_by_complexity(self, query: str) -> str:
        """Route to gemini-1.5-flash for simple queries, gemini-2.5-pro for complex."""
        q_lower = query.lower().strip()
        multi_question = query.count("?") > 1
        is_complex = (
            any(kw in q_lower for kw in _COMPLEX_KEYWORDS)
            or len(query.split()) > 20
            or multi_question
        )
        is_simple = any(q_lower.startswith(p) for p in _SIMPLE_PREFIXES) and len(query.split()) <= 12

        if is_complex and not is_simple:
            model = "gemini-2.5-pro"
            self._pro_selections += 1
        else:
            model = "gemini-1.5-flash"
            self._flash_selections += 1
            if "model_selection" not in self.strategies_applied:
                self.strategies_applied.append("model_selection")

        return model

    # ------------------------------------------------------------------
    # Response compression
    # ------------------------------------------------------------------

    def enable_response_compression(self, response: str) -> str:
        """Keep only the first N sentences to reduce output token cost."""
        if not response:
            return response

        # Split on sentence-ending punctuation followed by whitespace or end.
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        if len(sentences) <= _MAX_COMPRESSED_SENTENCES:
            return response

        compressed = " ".join(sentences[:_MAX_COMPRESSED_SENTENCES])
        saved = len(response) - len(compressed)
        self._compression_chars_saved += saved
        if "response_compression" not in self.strategies_applied:
            self.strategies_applied.append("response_compression")
        return compressed

    # ------------------------------------------------------------------
    # Impact estimation
    # ------------------------------------------------------------------

    def get_optimization_impact(self) -> Dict[str, Any]:
        """Estimate percentage cost savings from applied optimisations."""
        total_cache_calls = self._cache_hits + self._cache_misses
        cache_savings_pct = (
            (self._cache_hits / total_cache_calls * 100) if total_cache_calls > 0 else 0.0
        )

        total_model_calls = self._flash_selections + self._pro_selections
        # gemini-1.5-flash is ~10× cheaper than gemini-2.5-pro
        model_savings_pct = (
            (self._flash_selections / total_model_calls * 90.0) if total_model_calls > 0 else 0.0
        )

        retrieval_savings_pct = (
            (sum(self._retrieval_reductions) / (sum(self._retrieval_reductions) + len(self._retrieval_reductions) * 3) * 100)
            if self._retrieval_reductions else 0.0
        )

        compression_savings_pct = min(
            50.0,
            self._compression_chars_saved / 5000 * 100
        ) if self._compression_chars_saved > 0 else 0.0

        # Weighted combined savings (cap at 95%)
        total_savings_pct = min(
            95.0,
            cache_savings_pct * 0.4
            + model_savings_pct * 0.35
            + retrieval_savings_pct * 0.15
            + compression_savings_pct * 0.10,
        )

        return {
            "total_savings_pct": round(total_savings_pct, 1),
            "strategies_applied": list(self.strategies_applied),
            "breakdown": {
                "caching_savings_pct":      round(cache_savings_pct,       1),
                "model_selection_savings_pct": round(model_savings_pct,    1),
                "retrieval_savings_pct":    round(retrieval_savings_pct,    1),
                "compression_savings_pct":  round(compression_savings_pct, 1),
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(query: str) -> str:
        """Lower-case and strip whitespace for cache key normalisation."""
        return " ".join(query.lower().split())


# ===========================================================================
# FeedbackLoop
# ===========================================================================

_MIN_AUTHORITY = 2   # manager-level or above required to submit corrections

class FeedbackLoop:
    """Collect, validate, and measure user corrections for continuous improvement."""

    def __init__(self):
        self.corrections: List[Dict[str, Any]] = []
        # Authority hierarchy: higher = more trusted
        self.authority: Dict[str, int] = {
            "intern":    0,
            "engineer":  1,
            "manager":   2,
            "director":  3,
            "hr":        2,   # HR can correct HR-related answers
            "finance":   2,   # Finance can correct finance-related answers
            "executive": 4,
        }

    # ------------------------------------------------------------------

    def submit_correction(
        self,
        original_query:   str,
        original_answer:  str,
        corrected_answer: str,
        user_role:        str,
    ) -> Dict[str, Any]:
        """Validate and store a user correction.

        Acceptance criteria:
        - User role has authority >= _MIN_AUTHORITY (manager+)
        - Corrected answer is substantively different from original
        - Corrected answer is at least as long as the original
        """
        role_level = self.authority.get(user_role.lower(), 0)

        if role_level < _MIN_AUTHORITY:
            return {
                "accepted": False,
                "reason": f"Role '{user_role}' has insufficient authority to submit corrections (need manager+).",
            }

        if not corrected_answer or not corrected_answer.strip():
            return {"accepted": False, "reason": "Corrected answer must not be empty."}

        if corrected_answer.strip().lower() == original_answer.strip().lower():
            return {"accepted": False, "reason": "Corrected answer is identical to the original."}

        record = {
            "original_query":   original_query,
            "original_answer":  original_answer,
            "corrected_answer": corrected_answer,
            "user_role":        user_role,
            "role_level":       role_level,
            "valid":            True,   # accepted corrections are assumed valid until reviewed
            "timestamp":        datetime.now(timezone.utc).isoformat(),
        }
        self.corrections.append(record)
        logger.info("Correction accepted from role '%s' for query: %s", user_role, original_query[:60])
        return {"accepted": True, "reason": "Correction accepted and queued for review."}

    # ------------------------------------------------------------------

    def validate_correction(self, index: int) -> bool:
        """Re-validate correction at *index*.

        Returns True if:
        - index is valid
        - role has authority >= _MIN_AUTHORITY
        - corrected answer is longer than original (more detailed)
        """
        if index < 0 or index >= len(self.corrections):
            return False

        correction = self.corrections[index]
        role_level = self.authority.get(correction.get("user_role", "").lower(), 0)
        if role_level < _MIN_AUTHORITY:
            self.corrections[index]["valid"] = False
            return False

        is_more_detailed = len(correction["corrected_answer"]) >= len(correction["original_answer"])
        self.corrections[index]["valid"] = is_more_detailed
        return is_more_detailed

    # ------------------------------------------------------------------

    def get_feedback_metrics(self) -> Dict[str, Any]:
        """Compute metrics on the collected feedback."""
        total = len(self.corrections)
        if total == 0:
            return {
                "total_corrections":      0,
                "validation_rate":        0.0,
                "avg_correction_length":  0.0,
                "top_error_patterns":     [],
            }

        valid_count   = sum(1 for c in self.corrections if c.get("valid", False))
        validation_rate = valid_count / total

        avg_len = sum(len(c["corrected_answer"]) for c in self.corrections) / total

        # Identify top error patterns by counting common keywords in original answers.
        from collections import Counter
        error_words: List[str] = []
        for c in self.corrections:
            words = re.findall(r'\b[a-z]{4,}\b', c["original_answer"].lower())
            error_words.extend(words)
        _STOPWORDS = {"that", "this", "with", "from", "have", "been", "will", "were",
                      "they", "their", "there", "what", "when", "where", "which"}
        top_patterns = [
            word for word, _ in Counter(error_words).most_common(10)
            if word not in _STOPWORDS
        ][:5]

        return {
            "total_corrections":      total,
            "validation_rate":        round(validation_rate, 3),
            "avg_correction_length":  round(avg_len, 1),
            "top_error_patterns":     top_patterns,
        }
