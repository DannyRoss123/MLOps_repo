"""
Week 7: Agent Evaluation Script

Runs all test queries through the cost optimization pipeline and prints:
- Per-query cost breakdown
- Aggregate cost analysis
- Cost spike detection
- Optimization impact (caching, model selection, retrieval, compression)
- Feedback loop examples
"""

import json
import os
import sys
import random

# Ensure week7/ is on the path regardless of working directory
WEEK7_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WEEK7_DIR)

from app.cost_optimization import CostAnalyzer, OptimizationStrategy, FeedbackLoop

QUERIES_PATH = os.path.join(os.path.dirname(__file__), "test_queries.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _separator(title: str = "", width: int = 70):
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * pad)
    else:
        print("=" * width)


def _simulate_query_cost(query: str, role: str) -> dict:
    """
    Simulate realistic query costs without a live LLM.
    Costs vary by query complexity and role (more docs for privileged roles).
    """
    words = len(query.split())
    is_complex = any(kw in query.lower() for kw in
                     ["analyze", "compare", "design", "evaluate", "contrast"])

    base_retrieval = 0.0010 + (0.0005 if role in ("executive", "hr") else 0.0)
    base_llm       = 0.0050 if not is_complex else 0.0200
    base_tool      = 0.0005 if words > 10 else 0.0002
    base_error     = 0.0001 if random.random() < 0.1 else 0.0   # 10% chance of retry cost

    # Add noise
    noise = random.uniform(0.9, 1.1)
    retrieval = round(base_retrieval * noise, 6)
    llm       = round(base_llm       * noise, 6)
    tool      = round(base_tool      * noise, 6)
    error     = round(base_error,              6)
    total     = round(retrieval + llm + tool + error, 6)

    return {
        "query_text":     query,
        "retrieval_cost": retrieval,
        "llm_cost":       llm,
        "tool_cost":      tool,
        "error_cost":     error,
        "total_cost":     total,
    }


def _simulate_response(query: str) -> str:
    """Return a plausible-length synthetic response for the query."""
    snippets = {
        "travel":    "All business travel must be pre-approved by a manager. Domestic trips under 6 hours use economy class. International flights over 8 hours allow business class with VP approval. Hotel limits are $350/night in Tier 1 cities, $250 in Tier 2, and $150 in Tier 3. Meals are capped at $15 breakfast, $20 lunch, and $50 dinner.",
        "expense":   "Spending authorization limits: IC1-IC2 $500, IC3 $2,000, Manager $5,000, Director $25,000, VP $100,000, CFO unlimited. All expenses must be submitted within 30 days. Receipts are required for amounts over $25.",
        "pto":       "Individual contributors receive 15 PTO days per year. Managers receive 20 days and Directors/Executives receive 25 days. PTO resets January 1 and does not roll over. Sick leave (10 days) is separate. Parental leave is 16 weeks for primary caregiver.",
        "default":   "Based on TechCorp policy documentation, the relevant information has been retrieved and summarized above. Please consult HR or your manager for specific guidance. Additional details are available in the employee handbook.",
    }
    for kw, resp in snippets.items():
        if kw in query.lower():
            return resp
    return snippets["default"]


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_evaluation():
    random.seed(42)  # reproducible results

    with open(QUERIES_PATH, "r") as f:
        data = json.load(f)
    queries = data["test_queries"]

    analyzer    = CostAnalyzer()
    optimizer   = OptimizationStrategy()
    feedback    = FeedbackLoop()

    _separator("WEEK 7: AGENT COST EVALUATION")
    print(f"Running {len(queries)} test queries...\n")

    # -----------------------------------------------------------------------
    # Run each query
    # -----------------------------------------------------------------------
    results = []
    for i, q in enumerate(queries, 1):
        question = q["question"]
        role     = q["role"]
        category = q["category"]

        # 1. Model selection
        model = optimizer.select_model_by_complexity(question)

        # 2. Retrieval optimisation
        raw_docs = 15
        opt_docs = optimizer.optimize_retrieval_count(raw_docs)

        # 3. Simulate response
        raw_response = _simulate_response(question)

        # 4. Cache check
        is_hit, cached_response = optimizer.apply_caching(question, raw_response)

        # 5. Response compression
        final_response = optimizer.enable_response_compression(cached_response)

        # 6. Simulate cost (lower if cached)
        cost_record = _simulate_query_cost(question, role)
        if is_hit:
            # Cached queries cost almost nothing (only retrieval overhead)
            cost_record["llm_cost"]       = 0.0
            cost_record["tool_cost"]      = 0.0
            cost_record["total_cost"]     = cost_record["retrieval_cost"] * 0.1

        analyzer.record_query(cost_record)
        results.append({**q, **cost_record, "model": model, "cached": is_hit, "docs": opt_docs})

        cache_tag = "[CACHE HIT]" if is_hit else ""
        print(f"  [{i:02d}] [{role:10s}] [{category:10s}] {question[:55]:<55} "
              f"${cost_record['total_cost']:.5f}  {model.replace('gemini-',''):<15} {cache_tag}")

    # -----------------------------------------------------------------------
    # Repeat 3 popular queries to demonstrate caching
    # -----------------------------------------------------------------------
    print("\n  --- Repeating 3 queries to demonstrate cache hits ---")
    for q in queries[:3]:
        question = q["question"]
        cost_record = _simulate_query_cost(question, q["role"])
        is_hit, _ = optimizer.apply_caching(question, _simulate_response(question))
        if is_hit:
            cost_record["llm_cost"]   = 0.0
            cost_record["total_cost"] = cost_record["retrieval_cost"] * 0.1
        analyzer.record_query(cost_record)
        print(f"  [RE] {question[:55]:<55} ${cost_record['total_cost']:.5f}  {'[CACHE HIT]' if is_hit else ''}")

    # -----------------------------------------------------------------------
    # Inject a cost spike for demonstration
    # -----------------------------------------------------------------------
    analyzer.record_query({
        "query_text":     "SPIKE: bulk export all employee records",
        "retrieval_cost": 0.050,
        "llm_cost":       0.800,
        "tool_cost":      0.150,
        "error_cost":     0.100,
        "total_cost":     1.100,
    })

    # -----------------------------------------------------------------------
    # Cost breakdown
    # -----------------------------------------------------------------------
    _separator("COST BREAKDOWN")
    bd = analyzer.get_cost_breakdown()
    print(f"  Queries run:        {bd['query_count']}")
    print(f"  Retrieval cost:    ${bd['retrieval_total']:.5f}")
    print(f"  LLM cost:          ${bd['llm_total']:.5f}")
    print(f"  Tool cost:         ${bd['tool_total']:.5f}")
    print(f"  Error/retry cost:  ${bd['error_total']:.5f}")
    print(f"  -----------------------------")
    print(f"  TOTAL DAILY COST:  ${bd['total_daily']:.5f}")

    # -----------------------------------------------------------------------
    # Cost spikes
    # -----------------------------------------------------------------------
    _separator("COST SPIKE DETECTION")
    spikes = analyzer.identify_cost_spikes()
    if spikes:
        print(f"  {len(spikes)} spike(s) detected (> mean + 2 std dev):\n")
        for s in spikes:
            print(f"  • {s['query_text'][:60]}")
            print(f"    total=${s['total_cost']:.5f}  z-score={s['z_score']:.1f}  threshold=${s['spike_threshold']:.5f}")
    else:
        print("  No cost spikes detected.")

    # -----------------------------------------------------------------------
    # Optimization impact
    # -----------------------------------------------------------------------
    _separator("OPTIMIZATION IMPACT")
    impact = optimizer.get_optimization_impact()
    print(f"  Estimated total savings:   {impact['total_savings_pct']:.1f}%")
    print(f"  Strategies applied:        {', '.join(impact['strategies_applied']) or 'none'}")
    print(f"\n  Breakdown:")
    for strategy, pct in impact["breakdown"].items():
        bar = "#" * int(pct / 5)
        print(f"    {strategy:<35s}: {pct:5.1f}%  {bar}")

    # -----------------------------------------------------------------------
    # Feedback loop
    # -----------------------------------------------------------------------
    _separator("FEEDBACK LOOP EXAMPLES")

    # Accepted correction from manager
    r1 = feedback.submit_correction(
        original_query   = "What is the business class flight policy?",
        original_answer  = "There is no specific policy for business class.",
        corrected_answer = "Employees may book business class for flights over 8 hours with VP approval. This applies to international travel only.",
        user_role        = "manager",
    )
    print(f"  [manager] Correction: {'ACCEPTED' if r1['accepted'] else 'REJECTED'} — {r1['reason']}")

    # Rejected: engineer has no authority
    r2 = feedback.submit_correction(
        original_query   = "What is the per diem for NYC?",
        original_answer  = "$200/day",
        corrected_answer = "$250/day",
        user_role        = "engineer",
    )
    print(f"  [engineer] Correction: {'ACCEPTED' if r2['accepted'] else 'REJECTED'} — {r2['reason']}")

    # Accepted from HR
    r3 = feedback.submit_correction(
        original_query   = "How much parental leave do primary caregivers get?",
        original_answer  = "8 weeks paid parental leave.",
        corrected_answer = "Primary caregivers receive 16 weeks of paid parental leave. Secondary caregivers receive 8 weeks. Adoption also qualifies for 8 weeks paid leave.",
        user_role        = "hr",
    )
    print(f"  [hr]      Correction: {'ACCEPTED' if r3['accepted'] else 'REJECTED'} — {r3['reason']}")

    # Director correction
    r4 = feedback.submit_correction(
        original_query   = "Who approves expenses over $100,000?",
        original_answer  = "The VP approves all large expenses.",
        corrected_answer = "The CFO approves expenses up to unlimited. VP approves up to $100,000. Director approves up to $25,000 per the budget guidelines.",
        user_role        = "director",
    )
    print(f"  [director] Correction: {'ACCEPTED' if r4['accepted'] else 'REJECTED'} — {r4['reason']}")

    _separator("FEEDBACK METRICS")
    metrics = feedback.get_feedback_metrics()
    print(f"  Total corrections received:  {metrics['total_corrections']}")
    print(f"  Validation rate:             {metrics['validation_rate']*100:.0f}%")
    print(f"  Avg correction length:       {metrics['avg_correction_length']:.0f} chars")
    if metrics["top_error_patterns"]:
        print(f"  Top error patterns:          {', '.join(metrics['top_error_patterns'])}")

    _separator()
    print("  Evaluation complete.\n")


if __name__ == "__main__":
    run_evaluation()
