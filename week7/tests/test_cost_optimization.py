"""
Week 7 Tests: CostAnalyzer, OptimizationStrategy, FeedbackLoop
"""

import os
import sys
import pytest

WEEK7_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WEEK7_DIR)

from app.cost_optimization import CostAnalyzer, OptimizationStrategy, FeedbackLoop


# ===========================================================================
# Helpers
# ===========================================================================

def make_query(text="test query", retrieval=0.001, llm=0.005, tool=0.001, error=0.0):
    total = retrieval + llm + tool + error
    return {
        "query_text":     text,
        "retrieval_cost": retrieval,
        "llm_cost":       llm,
        "tool_cost":      tool,
        "error_cost":     error,
        "total_cost":     total,
    }


# ===========================================================================
# CostAnalyzer
# ===========================================================================

class TestCostAnalyzerInit:
    def test_starts_with_empty_history(self):
        ca = CostAnalyzer()
        assert ca.query_history == []


class TestRecordQuery:
    def test_record_stores_query(self):
        ca = CostAnalyzer()
        ca.record_query(make_query("What is the travel policy?"))
        assert len(ca.query_history) == 1

    def test_record_multiple_queries(self):
        ca = CostAnalyzer()
        for i in range(5):
            ca.record_query(make_query(f"query {i}"))
        assert len(ca.query_history) == 5

    def test_record_adds_timestamp(self):
        ca = CostAnalyzer()
        ca.record_query(make_query())
        assert "timestamp" in ca.query_history[0]

    def test_record_stores_cost_components(self):
        ca = CostAnalyzer()
        ca.record_query(make_query("q", retrieval=0.002, llm=0.010, tool=0.001, error=0.001))
        q = ca.query_history[0]
        assert q["retrieval_cost"] == pytest.approx(0.002)
        assert q["llm_cost"]       == pytest.approx(0.010)
        assert q["tool_cost"]      == pytest.approx(0.001)
        assert q["error_cost"]     == pytest.approx(0.001)

    def test_record_computes_total_if_missing(self):
        ca = CostAnalyzer()
        ca.record_query({
            "query_text": "q", "retrieval_cost": 0.002,
            "llm_cost": 0.008, "tool_cost": 0.0, "error_cost": 0.0,
        })
        assert ca.query_history[0]["total_cost"] == pytest.approx(0.010)


class TestGetCostBreakdown:
    def test_empty_history_returns_zeros(self):
        ca = CostAnalyzer()
        bd = ca.get_cost_breakdown()
        assert bd["total_daily"] == 0.0
        assert bd["query_count"] == 0

    def test_sums_across_queries(self):
        ca = CostAnalyzer()
        ca.record_query(make_query(retrieval=0.002, llm=0.010, tool=0.001, error=0.001))
        ca.record_query(make_query(retrieval=0.003, llm=0.015, tool=0.002, error=0.000))
        bd = ca.get_cost_breakdown()
        assert bd["retrieval_total"] == pytest.approx(0.005)
        assert bd["llm_total"]       == pytest.approx(0.025)
        assert bd["tool_total"]      == pytest.approx(0.003)
        assert bd["error_total"]     == pytest.approx(0.001)
        assert bd["total_daily"]     == pytest.approx(0.034)
        assert bd["query_count"]     == 2

    def test_breakdown_has_all_keys(self):
        ca = CostAnalyzer()
        bd = ca.get_cost_breakdown()
        for key in ("retrieval_total", "llm_total", "tool_total", "error_total", "total_daily", "query_count"):
            assert key in bd


class TestIdentifyCostSpikes:
    def test_no_spikes_when_uniform(self):
        ca = CostAnalyzer()
        for _ in range(10):
            ca.record_query(make_query(llm=0.005))
        assert ca.identify_cost_spikes() == []

    def test_detects_spike(self):
        ca = CostAnalyzer()
        # 9 cheap queries, 1 very expensive outlier
        for _ in range(9):
            ca.record_query(make_query("cheap", retrieval=0.001, llm=0.002, tool=0.0, error=0.0))
        ca.record_query(make_query("spike", retrieval=0.050, llm=0.200, tool=0.010, error=0.020))
        spikes = ca.identify_cost_spikes()
        assert len(spikes) == 1
        assert spikes[0]["query_text"] == "spike"

    def test_returns_empty_when_fewer_than_two_records(self):
        ca = CostAnalyzer()
        ca.record_query(make_query())
        assert ca.identify_cost_spikes() == []

    def test_spike_includes_z_score(self):
        ca = CostAnalyzer()
        for _ in range(9):
            ca.record_query(make_query(llm=0.001))
        ca.record_query(make_query("expensive", llm=1.0))
        spikes = ca.identify_cost_spikes()
        assert "z_score" in spikes[0]
        assert spikes[0]["z_score"] > 2

    def test_spike_sorted_most_expensive_first(self):
        ca = CostAnalyzer()
        for _ in range(8):
            ca.record_query(make_query(llm=0.001))
        ca.record_query(make_query("medium spike", llm=0.5))
        ca.record_query(make_query("large spike",  llm=1.0))
        spikes = ca.identify_cost_spikes()
        assert spikes[0]["total_cost"] >= spikes[-1]["total_cost"]


# ===========================================================================
# OptimizationStrategy
# ===========================================================================

class TestApplyCaching:
    def test_miss_returns_false(self):
        opt = OptimizationStrategy()
        hit, resp = opt.apply_caching("hello", "world")
        assert hit is False
        assert resp == "world"

    def test_second_call_is_cache_hit(self):
        opt = OptimizationStrategy()
        opt.apply_caching("travel policy?", "10 days PTO")
        hit, resp = opt.apply_caching("travel policy?", "irrelevant")
        assert hit is True
        assert resp == "10 days PTO"

    def test_cache_is_case_insensitive(self):
        opt = OptimizationStrategy()
        opt.apply_caching("What is the travel policy?", "answer")
        hit, resp = opt.apply_caching("what is the travel policy?", "different")
        assert hit is True

    def test_different_queries_independent(self):
        opt = OptimizationStrategy()
        opt.apply_caching("query A", "answer A")
        hit, _ = opt.apply_caching("query B", "answer B")
        assert hit is False


class TestOptimizeRetrievalCount:
    def test_reduces_by_five(self):
        opt = OptimizationStrategy()
        assert opt.optimize_retrieval_count(15) == 3

    def test_never_returns_zero(self):
        opt = OptimizationStrategy()
        assert opt.optimize_retrieval_count(1) >= 1

    def test_reduces_large_count(self):
        opt = OptimizationStrategy()
        result = opt.optimize_retrieval_count(50)
        assert result < 50

    def test_records_strategy(self):
        opt = OptimizationStrategy()
        opt.optimize_retrieval_count(10)
        assert "retrieval_optimization" in opt.strategies_applied


class TestSelectModelByComplexity:
    def test_simple_query_gets_flash(self):
        opt = OptimizationStrategy()
        model = opt.select_model_by_complexity("What is the travel policy?")
        assert model == "gemini-1.5-flash"

    def test_complex_query_gets_pro(self):
        opt = OptimizationStrategy()
        model = opt.select_model_by_complexity(
            "Analyze and compare the cost tradeoffs between all cloud providers "
            "and design a hybrid architecture with optimal cost-efficiency."
        )
        assert model == "gemini-2.5-pro"

    def test_returns_valid_model_name(self):
        opt = OptimizationStrategy()
        model = opt.select_model_by_complexity("How much is the per diem?")
        assert model in ("gemini-1.5-flash", "gemini-2.5-pro")

    def test_flash_selection_records_strategy(self):
        opt = OptimizationStrategy()
        opt.select_model_by_complexity("What is PTO?")
        assert "model_selection" in opt.strategies_applied


class TestEnableResponseCompression:
    def test_short_response_unchanged(self):
        opt = OptimizationStrategy()
        short = "Policy is 15 days. Contact HR for details."
        assert opt.enable_response_compression(short) == short

    def test_long_response_gets_truncated(self):
        opt = OptimizationStrategy()
        long_resp = ". ".join([f"Sentence {i}" for i in range(20)]) + "."
        compressed = opt.enable_response_compression(long_resp)
        assert len(compressed) < len(long_resp)

    def test_compressed_starts_same_as_original(self):
        opt = OptimizationStrategy()
        sentences = [f"Point {i}." for i in range(10)]
        original = " ".join(sentences)
        compressed = opt.enable_response_compression(original)
        assert compressed.startswith("Point 0.")

    def test_empty_response_unchanged(self):
        opt = OptimizationStrategy()
        assert opt.enable_response_compression("") == ""

    def test_compression_records_strategy(self):
        opt = OptimizationStrategy()
        long_resp = " ".join([f"Sentence {i}." for i in range(10)])
        opt.enable_response_compression(long_resp)
        assert "response_compression" in opt.strategies_applied


class TestGetOptimizationImpact:
    def test_returns_required_keys(self):
        opt = OptimizationStrategy()
        impact = opt.get_optimization_impact()
        assert "total_savings_pct" in impact
        assert "strategies_applied" in impact
        assert "breakdown" in impact

    def test_savings_increases_with_cache_hits(self):
        opt = OptimizationStrategy()
        opt.apply_caching("q", "a")
        opt.apply_caching("q", "a")  # hit
        opt.apply_caching("q", "a")  # hit
        impact = opt.get_optimization_impact()
        assert impact["total_savings_pct"] > 0

    def test_savings_never_exceed_95_pct(self):
        opt = OptimizationStrategy()
        for _ in range(100):
            opt.apply_caching("same query", "same answer")
        impact = opt.get_optimization_impact()
        assert impact["total_savings_pct"] <= 95.0

    def test_no_strategies_returns_zero_savings(self):
        opt = OptimizationStrategy()
        impact = opt.get_optimization_impact()
        assert impact["total_savings_pct"] == 0.0


# ===========================================================================
# FeedbackLoop
# ===========================================================================

class TestSubmitCorrection:
    def test_manager_can_submit_correction(self):
        fl = FeedbackLoop()
        result = fl.submit_correction("q", "wrong answer", "correct answer here", "manager")
        assert result["accepted"] is True

    def test_engineer_cannot_submit_correction(self):
        fl = FeedbackLoop()
        result = fl.submit_correction("q", "wrong", "correct", "engineer")
        assert result["accepted"] is False

    def test_intern_cannot_submit_correction(self):
        fl = FeedbackLoop()
        result = fl.submit_correction("q", "wrong", "correct", "intern")
        assert result["accepted"] is False

    def test_executive_can_submit_correction(self):
        fl = FeedbackLoop()
        result = fl.submit_correction("q", "wrong answer text", "this is the correct answer", "executive")
        assert result["accepted"] is True

    def test_identical_answer_rejected(self):
        fl = FeedbackLoop()
        result = fl.submit_correction("q", "same answer", "same answer", "manager")
        assert result["accepted"] is False

    def test_empty_correction_rejected(self):
        fl = FeedbackLoop()
        result = fl.submit_correction("q", "original", "", "manager")
        assert result["accepted"] is False

    def test_accepted_correction_stored(self):
        fl = FeedbackLoop()
        fl.submit_correction("q", "wrong", "this is the correct and detailed answer", "manager")
        assert len(fl.corrections) == 1

    def test_rejected_correction_not_stored(self):
        fl = FeedbackLoop()
        fl.submit_correction("q", "wrong", "correct", "engineer")
        assert len(fl.corrections) == 0

    def test_result_has_reason_field(self):
        fl = FeedbackLoop()
        result = fl.submit_correction("q", "wrong", "correct", "engineer")
        assert "reason" in result

    def test_director_can_submit(self):
        fl = FeedbackLoop()
        result = fl.submit_correction("q", "wrong answer here", "the actual correct answer here", "director")
        assert result["accepted"] is True


class TestValidateCorrection:
    def test_valid_correction_returns_true(self):
        fl = FeedbackLoop()
        fl.submit_correction("q", "short answer", "this is a much longer and more detailed correct answer", "manager")
        assert fl.validate_correction(0) is True

    def test_invalid_index_returns_false(self):
        fl = FeedbackLoop()
        assert fl.validate_correction(0) is False
        assert fl.validate_correction(-1) is False

    def test_short_correction_returns_false(self):
        fl = FeedbackLoop()
        # Force insert a short correction to bypass submit validation
        fl.corrections.append({
            "original_query": "q",
            "original_answer": "a longer original answer that has more text",
            "corrected_answer": "no",
            "user_role": "engineer",  # low authority
            "role_level": 1,
            "valid": True,
        })
        assert fl.validate_correction(0) is False


class TestGetFeedbackMetrics:
    def test_empty_returns_zeros(self):
        fl = FeedbackLoop()
        m = fl.get_feedback_metrics()
        assert m["total_corrections"] == 0
        assert m["validation_rate"] == 0.0

    def test_counts_total_corrections(self):
        fl = FeedbackLoop()
        fl.submit_correction("q1", "bad answer", "good detailed answer one here", "manager")
        fl.submit_correction("q2", "bad answer 2", "good detailed answer two here", "manager")
        m = fl.get_feedback_metrics()
        assert m["total_corrections"] == 2

    def test_validation_rate_between_zero_and_one(self):
        fl = FeedbackLoop()
        fl.submit_correction("q", "wrong", "this is the correct detailed answer", "manager")
        m = fl.get_feedback_metrics()
        assert 0.0 <= m["validation_rate"] <= 1.0

    def test_has_all_required_keys(self):
        fl = FeedbackLoop()
        m = fl.get_feedback_metrics()
        for key in ("total_corrections", "validation_rate", "avg_correction_length", "top_error_patterns"):
            assert key in m

    def test_avg_length_is_positive_when_corrections_exist(self):
        fl = FeedbackLoop()
        fl.submit_correction("q", "wrong answer", "this is a much longer correct answer", "manager")
        m = fl.get_feedback_metrics()
        assert m["avg_correction_length"] > 0
