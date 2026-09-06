"""
test_sota_features.py
=====================
Test all new SOTA features: company templates, session tracking, industry modules, salary negotiation.
"""

import sys
import os
import time
import json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def test_company_templates():
    """Test company-specific templates."""
    print("\n[1/4] Testing company templates...")
    try:
        from orchestrator.company_templates import (
            get_company_template, list_companies, get_questions_for_company,
            get_evaluation_criteria, CompanyTier
        )
        
        # Test listing companies
        all_companies = list_companies()
        assert len(all_companies) >= 5, f"Expected >=5 companies, got {len(all_companies)}"
        
        # Test FAANG tier
        faang_companies = list_companies(CompanyTier.FAANG)
        assert len(faang_companies) >= 4, f"Expected >=4 FAANG companies"
        
        # Test specific company
        google = get_company_template("google")
        assert google is not None
        assert google.company_name == "Google"
        assert len(google.rounds) >= 3
        assert len(google.common_questions) >= 3
        assert google.bar_level == "very_high"
        
        # Test Amazon
        amazon = get_company_template("amazon")
        assert amazon is not None
        assert "customer obsession" in amazon.culture_keywords
        
        # Test questions
        meta_questions = get_questions_for_company("meta")
        assert len(meta_questions) >= 3
        
        # Test evaluation criteria
        google_criteria = get_evaluation_criteria("google")
        assert "technical_depth" in google_criteria
        assert sum(google_criteria.values()) == 1.0
        
        print(f"  [OK] Company templates ({len(all_companies)} companies, {len(faang_companies)} FAANG)")
        return True
    except Exception as e:
        print(f"  [FAIL] Company templates error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_session_tracker():
    """Test multi-session tracking."""
    print("\n[2/4] Testing session tracker...")
    try:
        from orchestrator.session_tracker import SessionTracker, create_session_tracker
        import tempfile
        import shutil
        
        # Create temp directory for testing
        temp_dir = tempfile.mkdtemp()
        
        try:
            tracker = create_session_tracker(temp_dir)
            
            # Record some sessions with clear improvement
            for i in range(5):
                session_data = {
                    "duration_minutes": 30 + i * 5,
                    "overall_score": 0.4 + i * 0.1,  # Clear improvement: 0.4, 0.5, 0.6, 0.7, 0.8
                    "recommendation": "HIRE" if i >= 3 else "MAYBE",
                    "dimension_scores": {
                        "technical_depth": 0.4 + i * 0.1,
                        "communication": 0.5 + i * 0.08,
                        "problem_solving": 0.4 + i * 0.1,
                    },
                    "phase_scores": {
                        "technical": 0.4 + i * 0.1,
                        "behavioral": 0.5 + i * 0.08,
                    },
                    "skills_assessed": ["python", "system_design"],
                    "strengths": ["technical_depth"],
                    "weaknesses": ["communication"] if i < 3 else [],
                    "company_template": "google" if i % 2 == 0 else "meta",
                }
                session_id = tracker.record_session(session_data)
                assert session_id is not None
            
            # Test session history
            history = tracker.get_session_history()
            assert len(history) == 5
            
            # Test improvement trends
            trends = tracker.get_improvement_trends()
            assert trends["total_sessions"] == 5
            assert trends["overall_trend"] == "improving"
            assert trends["improvement"] > 0
            
            # Test skill progress
            skill_progress = tracker.get_skill_progress()
            assert "python" in skill_progress
            assert skill_progress["python"]["trend"] == "improving"
            
            # Test weakness analysis
            weaknesses = tracker.get_weakness_analysis()
            assert len(weaknesses) >= 1
            
            # Test goals
            goal_id = tracker.set_goal("Reach 0.8 score", 0.8, "2026-12-31")
            goals = tracker.get_goals()
            assert len(goals) == 1
            
            # Test overall stats
            stats = tracker.get_overall_stats()
            assert stats["total_sessions"] == 5
            assert stats["average_score"] > 0
            
            # Test persistence
            tracker2 = create_session_tracker(temp_dir)
            history2 = tracker2.get_session_history()
            assert len(history2) == 5
            
            print(f"  [OK] Session tracker (5 sessions, improving trend)")
            return True
        finally:
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"  [FAIL] Session tracker error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_industry_modules():
    """Test industry-specific modules."""
    print("\n[3/4] Testing industry modules...")
    try:
        from orchestrator.industry_modules import (
            get_industry_module, list_industries, get_questions_for_industry
        )
        
        # Test listing industries
        industries = list_industries()
        assert len(industries) >= 4, f"Expected >=4 industries, got {len(industries)}"
        
        # Test Backend module
        backend = get_industry_module("backend")
        assert backend is not None
        assert backend.name == "Backend Engineering"
        assert len(backend.core_skills) >= 5
        assert len(backend.system_design_topics) >= 5
        assert len(backend.coding_challenges) >= 5
        
        # Test Frontend module
        frontend = get_industry_module("frontend")
        assert frontend is not None
        assert frontend.name == "Frontend Engineering"
        assert len(frontend.core_skills) >= 5
        
        # Test Data Science module
        ds = get_industry_module("data_science")
        assert ds is not None
        assert ds.name == "Data Science"
        
        # Test DevOps module
        devops = get_industry_module("devops")
        assert devops is not None
        assert devops.name == "DevOps/SRE"
        
        # Test questions
        backend_design = get_questions_for_industry("backend", "system_design")
        assert len(backend_design) >= 5
        
        backend_coding = get_questions_for_industry("backend", "coding")
        assert len(backend_coding) >= 5
        
        # Test evaluation criteria
        assert sum(backend.evaluation_criteria.values()) == 1.0
        
        print(f"  [OK] Industry modules ({len(industries)} industries)")
        return True
    except Exception as e:
        print(f"  [FAIL] Industry modules error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_salary_negotiation():
    """Test salary negotiation practice."""
    print("\n[4/4] Testing salary negotiation...")
    try:
        from orchestrator.salary_negotiation import (
            SalaryNegotiationPractice, create_negotiation_practice
        )
        
        practice = create_negotiation_practice()
        
        # Test listing scenarios
        scenarios = practice.list_scenarios()
        assert len(scenarios) >= 4, f"Expected >=4 scenarios, got {len(scenarios)}"
        
        # Test starting negotiation
        result = practice.start_negotiation()
        assert "scenario_id" in result
        assert "initial_offer" in result
        assert "market_data" in result
        assert result["initial_offer"]["base_salary"] > 0
        
        # Test submitting response
        response = practice.submit_response(
            "Thank you for the offer. Based on market data from Glassdoor and Levels.fyi, "
            "the average base salary for this role is around $195,000. I'd like to discuss "
            "adjusting the compensation to be more in line with market rates. I'm confident "
            "we can find a mutually beneficial arrangement.",
            tactic="collaborative"
        )
        assert "feedback" in response
        assert "employer_response" in response
        assert response["feedback"]["overall"] > 0
        
        # Test multiple rounds
        for i in range(2):
            response = practice.submit_response(
                "I understand your constraints. Would it be possible to increase the signing bonus "
                "and equity to make the total compensation more competitive?",
                tactic="collaborative"
            )
        
        # Test summary
        summary = practice.get_negotiation_summary()
        assert summary["total_rounds"] >= 2
        assert "average_scores" in summary
        assert "recommendation" in summary
        
        # Test specific scenario
        result2 = practice.start_negotiation("faang_senior_1")
        assert result2["company_type"] == "faang"
        assert result2["role_level"] == "senior"
        
        print(f"  [OK] Salary negotiation ({len(scenarios)} scenarios, negotiation summary)")
        return True
    except Exception as e:
        print(f"  [FAIL] Salary negotiation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("  INTERVUE SOTA Features Test")
    print("=" * 70)
    
    tests = [
        test_company_templates,
        test_session_tracker,
        test_industry_modules,
        test_salary_negotiation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{passed + failed} tests passed")
    print("=" * 70)
    
    if failed == 0:
        print("\n  [SUCCESS] All SOTA feature tests passed!")
        return True
    else:
        print(f"\n  [WARNING] {failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
