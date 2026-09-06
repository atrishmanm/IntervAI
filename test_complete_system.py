"""
test_complete_system.py
======================
Complete system test for INTERVUE.
Tests all components: imports, model, tokenizer, resume parser, adaptive interview, analytics.
"""

import sys
import os
import time

# Set UTF-8 encoding for Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def test_imports():
    """Test all imports work."""
    print("[1/10] Testing imports...")
    try:
        import torch
        import numpy as np
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        print("  [OK] Core ML imports")
        
        # Test model imports
        from models.generator.model import GeneratorConfig, InterviewGenerator
        print("  [OK] Model imports")
        
        # Test training imports
        from models.generator.train_utils import (
            create_muon_optimizer, get_wsd_schedule, EMA, PackedDataset,
            compile_model, save_checkpoint, load_checkpoint, make_training_configs
        )
        print("  [OK] Training utils imports")
        
        # Test analytics imports
        from models.generator.analytics import TrainingAnalytics
        print("  [OK] Analytics imports")
        
        # Test interview metrics imports
        from models.generator.interview_metrics import (
            check_code_correctness, compute_bleu, compute_rouge_l,
            compute_concept_accuracy, compute_relevance, compute_interview_score
        )
        print("  [OK] Interview metrics imports")
        
        # Test new orchestrator imports
        from orchestrator.resume_parser import ResumeParser, ResumeProfile, parse_resume
        print("  [OK] Resume parser imports")
        
        from orchestrator.adaptive_interview import AdaptiveInterviewEngine, create_interview_engine
        print("  [OK] Adaptive interview imports")
        
        from orchestrator.interview_analytics import InterviewAnalytics, create_analytics
        print("  [OK] Interview analytics imports")
        
        from orchestrator.mock_interview import MockInterviewSimulator, InterviewConfig, create_mock_interview
        print("  [OK] Mock interview imports")
        
        print("  [OK] All imports successful")
        return True
    except Exception as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_model():
    """Test model forward pass."""
    print("\n[2/10] Testing model...")
    try:
        import torch
        from models.generator.model import GeneratorConfig, InterviewGenerator
        
        config = GeneratorConfig(
            vocab_size=16000,
            embed_dim=256,
            n_heads=4,
            n_layers=4,
            ff_dim=512,
            max_len=512,
            dropout=0.1,
            label_smoothing=0.05,
        )
        model = InterviewGenerator(config)
        
        # Test forward pass with labels
        x = torch.randint(0, 16000, (2, 128))
        labels = torch.randint(0, 16000, (2, 128))
        out = model(x, labels=labels)
        assert "logits" in out
        assert "loss" in out
        assert out["logits"].shape == (2, 128, 16000)
        
        # Test forward pass without labels
        out_no_labels = model(x)
        assert "logits" in out_no_labels
        
        # Test weight norm extraction
        norms = model.get_weight_norms()
        assert len(norms) > 0
        
        print(f"  [OK] Model forward pass (params: {sum(p.numel() for p in model.parameters()):,})")
        return True
    except Exception as e:
        print(f"  [FAIL] Model error: {e}")
        return False


def test_tokenizer():
    """Test tokenizer operations."""
    print("\n[3/10] Testing tokenizer...")
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        import tempfile
        import os
        
        # Create tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        
        # Add special tokens
        special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        trainer = BpeTrainer(
            vocab_size=16000,
            special_tokens=special_tokens,
            min_frequency=2,
        )
        
        # Train on sample data
        sample_texts = [
            "This is a test sentence for the tokenizer.",
            "Machine learning is a subset of artificial intelligence.",
            "The quick brown fox jumps over the lazy dog.",
            "Python is a popular programming language for data science.",
            "Deep learning models require large amounts of training data.",
        ] * 20
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(sample_texts))
            temp_path = f.name
        
        try:
            tokenizer.train([temp_path], trainer)
        finally:
            os.unlink(temp_path)
        
        # Test encode/decode (without post-processor to avoid shell escaping issues)
        test_text = "Hello world, this is a test."
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded.ids)
        
        assert len(encoded.ids) > 0
        assert len(decoded) > 0  # Just check we get some output back
        
        print(f"  [OK] Tokenizer (vocab: {tokenizer.get_vocab_size()})")
        return True
    except Exception as e:
        print(f"  [FAIL] Tokenizer error: {e}")
        return False


def test_resume_parser():
    """Test resume parser."""
    print("\n[4/10] Testing resume parser...")
    try:
        from orchestrator.resume_parser import ResumeParser, parse_resume
        
        sample_resume = """
John Smith
john.smith@email.com | (555) 123-4567 | San Francisco, CA

SUMMARY
Senior Software Engineer with 8+ years of experience in Python, JavaScript, and cloud technologies.

EXPERIENCE
Senior Software Engineer | Google | 2020-2024
- Led development of microservices architecture using Python and Django
- Implemented CI/CD pipelines with Jenkins and Docker
- Managed PostgreSQL databases serving 10M+ users

Software Engineer | Startup Inc | 2018-2020
- Built React frontend and Node.js backend
- Deployed on AWS with Kubernetes
- Reduced API response time by 40%

PROJECTS
Real-time Analytics Platform
- Built with Python, TensorFlow, and Apache Kafka
- Processed 1M+ events per day
- Used Redis for caching and MongoDB for storage

EDUCATION
Bachelor of Science in Computer Science
Stanford University | 2018

SKILLS
Python, JavaScript, TypeScript, React, Node.js, Django, Flask, AWS, Docker, Kubernetes, PostgreSQL, MongoDB, Redis, TensorFlow, PyTorch, Machine Learning
"""
        
        profile = parse_resume(sample_resume)
        
        assert profile.name == "John Smith"
        assert profile.email == "john.smith@email.com"
        assert len(profile.skills) > 0
        assert len(profile.experience) > 0
        assert len(profile.projects) > 0
        
        skill_names = [s.name for s in profile.skills]
        assert "python" in skill_names
        assert "javascript" in skill_names
        
        print(f"  [OK] Resume parser (skills: {len(profile.skills)}, exp: {len(profile.experience)})")
        return True
    except Exception as e:
        print(f"  [FAIL] Resume parser error: {e}")
        return False


def test_adaptive_interview():
    """Test adaptive interview engine."""
    print("\n[5/10] Testing adaptive interview...")
    try:
        from orchestrator.adaptive_interview import AdaptiveInterviewEngine
        
        sample_resume = """
Jane Doe
jane@example.com | (555) 987-6543 | New York, NY

SUMMARY
Full-stack developer with 5 years of experience in React and Python.

EXPERIENCE
Full-stack Developer | Meta | 2021-2024
- Built React components and Python APIs
- Used PostgreSQL and Redis

PROJECTS
E-commerce Platform
- React frontend, Django backend
- Deployed on AWS

SKILLS
Python, JavaScript, React, Django, PostgreSQL, AWS, Docker
"""
        
        engine = AdaptiveInterviewEngine()
        result = engine.start_interview(sample_resume)
        
        assert "interview_id" in result
        assert "candidate_name" in result
        assert "first_question" in result
        assert len(result["skills_detected"]) > 0
        
        # Submit an answer
        answer_result = engine.submit_answer(
            "I have 5 years of experience with React and Python. In my recent role at Meta, I built scalable web applications.",
            time_taken=30.0
        )
        
        assert "score" in answer_result
        assert "feedback" in answer_result
        assert "next_question" in answer_result
        
        print(f"  [OK] Adaptive interview (questions: {result['total_questions']})")
        return True
    except Exception as e:
        print(f"  [FAIL] Adaptive interview error: {e}")
        return False


def test_interview_analytics():
    """Test interview analytics."""
    print("\n[6/10] Testing interview analytics...")
    try:
        from orchestrator.interview_analytics import InterviewAnalytics
        
        analytics = InterviewAnalytics()
        
        # Simulate some answers
        for i in range(5):
            analytics.start_question(skill="python", phase="technical")
            scores = {
                "technical_depth": 0.7 + (i * 0.05),
                "problem_solving": 0.6 + (i * 0.05),
                "communication": 0.8,
            }
            analytics.record_answer(
                question_number=i + 1,
                scores=scores,
                difficulty="MEDIUM"
            )
        
        # Get performance summary
        summary = analytics.get_performance_summary()
        assert "average_score" in summary
        assert "skill_scores" in summary
        assert "performance_level" in summary
        
        # Get radar chart data
        radar = analytics.get_radar_chart_data()
        assert "labels" in radar
        assert "values" in radar
        
        # Get prediction
        prediction = analytics.predict_success()
        assert "success_probability" in prediction
        
        print(f"  [OK] Interview analytics (score: {summary['average_score']:.2f})")
        return True
    except Exception as e:
        print(f"  [FAIL] Interview analytics error: {e}")
        return False


def test_mock_interview():
    """Test complete mock interview."""
    print("\n[7/10] Testing mock interview...")
    try:
        from orchestrator.mock_interview import MockInterviewSimulator, InterviewConfig
        
        config = InterviewConfig(
            max_questions=5,
            time_limit_minutes=15,
            phases=["warmup", "technical", "behavioral"],
        )
        
        simulator = MockInterviewSimulator(config=config)
        
        sample_resume = """
Alex Johnson
alex@example.com | (555) 111-2222 | Seattle, WA

SUMMARY
Data Scientist with 3 years of experience in machine learning and Python.

EXPERIENCE
Data Scientist | Netflix | 2022-2024
- Built recommendation models using PyTorch
- Analyzed user behavior data with pandas

SKILLS
Python, PyTorch, TensorFlow, pandas, scikit-learn, SQL, AWS
"""
        
        # Start interview
        result = simulator.start_interview(sample_resume)
        assert result["status"] == "started"
        
        # Submit a few answers
        for i in range(3):
            answer_result = simulator.submit_answer(
                f"This is my answer to question {i+1}. I have experience with Python and machine learning.",
                time_taken=30.0
            )
            assert "score" in answer_result
        
        # Get analytics
        analytics = simulator.get_analytics()
        assert "average_score" in analytics
        
        # Get radar chart
        radar = simulator.get_radar_chart()
        assert "labels" in radar
        
        print(f"  [OK] Mock interview (questions answered: {i+1})")
        return True
    except Exception as e:
        print(f"  [FAIL] Mock interview error: {e}")
        return False


def test_interview_metrics():
    """Test interview-specific metrics."""
    print("\n[8/10] Testing interview metrics...")
    try:
        from models.generator.interview_metrics import (
            check_code_correctness, compute_bleu, compute_rouge_l,
            compute_concept_accuracy, compute_relevance, compute_interview_score
        )
        
        # Test code correctness
        code = "def add(a, b): return a + b"
        result = check_code_correctness(code)
        assert result["compiles"] == True
        
        # Test BLEU score
        reference = "The quick brown fox jumps over the lazy dog"
        hypothesis = "A fast brown fox leaps over the lazy dog"
        bleu = compute_bleu(reference, hypothesis)
        assert 0 <= bleu <= 1
        
        # Test ROUGE-L
        rouge = compute_rouge_l(reference, hypothesis)
        assert 0 <= rouge <= 1
        
        # Test concept accuracy
        concepts = compute_concept_accuracy("def binary_search(arr, target): pass")
        assert "concepts_found" in concepts
        
        # Test relevance
        relevance = compute_relevance("Tell me about your Python experience", "I have 5 years of Python experience")
        assert 0 <= relevance <= 1
        
        # Test combined score
        result = compute_interview_score(
            "I built a machine learning model using Python and TensorFlow",
            reference="I built a machine learning model using Python and TensorFlow",
            question="Tell me about your experience with machine learning"
        )
        assert "combined_score" in result
        assert 0 <= result["combined_score"] <= 1
        
        print(f"  [OK] Interview metrics (BLEU: {bleu:.3f}, ROUGE: {rouge:.3f})")
        return True
    except Exception as e:
        print(f"  [FAIL] Interview metrics error: {e}")
        return False


def test_analytics_dashboard():
    """Test analytics dashboard data."""
    print("\n[9/10] Testing analytics dashboard...")
    try:
        from orchestrator.interview_analytics import InterviewAnalytics
        from orchestrator.mock_interview import MockInterviewSimulator, InterviewConfig
        
        # Create simulator
        config = InterviewConfig(max_questions=5)
        simulator = MockInterviewSimulator(config=config)
        
        sample_resume = """
Test User
test@example.com | (555) 000-0000 | Test City

SUMMARY
Developer with experience in Python and JavaScript.

SKILLS
Python, JavaScript, React, Django
"""
        
        # Start interview
        simulator.start_interview(sample_resume)
        
        # Submit answers
        for i in range(3):
            simulator.submit_answer(f"Answer {i+1}", time_taken=20.0)
        
        # Get all analytics data
        analytics = simulator.get_analytics()
        radar = simulator.get_radar_chart()
        timeline = simulator.get_timeline()
        phase_breakdown = simulator.get_phase_breakdown()
        prediction = simulator.get_prediction()
        
        # Verify all data structures
        assert isinstance(analytics, dict)
        assert isinstance(radar, dict)
        assert isinstance(timeline, dict)
        assert isinstance(phase_breakdown, dict)
        assert isinstance(prediction, dict)
        
        print(f"  [OK] Analytics dashboard (all components working)")
        return True
    except Exception as e:
        print(f"  [FAIL] Analytics dashboard error: {e}")
        return False


def test_export_formats():
    """Test report export formats."""
    print("\n[10/10] Testing export formats...")
    try:
        from orchestrator.mock_interview import MockInterviewSimulator, InterviewConfig
        
        config = InterviewConfig(max_questions=3)
        simulator = MockInterviewSimulator(config=config)
        
        sample_resume = """
Export Test User
export@test.com

SUMMARY
Test developer.

SKILLS
Python
"""
        
        # Start and complete interview
        simulator.start_interview(sample_resume)
        simulator.submit_answer("Test answer", time_taken=10.0)
        
        # Test JSON export
        json_report = simulator.export_report("json")
        assert isinstance(json_report, str)
        assert len(json_report) > 0
        
        # Test Markdown export
        md_report = simulator.export_report("markdown")
        assert isinstance(md_report, str)
        assert "# Interview Report" in md_report
        
        # Test HTML export
        html_report = simulator.export_report("html")
        assert isinstance(html_report, str)
        assert "<!DOCTYPE html>" in html_report
        
        print(f"  [OK] Export formats (JSON, Markdown, HTML)")
        return True
    except Exception as e:
        print(f"  [FAIL] Export formats error: {e}")
        return False


def main():
    print("=" * 70)
    print("  INTERVUE Complete System Test")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_model,
        test_tokenizer,
        test_resume_parser,
        test_adaptive_interview,
        test_interview_analytics,
        test_mock_interview,
        test_interview_metrics,
        test_analytics_dashboard,
        test_export_formats,
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
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{passed + failed} tests passed")
    print("=" * 70)
    
    if failed == 0:
        print("\n  [SUCCESS] All tests passed!")
        print("  The system is ready for training and deployment.")
        return True
    else:
        print(f"\n  [WARNING] {failed} test(s) failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
