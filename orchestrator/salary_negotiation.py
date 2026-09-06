"""
orchestrator/salary_negotiation.py
===================================
Salary negotiation practice module.

Features:
1. Negotiation scenario generation
2. Response evaluation (assertiveness, professionalism, data-driven)
3. Common objection handling
4. Market data awareness
5. Total compensation negotiation (base, equity, bonus, benefits)
6. Multi-round negotiation practice
7. Feedback on negotiation tactics
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class NegotiationPhase(Enum):
    INITIAL_OFFER = "initial_offer"
    COUNTER_OFFER = "counter_offer"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    MULTI_ROUND = "multi_round"


class NegotiationStyle(Enum):
    COLLABORATIVE = "collaborative"  # Win-win approach
    COMPETITIVE = "competitive"  # Maximize own outcome
    PASSIVE = "passive"  # Accept what's given
    AGGRESSIVE = "aggressive"  # Demand more


@dataclass
class CompensationBreakdown:
    """Detailed compensation breakdown."""
    base_salary: float = 0
    signing_bonus: float = 0
    annual_bonus_pct: float = 0
    equity_annual: float = 0
    equity_total_4yr: float = 0
    benefits_value: float = 0
    other_perks: float = 0
    
    @property
    def total_first_year(self) -> float:
        return (
            self.base_salary
            + self.signing_bonus
            + (self.base_salary * self.annual_bonus_pct)
            + self.equity_annual
            + self.benefits_value
        )
    
    @property
    def total_4yr(self) -> float:
        return (
            self.base_salary * 4
            + self.signing_bonus
            + (self.base_salary * self.annual_bonus_pct * 4)
            + self.equity_total_4yr
            + self.benefits_value * 4
        )


@dataclass
class NegotiationScenario:
    """A negotiation practice scenario."""
    scenario_id: str
    company_type: str  # "faang", "startup", "enterprise"
    role_level: str  # "junior", "mid", "senior", "staff", "director"
    initial_offer: CompensationBreakdown
    market_data: CompensationBreakdown
    company_constraints: List[str]
    candidate_leverage: List[str]
    obje_ctions: List[str]
    tips: List[str]
    expected_range: Dict[str, float]  # min, target, max


@dataclass
class NegotiationFeedback:
    """Feedback on a negotiation response."""
    assertiveness_score: float  # 0-1
    professionalism_score: float  # 0-1
    data_driven_score: float  # 0-1
    collaboration_score: float  # 0-1
    overall_score: float  # 0-1
    strengths: List[str]
    improvements: List[str]
    suggested_response: str
    tactic_used: str
    next_steps: List[str]


class SalaryNegotiationPractice:
    """Salary negotiation practice module."""

    def __init__(self):
        self.scenarios = self._generate_scenarios()
        self.current_scenario: Optional[NegotiationScenario] = None
        self.negotiation_history: List[Dict] = []
        self.current_phase = NegotiationPhase.INITIAL_OFFER
        self.round_number = 0

    def _generate_scenarios(self) -> List[NegotiationScenario]:
        """Generate realistic negotiation scenarios."""
        scenarios = [
            # FAANG Senior Engineer
            NegotiationScenario(
                scenario_id="faang_senior_1",
                company_type="faang",
                role_level="senior",
                initial_offer=CompensationBreakdown(
                    base_salary=180000,
                    signing_bonus=25000,
                    annual_bonus_pct=0.15,
                    equity_annual=50000,
                    equity_total_4yr=200000,
                    benefits_value=15000,
                ),
                market_data=CompensationBreakdown(
                    base_salary=195000,
                    signing_bonus=30000,
                    annual_bonus_pct=0.15,
                    equity_annual=65000,
                    equity_total_4yr=260000,
                    benefits_value=15000,
                ),
                company_constraints=[
                    "We have strict leveling bands",
                    "Equity refresh is performance-based",
                    "Signing bonus is discretionary",
                ],
                candidate_leverage=[
                    "Competing offer from another FAANG",
                    "5+ years of relevant experience",
                    "Specialized skills in high demand",
                ],
                obje_ctions=[
                    "We're at the top of the band for this level",
                    "We don't negotiate on equity",
                    "Our benefits package is already competitive",
                ],
                tips=[
                    "Focus on total compensation, not just base",
                    "Use market data to justify your ask",
                    "Be professional but assertive",
                    "Consider non-monetary benefits",
                ],
                expected_range={"min": 250000, "target": 320000, "max": 380000},
            ),
            # Startup Senior Engineer
            NegotiationScenario(
                scenario_id="startup_senior_1",
                company_type="startup",
                role_level="senior",
                initial_offer=CompensationBreakdown(
                    base_salary=150000,
                    signing_bonus=10000,
                    annual_bonus_pct=0.10,
                    equity_annual=30000,
                    equity_total_4yr=120000,
                    benefits_value=10000,
                ),
                market_data=CompensationBreakdown(
                    base_salary=165000,
                    signing_bonus=15000,
                    annual_bonus_pct=0.10,
                    equity_annual=40000,
                    equity_total_4yr=160000,
                    benefits_value=10000,
                ),
                company_constraints=[
                    "We're pre-Series B, budget is tight",
                    "Equity is our biggest leverage",
                    "We can be flexible on title and responsibilities",
                ],
                candidate_leverage=[
                    "Bringing critical domain expertise",
                    "Can help raise next funding round",
                    "Multiple competing offers",
                ],
                obje_ctions=[
                    "We're a startup, we can't match FAANG salaries",
                    "Equity could be worth much more at IPO",
                    "We value work-life balance and impact",
                ],
                tips=[
                    "Negotiate for more equity at early stage",
                    "Ask for acceleration on change of control",
                    "Consider negotiating title and responsibilities",
                    "Ask about the equity refresh policy",
                ],
                expected_range={"min": 200000, "target": 250000, "max": 300000},
            ),
            # Enterprise Mid-Level
            NegotiationScenario(
                scenario_id="enterprise_mid_1",
                company_type="enterprise",
                role_level="mid",
                initial_offer=CompensationBreakdown(
                    base_salary=120000,
                    signing_bonus=5000,
                    annual_bonus_pct=0.10,
                    equity_annual=0,
                    equity_total_4yr=0,
                    benefits_value=20000,
                ),
                market_data=CompensationBreakdown(
                    base_salary=130000,
                    signing_bonus=8000,
                    annual_bonus_pct=0.12,
                    equity_annual=5000,
                    equity_total_4yr=20000,
                    benefits_value=20000,
                ),
                company_constraints=[
                    "We have standardized salary bands",
                    "Bonuses are company-wide, not negotiable",
                    "We offer excellent benefits and stability",
                ],
                candidate_leverage=[
                    "Specialized certification",
                    "Above-average experience for the level",
                    "Recession-proof industry experience",
                ],
                obje_ctions=[
                    "This is our standard offer for this level",
                    "We don't typically negotiate on salary",
                    "Our benefits package compensates for lower base",
                ],
                tips=[
                    "Focus on the total compensation package",
                    "Ask about performance review timelines",
                    "Negotiate signing bonus as a compromise",
                    "Ask about professional development budget",
                ],
                expected_range={"min": 145000, "target": 165000, "max": 185000},
            ),
            # FAANG Staff Engineer
            NegotiationScenario(
                scenario_id="faang_staff_1",
                company_type="faang",
                role_level="staff",
                initial_offer=CompensationBreakdown(
                    base_salary=220000,
                    signing_bonus=50000,
                    annual_bonus_pct=0.20,
                    equity_annual=100000,
                    equity_total_4yr=400000,
                    benefits_value=20000,
                ),
                market_data=CompensationBreakdown(
                    base_salary=240000,
                    signing_bonus=60000,
                    annual_bonus_pct=0.20,
                    equity_annual=130000,
                    equity_total_4yr=520000,
                    benefits_value=20000,
                ),
                company_constraints=[
                    "Staff level has a wide band",
                    "Equery grants are discretionary",
                    "Signing bonus is negotiable",
                ],
                candidate_leverage=[
                    "Extremely rare skill set",
                    "Can lead critical initiatives",
                    "Multiple competing offers",
                ],
                obje_ctions=[
                    "We need to see your impact first",
                    "Equity refresh is based on performance",
                    "Our leveling process is rigorous",
                ],
                tips=[
                    "Negotiate based on your unique value",
                    "Ask for guaranteed equity refresh",
                    "Negotiate for specific project ownership",
                    "Consider negotiating remote work flexibility",
                ],
                expected_range={"min": 400000, "target": 500000, "max": 600000},
            ),
            # Startup Early Stage
            NegotiationScenario(
                scenario_id="startup_early_1",
                company_type="startup",
                role_level="senior",
                initial_offer=CompensationBreakdown(
                    base_salary=130000,
                    signing_bonus=5000,
                    annual_bonus_pct=0.05,
                    equity_annual=20000,
                    equity_total_4yr=80000,
                    benefits_value=8000,
                ),
                market_data=CompensationBreakdown(
                    base_salary=150000,
                    signing_bonus=10000,
                    annual_bonus_pct=0.10,
                    equity_annual=30000,
                    equity_total_4yr=120000,
                    benefits_value=8000,
                ),
                company_constraints=[
                    "Pre-seed, very limited budget",
                    "Equity is our main leverage",
                    "We're offering significant ownership",
                ],
                candidate_leverage=[
                    "Can build the entire technical foundation",
                    "First engineering hire",
                    "Equity could be worth millions at exit",
                ],
                obje_ctions=[
                    "We're bootstrapped, every dollar counts",
                    "Equity at this stage is extremely valuable",
                    "We're offering 0.5-1% ownership",
                ],
                tips=[
                    "Negotiate for larger equity stake",
                    "Ask for board seat or observer rights",
                    "Negotiate for change of control acceleration",
                    "Ask about the cap table and investor terms",
                ],
                expected_range={"min": 180000, "target": 220000, "max": 280000},
            ),
        ]
        return scenarios

    def start_negotiation(self, scenario_id: str = None) -> Dict:
        """Start a negotiation practice session."""
        if scenario_id:
            self.current_scenario = next(
                (s for s in self.scenarios if s.scenario_id == scenario_id or scenario_id.lower() in s.scenario_id.lower() or s.scenario_id.lower() in scenario_id.lower()),
                None
            )
        if not self.current_scenario:
            import random
            self.current_scenario = self.scenarios[0] if self.scenarios else None
        
        if not self.current_scenario:
            return {"error": "Scenario not found"}
        
        self.current_phase = NegotiationPhase.INITIAL_OFFER
        self.round_number = 1
        self.negotiation_history = []
        
        return {
            "scenario_id": self.current_scenario.scenario_id,
            "company_type": self.current_scenario.company_type,
            "role_level": self.current_scenario.role_level,
            "initial_offer": {
                "base_salary": self.current_scenario.initial_offer.base_salary,
                "signing_bonus": self.current_scenario.initial_offer.signing_bonus,
                "annual_bonus_pct": self.current_scenario.initial_offer.annual_bonus_pct,
                "equity_annual": self.current_scenario.initial_offer.equity_annual,
                "total_first_year": self.current_scenario.initial_offer.total_first_year,
                "total_4yr": self.current_scenario.initial_offer.total_4yr,
            },
            "market_data": {
                "base_salary": self.current_scenario.market_data.base_salary,
                "total_first_year": self.current_scenario.market_data.total_first_year,
            },
            "phase": self.current_phase.value,
            "round": self.round_number,
            "instructions": (
                "You've received a job offer! Practice negotiating the compensation package. "
                "Consider the company type, market data, and your leverage. "
                "Be professional but assertive."
            ),
        }

    def submit_response(self, response: str, tactic: str = "collaborative") -> Dict:
        """Submit a negotiation response and get feedback."""
        if not self.current_scenario:
            return {"error": "No active negotiation"}
        
        # Evaluate the response
        feedback = self._evaluate_response(response, tactic)
        
        # Record in history
        self.negotiation_history.append({
            "round": self.round_number,
            "phase": self.current_phase.value,
            "response": response,
            "tactic": tactic,
            "feedback": feedback,
        })
        
        # Determine if employer accepts, counters, or objects
        employer_response = self._generate_employer_response(feedback, tactic)
        
        # Move to next phase
        self._advance_phase()
        
        return {
            "round": self.round_number,
            "phase": self.current_phase.value,
            "employer_response": employer_response,
            "feedback": {
                "assertiveness": feedback.assertiveness_score,
                "professionalism": feedback.professionalism_score,
                "data_driven": feedback.data_driven_score,
                "collaboration": feedback.collaboration_score,
                "overall": feedback.overall_score,
                "strengths": feedback.strengths,
                "improvements": feedback.improvements,
                "suggested_response": feedback.suggested_response,
                "tactic_used": feedback.tactic_used,
                "next_steps": feedback.next_steps,
            },
            "rounds_remaining": max(0, 3 - self.round_number),
        }

    def _evaluate_response(self, response: str, tactic: str) -> NegotiationFeedback:
        """Evaluate a negotiation response."""
        response_lower = response.lower()
        
        # Assertiveness score
        assertive_indicators = [
            "i would like", "i'm requesting", "i'd like to discuss",
            "based on my research", "market data shows", "i believe",
            "i'm confident", "i deserve", "i expect", "my expectation",
        ]
        assertiveness = min(1.0, sum(1 for ind in assertive_indicators if ind in response_lower) / 3)
        
        # Professionalism score
        professional_indicators = [
            "thank you", "i appreciate", "i'm excited", "i'm grateful",
            "would it be possible", "i understand", "i respect",
            "collaborative", "partnership", "mutual benefit",
        ]
        professionalism = min(1.0, sum(1 for ind in professional_indicators if ind in response_lower) / 3)
        
        # Data-driven score
        data_indicators = [
            "market rate", "glassdoor", "levels.fyi", "compensation data",
            "industry average", "comparable roles", "based on research",
            "market data", "compensation benchmark", "percentile",
        ]
        data_driven = min(1.0, sum(1 for ind in data_indicators if ind in response_lower) / 2)
        
        # Collaboration score
        collaborative_indicators = [
            "how can we", "let's find", "work together", "mutual benefit",
            "flexible", "open to", "willing to", "compromise",
            "find a solution", "creative solution",
        ]
        collaboration = min(1.0, sum(1 for ind in collaborative_indicators if ind in response_lower) / 2)
        
        # Overall score
        overall = (assertiveness * 0.3 + professionalism * 0.2 + 
                   data_driven * 0.3 + collaboration * 0.2)
        
        # Strengths and improvements
        strengths = []
        improvements = []
        
        if assertiveness > 0.7:
            strengths.append("Strong assertiveness in your ask")
        elif assertiveness < 0.3:
            improvements.append("Be more assertive in stating your expectations")
        
        if professionalism > 0.7:
            strengths.append("Professional and respectful tone")
        elif professionalism < 0.3:
            improvements.append("Maintain a more professional tone")
        
        if data_driven > 0.7:
            strengths.append("Good use of market data")
        elif data_driven < 0.3:
            improvements.append("Reference market data to support your ask")
        
        if collaboration > 0.7:
            strengths.append("Collaborative approach")
        elif collaboration < 0.3:
            improvements.append("Show willingness to find mutually beneficial solutions")
        
        # Generate suggested response
        suggested = self._generate_suggested_response(tactic, self.current_scenario)
        
        # Next steps
        next_steps = []
        if self.round_number < 3:
            next_steps.append("Consider another round of negotiation")
        next_steps.append("Get the final offer in writing")
        next_steps.append("Review the complete compensation package")
        
        return NegotiationFeedback(
            assertiveness_score=assertiveness,
            professionalism_score=professionalism,
            data_driven_score=data_driven,
            collaboration_score=collaboration,
            overall_score=overall,
            strengths=strengths,
            improvements=improvements,
            suggested_response=suggested,
            tactic_used=tactic,
            next_steps=next_steps,
        )

    def _generate_employer_response(self, feedback: NegotiationFeedback, tactic: str) -> Dict:
        """Generate employer response based on negotiation quality."""
        scenario = self.current_scenario
        
        if feedback.overall_score >= 0.7:
            # Strong negotiation - employer concedes more
            return {
                "response_type": "concession",
                "message": "We appreciate your thoughtful approach. We can increase the signing bonus and adjust the base salary.",
                "new_offer": {
                    "base_salary": scenario.initial_offer.base_salary + 10000,
                    "signing_bonus": scenario.initial_offer.signing_bonus + 5000,
                    "equity_annual": scenario.initial_offer.equity_annual + 5000,
                },
                "remaining_objections": [],
            }
        elif feedback.overall_score >= 0.4:
            # Moderate negotiation - partial concession
            return {
                "response_type": "partial_concession",
                "message": "We can meet you partway. We'll increase the signing bonus but the base is at the top of our band.",
                "new_offer": {
                    "base_salary": scenario.initial_offer.base_salary,
                    "signing_bonus": scenario.initial_offer.signing_bonus + 3000,
                    "equity_annual": scenario.initial_offer.equity_annual,
                },
                "remaining_objections": scenario.obje_ctions[:1],
            }
        else:
            # Weak negotiation - employer holds firm
            return {
                "response_type": "hold_firm",
                "message": "This is our best offer for this level. We've already considered your experience.",
                "new_offer": {
                    "base_salary": scenario.initial_offer.base_salary,
                    "signing_bonus": scenario.initial_offer.signing_bonus,
                    "equity_annual": scenario.initial_offer.equity_annual,
                },
                "remaining_objections": scenario.obje_ctions,
            }

    def _generate_suggested_response(self, tactic: str, scenario: NegotiationScenario) -> str:
        """Generate a suggested response based on tactic."""
        if tactic == "collaborative":
            return (
                f"Thank you for the offer! I'm excited about the opportunity. "
                f"Based on my research, the market rate for this role is around "
                f"${scenario.market_data.base_salary:,}. Would it be possible to "
                f"adjust the compensation to be more in line with market data? "
                f"I'm confident we can find a mutually beneficial arrangement."
            )
        elif tactic == "competitive":
            return (
                f"I appreciate the offer. However, based on my experience and "
                f"market data, I believe a base salary of ${scenario.market_data.base_salary:,} "
                f"would be more appropriate. I've also received competing offers "
                f"at this level and I'd like to explore how we can bridge this gap."
            )
        else:
            return (
                f"Thank you for the offer. I'd like to discuss the compensation "
                f"package. I believe there's room for adjustment based on my "
                f"qualifications and market standards."
            )

    def _advance_phase(self):
        """Advance to next negotiation phase."""
        self.round_number += 1
        
        if self.current_phase == NegotiationPhase.INITIAL_OFFER:
            self.current_phase = NegotiationPhase.COUNTER_OFFER
        elif self.current_phase == NegotiationPhase.COUNTER_OFFER:
            self.current_phase = NegotiationPhase.OBJECTION_HANDLING
        elif self.current_phase == NegotiationPhase.OBJECTION_HANDLING:
            self.current_phase = NegotiationPhase.CLOSING

    def get_negotiation_summary(self) -> Dict:
        """Get summary of the negotiation practice."""
        if not self.negotiation_history:
            return {"status": "no_negotiation"}
        
        # Calculate average scores
        all_feedback = [h["feedback"] for h in self.negotiation_history]
        avg_scores = {
            "assertiveness": sum(f.assertiveness_score for f in all_feedback) / len(all_feedback),
            "professionalism": sum(f.professionalism_score for f in all_feedback) / len(all_feedback),
            "data_driven": sum(f.data_driven_score for f in all_feedback) / len(all_feedback),
            "collaboration": sum(f.collaboration_score for f in all_feedback) / len(all_feedback),
            "overall": sum(f.overall_score for f in all_feedback) / len(all_feedback),
        }
        
        # Get final offer
        final_response = self.negotiation_history[-1]["feedback"]
        
        # Calculate improvement
        if len(all_feedback) >= 2:
            first_score = all_feedback[0].overall_score
            last_score = all_feedback[-1].overall_score
            improvement = last_score - first_score
        else:
            improvement = 0
        
        return {
            "total_rounds": self.round_number - 1,
            "average_scores": avg_scores,
            "final_overall_score": avg_scores["overall"],
            "improvement": improvement,
            "tactics_used": list(set(h["tactic"] for h in self.negotiation_history)),
            "recommendation": self._get_recommendation(avg_scores["overall"]),
            "key_takeaways": self._get_key_takeaways(all_feedback),
        }

    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on negotiation score."""
        if score >= 0.8:
            return "Excellent negotiation! You'd likely secure a strong compensation package."
        elif score >= 0.6:
            return "Good negotiation. With more practice, you could maximize your offer."
        elif score >= 0.4:
            return "Decent attempt. Focus on using more data and being more assertive."
        else:
            return "Keep practicing. Focus on market research and assertive communication."

    def _get_key_takeaways(self, feedback_list: List[NegotiationFeedback]) -> List[str]:
        """Extract key takeaways from negotiation."""
        takeaways = []
        
        # Analyze patterns
        avg_assertiveness = sum(f.assertiveness_score for f in feedback_list) / len(feedback_list)
        avg_data = sum(f.data_driven_score for f in feedback_list) / len(feedback_list)
        
        if avg_assertiveness < 0.5:
            takeaways.append("Practice being more assertive in stating your value and expectations")
        
        if avg_data < 0.5:
            takeaways.append("Always reference market data (Glassdoor, Levels.fyi) to support your ask")
        
        takeaways.append("Focus on total compensation, not just base salary")
        takeaways.append("Get the final offer in writing before accepting")
        
        return takeaways

    def list_scenarios(self) -> List[Dict]:
        """List available negotiation scenarios."""
        return [{
            "scenario_id": s.scenario_id,
            "company_type": s.company_type,
            "role_level": s.role_level,
            "expected_range": s.expected_range,
        } for s in self.scenarios]


# Convenience function
def create_negotiation_practice() -> SalaryNegotiationPractice:
    """Create a salary negotiation practice instance."""
    return SalaryNegotiationPractice()
