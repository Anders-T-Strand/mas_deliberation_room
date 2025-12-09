#!/usr/bin/env python
"""
LLM-as-Judge Evaluation Module

Addresses Feedback Item #4: Limited evaluation metrics
- Adds LLM-based evaluation for creativity, relevance, and actionability
- Provides triangulation with human evaluation criteria
- Measures aspects not captured by deterministic rubric
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Try to import LiteLLM for API calls
try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class LLMJudge:
    """
    LLM-as-Judge evaluation for marketing strategies.
    
    Uses a separate LLM to evaluate strategy quality on dimensions
    that are difficult to measure with deterministic rubrics.
    """
    
    CRITERIA = {
        "creativity": {
            "description": "Novel approaches, innovative tactics, unique positioning",
            "scoring_guide": """
                1-2: Generic, template-like strategy with no novel elements
                3-4: Some interesting ideas but mostly standard approaches
                5-6: Several creative tactics or unique positioning elements
                7-8: Notably innovative with multiple fresh approaches
                9-10: Highly creative, breakthrough thinking throughout
            """,
        },
        "actionability": {
            "description": "Clear next steps, implementable recommendations, practical guidance",
            "scoring_guide": """
                1-2: Vague recommendations, unclear what to do first
                3-4: Some actionable items but missing details or priorities
                5-6: Clear tactics with basic implementation guidance
                7-8: Well-defined actions with timelines and responsibilities
                9-10: Comprehensive action plan ready for immediate execution
            """,
        },
        "relevance": {
            "description": "Alignment with target audience, industry appropriateness, context fit",
            "scoring_guide": """
                1-2: Generic strategy not tailored to the specific context
                3-4: Some industry awareness but misses key audience needs
                5-6: Reasonably aligned with industry and audience
                7-8: Well-tailored to specific industry and audience characteristics
                9-10: Deeply relevant with precise targeting and industry expertise
            """,
        },
        "strategic_coherence": {
            "description": "Logical flow, internal consistency, goal alignment",
            "scoring_guide": """
                1-2: Disjointed tactics, conflicting recommendations
                3-4: Some logical gaps or inconsistencies
                5-6: Generally coherent with minor alignment issues
                7-8: Well-integrated strategy with clear goal connection
                9-10: Highly coherent, all elements synergistically aligned
            """,
        },
        "practical_feasibility": {
            "description": "Resource realism, implementation practicality, risk awareness",
            "scoring_guide": """
                1-2: Unrealistic expectations, ignores resource constraints
                3-4: Some practical concerns not addressed
                5-6: Generally feasible with standard resources
                7-8: Practical plan with realistic resource allocation
                9-10: Highly feasible with contingencies and risk mitigation
            """,
        },
    }
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 output_dir: str = "output"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluations: List[Dict] = []
    
    def evaluate_strategy(self,
                          strategy_doc: Dict,
                          user_inputs: Dict,
                          criteria: List[str] = None,
                          run_id: str = None) -> Dict[str, Any]:
        """
        Evaluate a strategy document using LLM-as-judge.
        """
        if not LITELLM_AVAILABLE:
            return self._mock_evaluation(strategy_doc, criteria)
        
        if criteria is None:
            criteria = list(self.CRITERIA.keys())
        
        results = {
            "run_id": run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "criteria_evaluated": criteria,
            "scores": {},
            "reasoning": {},
            "overall_score": None,
        }
        
        strategy_text = self._prepare_strategy_text(strategy_doc)
        context_text = self._prepare_context_text(user_inputs)
        
        for criterion in criteria:
            if criterion not in self.CRITERIA:
                continue
            
            score, reasoning = self._evaluate_criterion(
                strategy_text, context_text, criterion
            )
            
            results["scores"][criterion] = score
            results["reasoning"][criterion] = reasoning
        
        if results["scores"]:
            results["overall_score"] = round(
                sum(results["scores"].values()) / len(results["scores"]), 1
            )
        
        self.evaluations.append(results)
        self._save_evaluations()
        
        return results
    
    def _prepare_strategy_text(self, strategy_doc: Dict) -> str:
        """Prepare strategy document as readable text."""
        if isinstance(strategy_doc, str):
            return strategy_doc
        
        parts = []
        
        if strategy_doc.get("strategy_text"):
            parts.append(f"EXECUTIVE SUMMARY:\n{strategy_doc['strategy_text']}")
        
        if strategy_doc.get("weaknesses"):
            parts.append("\nWEAKNESSES IDENTIFIED:")
            for w in strategy_doc["weaknesses"]:
                if isinstance(w, dict):
                    parts.append(f"- {w.get('title', 'N/A')}: {w.get('description', 'N/A')}")
        
        if strategy_doc.get("kpis"):
            parts.append("\nKEY PERFORMANCE INDICATORS:")
            for k in strategy_doc["kpis"]:
                if isinstance(k, dict):
                    parts.append(f"- {k.get('name', 'N/A')}: Current {k.get('current')} -> Target {k.get('target')} in {k.get('timeframe_days')} days")
        
        if strategy_doc.get("tactics"):
            parts.append("\nRECOMMENDED TACTICS:")
            for t in strategy_doc["tactics"]:
                if isinstance(t, dict):
                    parts.append(f"- {t.get('name', 'N/A')} (ROI Rank: {t.get('roi_rank')}): {t.get('description', 'N/A')}")
        
        if strategy_doc.get("budget"):
            budget = strategy_doc["budget"]
            parts.append(f"\nBUDGET: ${budget.get('total_usd', 0):,.0f}")
            for item in budget.get("items", []):
                if isinstance(item, dict):
                    parts.append(f"  - {item.get('category', 'N/A')}: ${item.get('usd', 0):,.0f}")
        
        if strategy_doc.get("implementation_phases"):
            parts.append("\nIMPLEMENTATION PHASES:")
            for phase in strategy_doc["implementation_phases"]:
                if isinstance(phase, dict):
                    parts.append(f"- {phase.get('phase', 'N/A')}: {phase.get('focus', 'N/A')}")
        
        return "\n".join(parts)
    
    def _prepare_context_text(self, user_inputs: Dict) -> str:
        """Prepare context text from user inputs."""
        return f"""
CONTEXT:
- Industry: {user_inputs.get('industry', 'Not specified')}
- Target Audience: {user_inputs.get('target_audience', 'Not specified')}
- Current Strategy Summary: {user_inputs.get('current_strategy', 'Not provided')[:500] if user_inputs.get('current_strategy') else 'Not provided'}
"""
    
    def _evaluate_criterion(self, 
                            strategy_text: str, 
                            context_text: str,
                            criterion: str) -> Tuple[float, str]:
        """Evaluate a single criterion using LLM."""
        criterion_info = self.CRITERIA[criterion]
        
        prompt = f"""You are an expert marketing strategy evaluator. Evaluate the following marketing strategy on the criterion of "{criterion}".

CRITERION DEFINITION:
{criterion_info['description']}

SCORING GUIDE:
{criterion_info['scoring_guide']}

{context_text}

STRATEGY TO EVALUATE:
{strategy_text}

INSTRUCTIONS:
1. Carefully analyze the strategy against the criterion
2. Consider both strengths and weaknesses
3. Provide a score from 1-10
4. Explain your reasoning in 2-3 sentences

OUTPUT FORMAT (JSON only):
{{"score": <number 1-10>, "reasoning": "<your explanation>"}}

Respond with ONLY the JSON object, no additional text."""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if response_text.startswith("```"):
                response_text = response_text.strip("`").strip()
                if response_text.startswith("json"):
                    response_text = response_text[4:].strip()
            
            result = json.loads(response_text)
            score = float(result.get("score", 5))
            reasoning = result.get("reasoning", "No reasoning provided")
            
            return min(10, max(1, score)), reasoning
            
        except Exception as e:
            print(f"[WARN] LLM evaluation failed for {criterion}: {e}")
            return 5.0, f"Evaluation failed: {str(e)}"
    
    def _mock_evaluation(self, strategy_doc: Dict, criteria: List[str] = None) -> Dict[str, Any]:
        """Mock evaluation when LiteLLM is not available."""
        if criteria is None:
            criteria = list(self.CRITERIA.keys())
        
        import random
        random.seed(42)
        
        results = {
            "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "model": "mock",
            "criteria_evaluated": criteria,
            "scores": {},
            "reasoning": {},
            "note": "Mock evaluation - LiteLLM not available",
        }
        
        for criterion in criteria:
            results["scores"][criterion] = round(random.uniform(5, 8), 1)
            results["reasoning"][criterion] = f"Mock evaluation for {criterion}"
        
        results["overall_score"] = round(
            sum(results["scores"].values()) / len(results["scores"]), 1
        )
        
        return results
    
    def _save_evaluations(self):
        """Save evaluations to file."""
        output_file = self.output_dir / "llm_judge_evaluations.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluations, f, indent=2, ensure_ascii=False)
    
    def compare_strategies(self,
                           strategy_a: Dict,
                           strategy_b: Dict,
                           user_inputs: Dict,
                           labels: Tuple[str, str] = ("Strategy A", "Strategy B")) -> Dict[str, Any]:
        """
        Compare two strategies head-to-head.
        """
        eval_a = self.evaluate_strategy(strategy_a, user_inputs, run_id=f"compare_{labels[0]}")
        eval_b = self.evaluate_strategy(strategy_b, user_inputs, run_id=f"compare_{labels[1]}")
        
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "labels": labels,
            "strategy_a": eval_a,
            "strategy_b": eval_b,
            "comparison": {},
            "winner_by_criterion": {},
            "overall_winner": None,
        }
        
        for criterion in self.CRITERIA.keys():
            score_a = eval_a["scores"].get(criterion, 0)
            score_b = eval_b["scores"].get(criterion, 0)
            
            comparison["comparison"][criterion] = {
                "score_a": score_a,
                "score_b": score_b,
                "difference": round(score_a - score_b, 1),
            }
            
            if score_a > score_b:
                comparison["winner_by_criterion"][criterion] = labels[0]
            elif score_b > score_a:
                comparison["winner_by_criterion"][criterion] = labels[1]
            else:
                comparison["winner_by_criterion"][criterion] = "Tie"
        
        if eval_a["overall_score"] > eval_b["overall_score"]:
            comparison["overall_winner"] = labels[0]
        elif eval_b["overall_score"] > eval_a["overall_score"]:
            comparison["overall_winner"] = labels[1]
        else:
            comparison["overall_winner"] = "Tie"
        
        # Save comparison
        output_file = self.output_dir / "llm_judge_comparison.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        return comparison
    
    def generate_evaluation_report(self) -> str:
        """Generate markdown report of all evaluations."""
        if not self.evaluations:
            return "# LLM-as-Judge Evaluation Report\n\nNo evaluations recorded."
        
        report = f"""# LLM-as-Judge Evaluation Report

**Generated:** {datetime.now().isoformat()}
**Model Used:** {self.model}
**Total Evaluations:** {len(self.evaluations)}

## Evaluation Criteria

"""
        
        for criterion, info in self.CRITERIA.items():
            report += f"### {criterion.replace('_', ' ').title()}\n"
            report += f"{info['description']}\n\n"
        
        report += "## Evaluation Results\n\n"
        
        for eval_result in self.evaluations:
            report += f"### Run: {eval_result.get('run_id', 'N/A')}\n"
            report += f"**Timestamp:** {eval_result.get('timestamp', 'N/A')}\n"
            report += f"**Overall Score:** {eval_result.get('overall_score', 'N/A')}/10\n\n"
            
            report += "| Criterion | Score | Reasoning |\n"
            report += "|-----------|-------|----------|\n"
            
            for criterion in eval_result.get("criteria_evaluated", []):
                score = eval_result["scores"].get(criterion, "N/A")
                reasoning = eval_result["reasoning"].get(criterion, "N/A")
                # Truncate reasoning for table
                reasoning_short = reasoning[:100] + "..." if len(reasoning) > 100 else reasoning
                report += f"| {criterion} | {score} | {reasoning_short} |\n"
            
            report += "\n"
        
        # Summary statistics
        if len(self.evaluations) > 1:
            report += "## Summary Statistics\n\n"
            
            for criterion in self.CRITERIA.keys():
                scores = [e["scores"].get(criterion, 0) for e in self.evaluations if criterion in e.get("scores", {})]
                if scores:
                    avg = sum(scores) / len(scores)
                    report += f"- **{criterion}**: Avg {avg:.1f}/10 (n={len(scores)})\n"
        
        # Save report
        report_file = self.output_dir / "llm_judge_report.md"
        report_file.write_text(report, encoding='utf-8')
        print(f"[SAVED] LLM Judge report: {report_file}")
        
        return report


def evaluate_with_llm_judge(strategy_path: str = None, 
                            user_inputs: Dict = None,
                            output_dir: str = "output") -> Dict[str, Any]:
    """Convenience function to run LLM-as-judge evaluation."""
    
    if strategy_path:
        with open(strategy_path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                strategy_doc = json.loads(content)
            except:
                strategy_doc = {"strategy_text": content}
    else:
        strategy_doc = {}
    
    if user_inputs is None:
        user_inputs = {
            "industry": "Fashion E-commerce",
            "target_audience": "Women 25-45",
        }
    
    judge = LLMJudge(output_dir=output_dir)
    results = judge.evaluate_strategy(strategy_doc, user_inputs)
    judge.generate_evaluation_report()
    
    return results


if __name__ == "__main__":
    # Example usage
    sample_strategy = {
        "strategy_text": "This fashion e-commerce strategy targets women 25-45...",
        "tactics": [
            {"name": "Loyalty Program", "description": "Points-based system", "roi_rank": 1},
        ],
    }
    
    sample_inputs = {
        "industry": "Fashion E-commerce",
        "target_audience": "Women 25-45",
    }
    
    judge = LLMJudge()
    results = judge.evaluate_strategy(sample_strategy, sample_inputs)
    print(json.dumps(results, indent=2))