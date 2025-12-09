"""
Enhanced Evaluation Module for AI Marketing Strategy System
"""

from __future__ import annotations

import json
import difflib
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Optional imports for enhanced metrics
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.spatial.distance import cosine
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ---- Token Metering --------------------------------------------------------
_TOKEN_METER = {"total_prompt": 0, "total_completion": 0, "cost_usd": 0.0}

def init_litellm_token_meter():
    """Initialize LiteLLM token tracking if available."""
    try:
        from litellm import callbacks
    except Exception:
        return

    def _on_success(**kwargs):
        try:
            resp = kwargs.get("response", {}) or {}
            usage = resp.get("usage", {}) or {}
            _TOKEN_METER["total_prompt"] += int(usage.get("prompt_tokens", 0))
            _TOKEN_METER["total_completion"] += int(usage.get("completion_tokens", 0))
            _TOKEN_METER["cost_usd"] += float(resp.get("cost", 0.0))
        except Exception:
            pass

    callbacks.on_success = getattr(callbacks, "on_success", [])
    callbacks.on_success.append(_on_success)


# =============================================================================
# GROUNDING METRICS (Feedback Item #5)
# =============================================================================

class GroundingScorer:
    """
    Calculate grounding completeness scores.
    """
    
    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract all numeric values from text."""
        # Match integers, decimals, percentages, currency
        pattern = r'[\$]?(\d+(?:,\d{3})*(?:\.\d+)?)\s*%?'
        matches = re.findall(pattern, text)
        numbers = []
        for m in matches:
            try:
                numbers.append(float(m.replace(',', '')))
            except:
                pass
        return numbers
    
    @staticmethod
    def score_grounding(doc: dict, user_inputs: Dict[str, str]) -> Dict[str, Any]:
        """
        Calculate comprehensive grounding score.
        """
        text = (doc.get("strategy_text", "") + " " + 
                json.dumps(doc.get("kpis", [])) + " " +
                json.dumps(doc.get("weaknesses", [])) + " " +
                json.dumps(doc.get("tactics", []))).lower()
        
        industry = (user_inputs.get("industry") or "").lower()
        audience = (user_inputs.get("target_audience") or "").lower()
        sales_data = (user_inputs.get("sales_data") or "").lower()
        
        result = {
            "industry_mentions": 0,
            "audience_mentions": 0,
            "data_citations": 0,
            "grounding_details": [],
        }
        
        # Count industry mentions
        if industry and industry != "general":
            result["industry_mentions"] = text.count(industry)
            # Also check for partial matches
            industry_words = [w for w in industry.split() if len(w) > 3]
            for word in industry_words:
                result["industry_mentions"] += text.count(word)
            if result["industry_mentions"] > 0:
                result["grounding_details"].append(f"Industry '{industry}' referenced {result['industry_mentions']} times")
        
        # Count audience mentions
        if audience and audience != "general audience":
            result["audience_mentions"] = text.count(audience)
            # Check for demographic patterns
            audience_patterns = re.findall(r'women?\s+\d+[-–]\d+|\d+[-–]\d+\s+(?:year|age)', text)
            result["audience_mentions"] += len(audience_patterns)
            if result["audience_mentions"] > 0:
                result["grounding_details"].append(f"Audience referenced {result['audience_mentions']} times")
        
        # Count data citations
        if sales_data:
            # Extract key numbers from sales data
            sales_numbers = GroundingScorer.extract_numbers(sales_data)
            strategy_numbers = GroundingScorer.extract_numbers(text)
            
            # Count overlapping numbers (data citations)
            for num in sales_numbers:
                if num in strategy_numbers or any(abs(num - sn) < 0.01 for sn in strategy_numbers):
                    result["data_citations"] += 1
            
            # Also check for key terms from sales data
            key_terms = ["cac", "cltv", "aov", "conversion", "revenue", "engagement"]
            for term in key_terms:
                if term in sales_data and term in text:
                    result["data_citations"] += 1
            
            if result["data_citations"] > 0:
                result["grounding_details"].append(f"Data points cited: {result['data_citations']}")
        
        # Calculate overall grounding score (0-100)
        # Weights: industry (30), audience (30), data citations (40)
        industry_score = min(result["industry_mentions"] * 15, 30)  # Max 30 points
        audience_score = min(result["audience_mentions"] * 15, 30)  # Max 30 points
        data_score = min(result["data_citations"] * 10, 40)  # Max 40 points
        
        result["grounding_score"] = industry_score + audience_score + data_score
        result["grounding_breakdown"] = {
            "industry_score": industry_score,
            "audience_score": audience_score,
            "data_citation_score": data_score,
        }
        
        return result


# =============================================================================
# VOCABULARY DIVERGENCE METRICS (Feedback Item #6)
# =============================================================================

class DivergenceMetrics:
    """
    Formalized vocabulary divergence calculations.
    
    Provides rigorous metrics with proper mathematical definitions
    and length normalization.
    """
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        return [w.lower() for w in re.findall(r'\b\w+\b', text)]
    
    @staticmethod
    def jaccard_distance(text1: str, text2: str) -> float:
        """
        Jaccard distance between two texts.
        """
        words1 = set(DivergenceMetrics.tokenize(text1))
        words2 = set(DivergenceMetrics.tokenize(text2))
        
        if not words1 and not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return 1.0 - (intersection / union) if union > 0 else 0.0
    
    @staticmethod
    def cosine_tfidf_distance(text1: str, text2: str) -> float:
        """
        Cosine distance over TF-IDF vectors.
        """
        if not SKLEARN_AVAILABLE:
            # Fall back to Jaccard if sklearn not available
            return DivergenceMetrics.jaccard_distance(text1, text2)
        
        try:
            vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
            tfidf = vectorizer.fit_transform([text1, text2])
            
            vec1 = tfidf[0].toarray().flatten()
            vec2 = tfidf[1].toarray().flatten()
            
            # Handle zero vectors
            if np.sum(vec1) == 0 or np.sum(vec2) == 0:
                return 1.0
            
            return float(cosine(vec1, vec2))
        except Exception:
            return DivergenceMetrics.jaccard_distance(text1, text2)
    
    @staticmethod
    def length_normalized_divergence(text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate divergence with length normalization.
        """
        len1 = len(DivergenceMetrics.tokenize(text1))
        len2 = len(DivergenceMetrics.tokenize(text2))
        
        raw_jaccard = DivergenceMetrics.jaccard_distance(text1, text2)
        raw_cosine = DivergenceMetrics.cosine_tfidf_distance(text1, text2)
        
        # Length ratio (>1 means text2 is longer)
        length_ratio = len2 / max(len1, 1)
        
        # Normalized divergence: penalize if divergence comes mainly from length
        # If text2 is much longer, some "new" words are expected
        expected_divergence = min(0.5, (length_ratio - 1) * 0.2) if length_ratio > 1 else 0
        adjusted_jaccard = max(0, raw_jaccard - expected_divergence)
        
        return {
            "jaccard_raw": round(raw_jaccard, 4),
            "cosine_tfidf": round(raw_cosine, 4),
            "length_ratio": round(length_ratio, 3),
            "jaccard_length_normalized": round(adjusted_jaccard, 4),
            "text1_tokens": len1,
            "text2_tokens": len2,
        }
    
    @staticmethod
    def calculate_edit_ratio(original: str, modified: str) -> float:
        """
        Calculate edit ratio between two texts.
        Uses SequenceMatcher for accurate diff calculation.
        """
        if not original:
            return 1.0
        
        sm = difflib.SequenceMatcher(a=original, b=modified)
        return round(1.0 - sm.ratio(), 4)


# =============================================================================
# ENHANCED RUBRIC SCORING (Feedback Item #4)
# =============================================================================

class EnhancedRubric:
    """
    Enhanced rubric with continuous scoring for items 8-10.
    """
    
    @staticmethod
    def score_measurement_methods(kpis: List[dict]) -> float:
        """
        Score KPI measurement methods (0-1 continuous).
        """
        if not kpis:
            return 0.0
        
        scores = []
        for kpi in kpis:
            method = kpi.get("measurement_method", "")
            if not method:
                scores.append(0.0)
            elif len(method) < 20:  # Very brief
                scores.append(0.25)
            elif len(method) < 50:  # Basic
                scores.append(0.5)
            elif any(term in method.lower() for term in ["track", "measure", "calculate", "via", "using"]):
                scores.append(0.75)
            else:
                scores.append(1.0)
        
        return round(np.mean(scores), 2) if scores else 0.0
    
    @staticmethod
    def score_implementation_phases(phases: List[dict]) -> float:
        """
        Score implementation phases (0-1 continuous).
        
        Considers: count, detail level, milestone specificity
        """
        if not phases:
            return 0.0
        
        base_score = min(len(phases) / 4, 1.0) * 0.4  # Up to 0.4 for having phases
        
        detail_scores = []
        for phase in phases:
            phase_score = 0.0
            if phase.get("phase"):
                phase_score += 0.25
            if phase.get("focus"):
                phase_score += 0.25
            milestones = phase.get("milestones", [])
            if milestones:
                phase_score += min(len(milestones) / 3, 0.5)
            detail_scores.append(phase_score)
        
        detail_avg = np.mean(detail_scores) if detail_scores else 0
        
        return round(base_score + detail_avg * 0.6, 2)
    
    @staticmethod
    def score_team_requirements(team: List[dict]) -> float:
        """
        Score team requirements (0-1 continuous).
        
        Considers: role count, FTE specificity, responsibility detail
        """
        if not team:
            return 0.0
        
        base_score = min(len(team) / 5, 0.4)  # Up to 0.4 for having team
        
        detail_scores = []
        for member in team:
            member_score = 0.0
            if member.get("role"):
                member_score += 0.3
            if member.get("fte") is not None:
                member_score += 0.3
            if member.get("responsibilities"):
                resp = member["responsibilities"]
                member_score += min(len(resp) / 50, 0.4)  # Up to 0.4 for detailed responsibilities
            detail_scores.append(member_score)
        
        detail_avg = np.mean(detail_scores) if detail_scores else 0
        
        return round(base_score + detail_avg * 0.6, 2)


# =============================================================================
# CORE EVALUATOR
# =============================================================================

class EvaluationMetrics:
    """Enhanced evaluation metrics with grounding and divergence analysis."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.metrics: Dict[str, Any] = {}
        self.grounding_scorer = GroundingScorer()
        self.divergence_metrics = DivergenceMetrics()
        self.enhanced_rubric = EnhancedRubric()

    def _extract_json(self, raw: str) -> Optional[dict]:
        """Best-effort JSON extraction from raw LLM response."""
        if not raw:
            return None
        raw = raw.strip()
        
        # Remove Markdown fences
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if "\n" in raw:
                parts = raw.split("\n", 1)
                if parts[0].isalpha() or parts[0].startswith("json"):
                    raw = parts[1]
        
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        
        try:
            return json.loads(raw[start:end+1])
        except Exception:
            return None

    def validate_schema(self, doc: dict) -> Dict[str, Any]:
        """Validate document against required schema."""
        issues: List[str] = []

        def req(key, typ):
            if key not in doc:
                issues.append(f"missing:{key}")
                return
            if not isinstance(doc[key], typ):
                issues.append(f"type:{key} expected {typ.__name__}")

        req("strategy_text", str)
        req("weaknesses", list)
        req("kpis", list)
        req("competitors", list)
        req("tactics", list)
        req("budget", dict)
        req("benchmarks", list)
        req("assumptions", list)

        # Weakness count validation
        if isinstance(doc.get("weaknesses"), list) and not (3 <= len(doc["weaknesses"]) <= 5):
            issues.append("weaknesses_count_out_of_range")

        # Competitor validation
        comps = doc.get("competitors") or []
        if isinstance(comps, list) and len(comps) < 2:
            issues.append("too_few_competitors")

        # Budget validation
        b = doc.get("budget") or {}
        items = (b.get("items") or []) if isinstance(b, dict) else []
        if items and all(isinstance(it.get("usd"), (int, float)) for it in items) and isinstance(b.get("total_usd"), (int, float)):
            total_items = sum(it["usd"] for it in items)
            if abs(total_items - b["total_usd"]) > 1e-6:
                issues.append(f"budget_mismatch items_sum={total_items} total={b['total_usd']}")
        else:
            issues.append("budget_numbers_missing")

        # KPI validation
        kpi_bad = 0
        for k in doc.get("kpis", []):
            if not isinstance(k, dict) or "name" not in k:
                kpi_bad += 1
                continue
            tf_ok = isinstance(k.get("timeframe_days"), (int, float))
            tgt_ok = isinstance(k.get("target"), (int, float, str))
            if not (tf_ok and tgt_ok):
                kpi_bad += 1
            if isinstance(k.get("target"), str) and ("X%" in k["target"] or "Y%" in k["target"]):
                kpi_bad += 1
        if kpi_bad:
            issues.append(f"kpi_invalid_count={kpi_bad}")

        return {"valid": len(issues) == 0, "issues": issues}

    def score_strategy_doc(self, doc: dict, user_inputs: Dict[str, str]) -> Dict[str, Any]:
        """
        Enhanced rubric scoring with continuous metrics.
        
        Now uses 0-1 continuous scoring for items 8-10.
        """
        industry = (user_inputs.get("industry") or "").lower()
        audience = (user_inputs.get("target_audience") or "").lower()
        sales_data = (user_inputs.get("sales_data") or "").lower()

        score = 0.0
        out = {"checks": {}}

        # 1) weaknesses 3–5 (binary)
        ok_weak = isinstance(doc.get("weaknesses"), list) and 3 <= len(doc["weaknesses"]) <= 5
        out["checks"]["weaknesses_3_to_5"] = ok_weak
        score += 1.0 if ok_weak else 0.0

        # 2) competitors ≥2 (binary)
        comps = doc.get("competitors") or []
        ok_comp = isinstance(comps, list) and len(comps) >= 2 and all(
            isinstance(c, dict) and isinstance(c.get("name"), str) and c.get("name") 
            for c in comps
        )
        out["checks"]["competitors>=2"] = ok_comp
        score += 1.0 if ok_comp else 0.0

        # 3) numeric KPIs + timeframe (binary)
        kpis = doc.get("kpis") or []
        ok_kpi = True if kpis else False
        for k in kpis:
            if not isinstance(k, dict) or "timeframe_days" not in k or "target" not in k:
                ok_kpi = False
                break
            if isinstance(k["target"], str) and ("X%" in k["target"] or "Y%" in k["target"]):
                ok_kpi = False
                break
        out["checks"]["kpis_numeric_timebound"] = ok_kpi
        score += 1.0 if ok_kpi else 0.0

        # 4) ROI prioritization (binary)
        tacts = doc.get("tactics") or []
        ok_roi = bool(tacts) and all(isinstance(t, dict) and ("roi_rank" in t) for t in tacts)
        out["checks"]["tactics_have_roi_rank"] = ok_roi
        score += 1.0 if ok_roi else 0.0

        # 5) budget consistency (binary)
        b = doc.get("budget") or {}
        items = b.get("items") or []
        ok_budget = (isinstance(b.get("total_usd"), (int, float)) and 
                     isinstance(items, list) and len(items) > 0 and
                     abs(sum((it.get("usd") or 0) for it in items) - (b.get("total_usd") or 0)) < 1e-6 and
                     (b.get("total_usd") or 0) > 0)
        out["checks"]["budget_consistent"] = ok_budget
        score += 1.0 if ok_budget else 0.0

        # 6) context mentions (binary)
        text = (doc.get("strategy_text") or "").lower()
        ok_ctx = (industry and industry in text) and (audience and audience in text)
        out["checks"]["context_mentions_industry_audience"] = ok_ctx
        score += 1.0 if ok_ctx else 0.0

        # 7) references sales data (binary)
        ok_sales = False
        if sales_data and text:
            toks = [t.strip() for t in sales_data.split() if t.strip()]
            ok_sales = any(t in text for t in toks[:5])
        out["checks"]["references_sales_data"] = ok_sales
        score += 1.0 if ok_sales else 0.0

        # 8) KPI measurement methods (CONTINUOUS 0-1)
        measurement_score = self.enhanced_rubric.score_measurement_methods(kpis)
        out["checks"]["kpis_have_measurement_methods"] = measurement_score
        score += measurement_score

        # 9) Implementation phases (CONTINUOUS 0-1)
        phases = doc.get("implementation_phases") or []
        phase_score = self.enhanced_rubric.score_implementation_phases(phases)
        out["checks"]["has_implementation_phases"] = phase_score
        score += phase_score

        # 10) Team requirements (CONTINUOUS 0-1)
        team = doc.get("team_requirements") or []
        team_score = self.enhanced_rubric.score_team_requirements(team)
        out["checks"]["has_team_requirements"] = team_score
        score += team_score

        out["score_0_to_10"] = round(score, 2)
        out["score_0_to_7"] = min(round(score), 7)  # Legacy compatibility
        out["percent"] = round(100 * score / 10, 1)
        
        return out

    def score_grounding(self, doc: dict, user_inputs: Dict[str, str]) -> Dict[str, Any]:
        """Calculate grounding score for the strategy document."""
        return self.grounding_scorer.score_grounding(doc, user_inputs)

    def calculate_divergence(self, text1: str, text2: str) -> Dict[str, Any]:
        """Calculate divergence metrics between two texts."""
        return self.divergence_metrics.length_normalized_divergence(text1, text2)

    def compare_openai_vs_claude(self, base_doc: dict, refined_doc: dict) -> Dict[str, Any]:
        """Compare OpenAI and Claude outputs with enhanced metrics."""
        base_txt = base_doc.get("strategy_text", "") if isinstance(base_doc, dict) else ""
        ref_txt = refined_doc.get("strategy_text", "") if isinstance(refined_doc, dict) else ""
        
        # Calculate divergence
        divergence = self.calculate_divergence(base_txt, ref_txt)
        
        base_kpis = len(base_doc.get("kpis") or []) if isinstance(base_doc, dict) else 0
        ref_kpis = len(refined_doc.get("kpis") or []) if isinstance(refined_doc, dict) else 0

        delta = {
            "edit_ratio": divergence["jaccard_raw"],
            "edit_ratio_normalized": divergence["jaccard_length_normalized"],
            "cosine_divergence": divergence["cosine_tfidf"],
            "length_ratio": divergence["length_ratio"],
            "kpis_delta": ref_kpis - base_kpis,
            "tactics_delta": len((refined_doc.get("tactics") or [])) - len((base_doc.get("tactics") or [])),
            "has_budget_now": bool((refined_doc.get("budget") or {}).get("total_usd")),
        }
        return delta

    def track_functionality(self, 
                            execution_successful: bool,
                            all_tasks_completed: bool,
                            output_file_generated: bool,
                            error_log: str = "") -> Dict[str, Any]:
        """Track basic functionality metrics."""
        functionality = {
            "execution_successful": execution_successful,
            "all_tasks_completed": all_tasks_completed,
            "output_file_generated": output_file_generated,
            "overall_pass": all([execution_successful, all_tasks_completed, output_file_generated]),
            "error_log": error_log,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics["functionality"] = functionality
        return functionality

    def track_performance(self,
                          execution_time: float,
                          token_usage: Dict[str, int],
                          cost_estimate: float) -> Dict[str, Any]:
        """Track performance metrics."""
        performance = {
            "execution_time_seconds": execution_time,
            "total_tokens": int(token_usage.get("prompt_tokens", 0)) + int(token_usage.get("completion_tokens", 0)),
            "tokens_by_type": token_usage,
            "cost_usd": float(cost_estimate),
            "timestamp": datetime.now().isoformat()
        }
        self.metrics["performance"] = performance
        return performance

    def generate_report(self, test_case_name: str = "default") -> str:
        """Generate comprehensive evaluation report."""
        report_path = self.output_dir / f"evaluation_report_{test_case_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        full_report = {
            "test_case": test_case_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "summary": self._generate_summary()
        }
        with open(report_path, 'w', encoding="utf-8") as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {test_case_name}")
        print(f"{'='*60}\n")
        print(json.dumps(full_report["summary"], indent=2))
        print(f"\nFull report saved to: {report_path}")
        return str(report_path)

    def write_markdown_summary(self, test_case_name: str):
        """Write Markdown summary."""
        md_lines = [
            "# Evaluation Summary",
            f"- **Case**: {test_case_name}",
            f"- **When**: {datetime.now().isoformat()}",
            "## Results"
        ]
        for k, v in self.metrics.items():
            md_lines.append(f"### {k}\n```json\n{json.dumps(v, indent=2, ensure_ascii=False)}\n```")
        out = self.output_dir / f"evaluation_report_{test_case_name}.md"
        out.write_text("\n\n".join(md_lines), encoding="utf-8")
        print(f"Markdown summary saved to: {out}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate executive summary."""
        summary: Dict[str, Any] = {}

        if "functionality" in self.metrics:
            func = self.metrics["functionality"]
            summary["functionality_status"] = "PASS" if func.get("overall_pass") else "FAIL"

        if "performance" in self.metrics:
            perf = self.metrics["performance"]
            summary["execution_time"] = f"{perf.get('execution_time_seconds', 0):.2f}s"
            summary["total_cost"] = f"${float(perf.get('cost_usd', 0.0)):.4f}"

        if "schema" in self.metrics:
            schema = self.metrics["schema"]
            summary["schema_valid"] = bool(schema.get("valid"))
            summary["schema_issues"] = schema.get("issues", [])
        
        if "rubric" in self.metrics:
            rub = self.metrics["rubric"]
            summary["rubric_percent"] = rub.get("percent", 0.0)

        if "grounding_score" in self.metrics:
            summary["grounding_score"] = self.metrics["grounding_score"].get("grounding_score", 0)

        if "refinement_delta" in self.metrics:
            summary["refinement_delta"] = self.metrics["refinement_delta"]

        return summary


# =============================================================================
# EVALUATION HARNESS
# =============================================================================

class EvaluationHarness:
    """Wrapper to run evaluations on crew executions."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.evaluator = EvaluationMetrics(output_dir=output_dir)
        self.baseline_text = self._load_baseline()

    def _load_baseline(self) -> str:
        """Load baseline marketing strategy document if present."""
        try:
            return Path('marketing_strategy_report.md').read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def run_evaluation(self, 
                       crew_result: Any,
                       user_inputs: Dict[str, str],
                       execution_time: float,
                       test_case_name: str = "default",
                       final_output_path: Optional[Path] = None,
                       openai_output_path: Optional[Path] = None,
                       save_individual_report: bool = True) -> Dict[str, Any]:
        """
        Run complete evaluation suite with enhanced metrics.
        
        Args:
            save_individual_report: If False, skip saving JSON/MD reports (useful for multi-run aggregation)
        """
        
        # 1) Functionality
        execution_successful = crew_result is not None
        path_to_read = final_output_path or Path(f"output/{user_inputs.get('module_name', 'marketing_strategy_report.md')}")
        output_file_generated = path_to_read.exists()

        error_log = ""
        if not execution_successful:
            error_log = str(getattr(crew_result, 'error', 'Unknown error'))

        self.evaluator.track_functionality(
            execution_successful=execution_successful,
            all_tasks_completed=execution_successful,
            output_file_generated=output_file_generated,
            error_log=error_log
        )

        # 2) Performance
        token_usage = {
            "prompt_tokens": _TOKEN_METER["total_prompt"],
            "completion_tokens": _TOKEN_METER["total_completion"],
        }
        cost_estimate = _TOKEN_METER["cost_usd"]
        self.evaluator.track_performance(
            execution_time=execution_time,
            token_usage=token_usage,
            cost_estimate=cost_estimate
        )

        # 3) Parse document
        generated_text = path_to_read.read_text(encoding="utf-8") if output_file_generated else ""
        doc = self.evaluator._extract_json(generated_text)

        # Fallback parsing
        if not doc:
            claude_path = Path("output/claude_strategy.json")
            if claude_path.exists():
                try:
                    base_text = claude_path.read_text(encoding="utf-8")
                    doc = self.evaluator._extract_json(base_text)
                    if doc:
                        self.evaluator.metrics["parse_fallback"] = "claude_strategy.json"
                except Exception:
                    pass
        
        if not doc and openai_output_path and openai_output_path.exists():
            try:
                base_text = openai_output_path.read_text(encoding="utf-8")
                doc = self.evaluator._extract_json(base_text)
                if doc:
                    self.evaluator.metrics["parse_fallback"] = "openai_output_path"
            except Exception:
                pass

        # 4) Schema and rubric scoring
        if doc:
            schema = self.evaluator.validate_schema(doc)
            rubric = self.evaluator.score_strategy_doc(doc, user_inputs)
            grounding = self.evaluator.score_grounding(doc, user_inputs)
            
            self.evaluator.metrics["schema"] = schema
            self.evaluator.metrics["rubric"] = rubric
            self.evaluator.metrics["grounding_score"] = grounding
        else:
            self.evaluator.metrics["schema"] = {"valid": False, "issues": ["no_parseable_json"]}
            self.evaluator.metrics["rubric"] = {"score_0_to_10": 0, "percent": 0.0, "checks": {"parse_error": True}}
            self.evaluator.metrics["grounding_score"] = {"grounding_score": 0}

        # 5) Refinement delta with enhanced divergence metrics
        if openai_output_path and openai_output_path.exists():
            try:
                base_doc_text = openai_output_path.read_text(encoding="utf-8")
                base_doc = self.evaluator._extract_json(base_doc_text) or {}

                refined_doc = doc
                if not refined_doc or self.evaluator.metrics.get("parse_fallback") == "openai_output_path":
                    claude_path = Path("output/claude_strategy.json")
                    if claude_path.exists():
                        try:
                            claude_text = claude_path.read_text(encoding="utf-8")
                            refined_doc = self.evaluator._extract_json(claude_text)
                            if refined_doc:
                                self.evaluator.metrics["refinement_source"] = "claude_strategy.json"
                        except Exception:
                            pass

                if base_doc and refined_doc:
                    self.evaluator.metrics["refinement_delta"] = self.evaluator.compare_openai_vs_claude(
                        base_doc, refined_doc
                    )
            except Exception as e:
                self.evaluator.metrics["refinement_error"] = str(e)

        # 6) Reports - only save if requested
        if save_individual_report:
            self.evaluator.generate_report(test_case_name)
            self.evaluator.write_markdown_summary(test_case_name)

        return self.evaluator.metrics


# =============================================================================
# STANDALONE USAGE
# =============================================================================

def run_with_evaluation_example():
    """Example standalone usage."""
    harness = EvaluationHarness()

    test_case = {
        "name": "example_case",
        "inputs": {
            "module_name": "claude_strategy.txt",
            "current_strategy": "Instagram influencer partnerships",
            "industry": "E-commerce Fashion",
            "target_audience": "Women 18-35",
            "sales_data": "Q3 2024: $250K revenue, 15% conversion rate"
        }
    }

    start = time.time()
    crew_result = object()
    exec_time = time.time() - start

    harness.run_evaluation(
        crew_result=crew_result,
        user_inputs=test_case["inputs"],
        execution_time=exec_time,
        test_case_name=test_case["name"],
        final_output_path=Path("output/claude_strategy.txt"),
        openai_output_path=Path("output/openai_strategy.txt"),
    )


if __name__ == "__main__":
    run_with_evaluation_example()