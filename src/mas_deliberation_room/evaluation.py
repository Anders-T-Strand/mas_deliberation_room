

from __future__ import annotations

import json
import difflib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

# ---- Optional LiteLLM token/cost metering ----------------------------------
# If you're using LiteLLM in your project, call `init_litellm_token_meter()`
# once at startup (before running the crew) to capture prompt/completion tokens
# and cost in _TOKEN_METER below.
_TOKEN_METER = {"total_prompt": 0, "total_completion": 0, "cost_usd": 0.0}

def init_litellm_token_meter():
    try:
        from litellm import callbacks  # type: ignore
    except Exception:
        return  # LiteLLM not installed or not in use; skip silently

    def _on_success(**kwargs):
        try:
            resp = kwargs.get("response", {}) or {}
            usage = resp.get("usage", {}) or {}
            _TOKEN_METER["total_prompt"] += int(usage.get("prompt_tokens", 0))
            _TOKEN_METER["total_completion"] += int(usage.get("completion_tokens", 0))
            # Some LiteLLM providers include a 'cost' field
            _TOKEN_METER["cost_usd"] += float(resp.get("cost", 0.0))
        except Exception:
            # Don't ever throw inside callbacks
            pass

    callbacks.on_success = getattr(callbacks, "on_success", [])
    callbacks.on_success.append(_on_success)


# ---- Core Evaluator ---------------------------------------------------------

class EvaluationMetrics:
    """Tracks and calculates evaluation metrics for the AI board meeting system."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.metrics: Dict[str, Any] = {}

    # ---------- Utility: JSON parsing & validation ----------

    def _extract_json(self, raw: str) -> Optional[dict]:
        """Best-effort JSON extraction from a raw LLM response."""
        if not raw:
            return None
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw[start:end+1])
        except Exception:
            return None

    def validate_schema(self, doc: dict) -> Dict[str, Any]:
        """Hard checks against the required keys + types based on your task rules."""
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

        # specifics
        if isinstance(doc.get("weaknesses"), list) and not (3 <= len(doc["weaknesses"]) <= 5):
            issues.append("weaknesses_count_out_of_range")

        comps = doc.get("competitors") or []
        if isinstance(comps, list) and len(comps) < 2:
            issues.append("too_few_competitors")

        # budget totals
        b = doc.get("budget") or {}
        items = (b.get("items") or []) if isinstance(b, dict) else []
        if items and all(isinstance(it.get("usd"), (int, float)) for it in items) and isinstance(b.get("total_usd"), (int, float)):
            total_items = sum(it["usd"] for it in items)
            if abs(total_items - b["total_usd"]) > 1e-6:
                issues.append(f"budget_mismatch items_sum={total_items} total={b['total_usd']}")
        else:
            issues.append("budget_numbers_missing")

        # KPIs: numeric or % and timeframe_days present, and no placeholders
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

    # ---------- Rubric scoring (constraint-aware) ----------

    def score_strategy_doc(self, doc: dict, user_inputs: Dict[str, str]) -> Dict[str, Any]:
        """Deterministic rubric based on your acceptance criteria (0–7 points)."""
        industry = (user_inputs.get("industry") or "").lower()
        audience = (user_inputs.get("target_audience") or "").lower()
        sales_data = (user_inputs.get("sales_data") or "").lower()

        score = 0
        out = {"checks": {}}

        # 1) weaknesses 3–5
        ok_weak = isinstance(doc.get("weaknesses"), list) and 3 <= len(doc["weaknesses"]) <= 5
        out["checks"]["weaknesses_3_to_5"] = ok_weak; score += 1 if ok_weak else 0

        # 2) competitors ≥2 w/ names
        comps = doc.get("competitors") or []
        ok_comp = isinstance(comps, list) and len(comps) >= 2 and all(isinstance(c, dict) and isinstance(c.get("name"), str) and c.get("name") for c in comps)
        out["checks"]["competitors>=2"] = ok_comp; score += 1 if ok_comp else 0

        # 3) numeric KPIs + timeframe (no placeholders)
        kpis = doc.get("kpis") or []
        ok_kpi = True if kpis else False
        for k in kpis:
            if not isinstance(k, dict) or "timeframe_days" not in k or "target" not in k:
                ok_kpi = False; break
            if isinstance(k["target"], str) and ("X%" in k["target"] or "Y%" in k["target"]):
                ok_kpi = False; break
        out["checks"]["kpis_numeric_timebound"] = ok_kpi; score += 1 if ok_kpi else 0

        # 4) ROI prioritization present
        tacts = doc.get("tactics") or []
        ok_roi = bool(tacts) and all(isinstance(t, dict) and ("roi_rank" in t) for t in tacts)
        out["checks"]["tactics_have_roi_rank"] = ok_roi; score += 1 if ok_roi else 0

        # 5) budget totals consistent and > 0
        b = doc.get("budget") or {}
        items = b.get("items") or []
        ok_budget = isinstance(b.get("total_usd"), (int, float)) and isinstance(items, list) and len(items) > 0 \
                    and abs(sum((it.get("usd") or 0) for it in items) - (b.get("total_usd") or 0)) < 1e-6 \
                    and (b.get("total_usd") or 0) > 0
        out["checks"]["budget_consistent"] = ok_budget; score += 1 if ok_budget else 0

        # 6) mentions industry and audience in text
        text = (doc.get("strategy_text") or "").lower()
        ok_ctx = (industry and industry in text) and (audience and audience in text)
        out["checks"]["context_mentions_industry_audience"] = ok_ctx; score += 1 if ok_ctx else 0

        # 7) references sales_data at least once (simple token overlap heuristic)
        ok_sales = False
        if sales_data and text:
            toks = [t.strip() for t in sales_data.split() if t.strip()]
            ok_sales = any(t in text for t in toks[:5])  # look for a few salient tokens
        out["checks"]["references_sales_data"] = ok_sales; score += 1 if ok_sales else 0

        out["score_0_to_7"] = score
        out["percent"] = round(100 * score / 7, 1)
        return out

    # ---------- Refinement (OpenAI → Claude) delta ----------

    def compare_openai_vs_claude(self, base_doc: dict, refined_doc: dict) -> Dict[str, Any]:
        base_txt = base_doc.get("strategy_text","") if isinstance(base_doc, dict) else ""
        ref_txt  = refined_doc.get("strategy_text","") if isinstance(refined_doc, dict) else ""
        sm = difflib.SequenceMatcher(a=base_txt, b=ref_txt)
        edit_ratio = 1.0 - sm.quick_ratio()  # rough “how much changed”

        base_kpis = len(base_doc.get("kpis") or []) if isinstance(base_doc, dict) else 0
        ref_kpis  = len(refined_doc.get("kpis") or []) if isinstance(refined_doc, dict) else 0

        delta = {
            "edit_ratio": round(edit_ratio, 3),
            "kpis_delta": ref_kpis - base_kpis,
            "tactics_delta": len((refined_doc.get("tactics") or [])) - len((base_doc.get("tactics") or [])),
            "has_budget_now": bool((refined_doc.get("budget") or {}).get("total_usd")),
        }
        return delta

    # ---------- Legacy/basic trackers (kept for compatibility) ----------

    def track_functionality(self, 
                            execution_successful: bool,
                            all_tasks_completed: bool,
                            output_file_generated: bool,
                            error_log: str = "") -> Dict[str, Any]:
        """Track basic functionality metrics (Pass/Fail)."""
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
        """Track performance metrics, using real token/cost numbers if available."""
        performance = {
            "execution_time_seconds": execution_time,
            "total_tokens": int(token_usage.get("prompt_tokens", 0)) + int(token_usage.get("completion_tokens", 0)),
            "tokens_by_type": token_usage,
            "cost_usd": float(cost_estimate),
            "timestamp": datetime.now().isoformat()
        }
        self.metrics["performance"] = performance
        return performance

    # ---------- Reporting ----------

    def generate_report(self, test_case_name: str = "default") -> str:
        """Generate comprehensive evaluation report (JSON)."""
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
        """Write a human-friendly Markdown summary next to the JSON."""
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
        """Generate executive summary of all metrics."""
        summary: Dict[str, Any] = {}

        # Functionality
        if "functionality" in self.metrics:
            func = self.metrics["functionality"]
            summary["functionality_status"] = "PASS" if func.get("overall_pass") else "FAIL"

        # Performance
        if "performance" in self.metrics:
            perf = self.metrics["performance"]
            summary["execution_time"] = f"{perf.get('execution_time_seconds', 0):.2f}s"
            summary["total_cost"] = f"${float(perf.get('cost_usd', 0.0)):.4f}"

        # Schema/Rubric
        if "schema" in self.metrics:
            schema = self.metrics["schema"]
            summary["schema_valid"] = bool(schema.get("valid"))
            summary["schema_issues"] = schema.get("issues", [])
        if "rubric" in self.metrics:
            rub = self.metrics["rubric"]
            summary["rubric_percent"] = rub.get("percent", 0.0)

        # Refinement delta
        if "refinement_delta" in self.metrics:
            summary["refinement_delta"] = self.metrics["refinement_delta"]

        return summary


class EvaluationHarness:
    """Wrapper to run evaluations on crew executions (OpenAI → Claude chain supported)."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.evaluator = EvaluationMetrics(output_dir=output_dir)
        self.baseline_text = self._load_baseline()

    def _load_baseline(self) -> str:
        """Load baseline marketing strategy document if present (optional)."""
        try:
            return Path('marketing_strategy_report.md').read_text(encoding="utf-8")
        except FileNotFoundError:
            print("Warning: Baseline document not found")
            return ""

    def run_evaluation(self, 
                       crew_result: Any,
                       user_inputs: Dict[str, str],
                       execution_time: float,
                       test_case_name: str = "default",
                       final_output_path: Optional[Path] = None,
                       openai_output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run complete evaluation suite on a crew execution.

        Args:
            crew_result: Result object from crew.kickoff()
            user_inputs: Dictionary of user inputs used for the run
            execution_time: Time taken for execution in seconds
            test_case_name: Name for this test case
            final_output_path: Path to the final (Claude) strategy file
            openai_output_path: Path to the initial (OpenAI) strategy file, for delta comparison

        Returns:
            Dictionary containing all evaluation metrics.
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
            all_tasks_completed=execution_successful,  # Simplified assumption
            output_file_generated=output_file_generated,
            error_log=error_log
        )

        # 2) Performance (use real meter if initialized)
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

        # 3) Quality via strict JSON + rubric
        generated_text = path_to_read.read_text(encoding="utf-8") if output_file_generated else ""
        doc = self.evaluator._extract_json(generated_text)
        if doc:
            schema = self.evaluator.validate_schema(doc)
            rubric = self.evaluator.score_strategy_doc(doc, user_inputs)
            self.evaluator.metrics["schema"] = schema
            self.evaluator.metrics["rubric"] = rubric
        else:
            self.evaluator.metrics["schema"] = {"valid": False, "issues": ["no_parseable_json"]}
            self.evaluator.metrics["rubric"] = {"score_0_to_7": 0, "percent": 0.0, "checks": {"parse_error": True}}

        # 4) Optional: baseline comparison (kept from previous version)
        if self.baseline_text and generated_text:
            self._compare_to_baseline_legacy(generated_text)

        # 5) Optional: OpenAI→Claude delta
        if openai_output_path and openai_output_path.exists() and doc:
            try:
                base_doc_text = openai_output_path.read_text(encoding="utf-8")
                base_doc = self.evaluator._extract_json(base_doc_text) or {}
                self.evaluator.metrics["refinement_delta"] = self.evaluator.compare_openai_vs_claude(base_doc, doc or {})
            except Exception:
                pass

        # 6) Reports
        self.evaluator.generate_report(test_case_name)
        self.evaluator.write_markdown_summary(test_case_name)

        return self.evaluator.metrics

    # --- Legacy helper retained for those still using it ---
    def _compare_to_baseline_legacy(self, generated_text: str):
        comparison = {}
        gen_words = len(generated_text.split())
        base_words = len(self.baseline_text.split()) if self.baseline_text else 0
        comparison["generated_word_count"] = gen_words
        comparison["baseline_word_count"] = base_words
        comparison["length_ratio"] = round(gen_words / base_words, 2) if base_words > 0 else 0
        gen_vocab = set(generated_text.lower().split())
        base_vocab = set((self.baseline_text or "").lower().split())
        inter = len(gen_vocab & base_vocab)
        union = len(gen_vocab | base_vocab) or 1
        comparison["vocabulary_overlap"] = round(inter / union, 3)
        self.evaluator.metrics["baseline_comparison"] = comparison


# ---------------------------
# Example of standalone usage
# ---------------------------
def run_with_evaluation_example():
    """
    Example: call after your crew finishes. Adjust paths for your project.
    """
    # init_litellm_token_meter()  # uncomment if using LiteLLM in this process
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
    # ... run your crew here ...
    crew_result = object()  # placeholder for success
    exec_time = time.time() - start

    final_output = Path("output/claude_strategy.txt")      # Claude (refined)
    openai_output = Path("output/openai_strategy.txt")     # OpenAI (initial)

    harness.run_evaluation(
        crew_result=crew_result,
        user_inputs=test_case["inputs"],
        execution_time=exec_time,
        test_case_name=test_case["name"],
        final_output_path=final_output,
        openai_output_path=openai_output,
    )


if __name__ == "__main__":
    run_with_evaluation_example()
