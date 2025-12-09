#!/usr/bin/env python
"""
Failure Analysis Module

Addresses Feedback Item #13: Missing discussion of failure cases
- Systematic failure tracking and categorization
- Quantification of failure rates
- Connection of failures to agent behaviors
- Root cause analysis
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class FailureCategory:
    """Enumeration of failure categories."""
    JSON_PARSE_ERROR = "json_parse_error"
    SCHEMA_VIOLATION = "schema_violation"
    GROUNDING_FAILURE = "grounding_failure"
    HALLUCINATION = "hallucination"
    TRUNCATION = "truncation"
    BUDGET_MISMATCH = "budget_mismatch"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    CONTEXT_LOSS = "context_loss"
    PROMPT_INJECTION = "prompt_injection"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    OTHER = "other"


class FailureAnalyzer:
    """
    Analyzes and categorizes failures in the multi-agent system.
    
    Provides:
    - Systematic failure tracking
    - Categorization by type
    - Agent-specific failure analysis
    - Root cause identification
    - Trend analysis over runs
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.failures: List[Dict[str, Any]] = []
        self.failure_log_file = self.output_dir / "failure_log.json"
        
        # Load existing failures if any
        self._load_existing_failures()
    
    def _load_existing_failures(self):
        """Load existing failure log if it exists."""
        if self.failure_log_file.exists():
            try:
                with open(self.failure_log_file, 'r', encoding='utf-8') as f:
                    self.failures = json.load(f)
            except:
                self.failures = []
    
    def log_failure(self, 
                    category: str,
                    agent: str,
                    description: str,
                    raw_output: str = None,
                    expected: str = None,
                    run_id: str = None,
                    severity: str = "medium",
                    recoverable: bool = True,
                    metadata: Dict = None) -> Dict[str, Any]:
        """
        Log a failure with detailed information.
        
        Args:
            category: Failure category from FailureCategory
            agent: Agent that produced the failure (openai, claude, gemini)
            description: Human-readable description of the failure
            raw_output: The actual output that failed
            expected: What was expected
            run_id: Identifier for the run
            severity: low, medium, high, critical
            recoverable: Whether the pipeline recovered from this failure
            metadata: Additional metadata
            
        Returns:
            The logged failure entry
        """
        failure_entry = {
            "id": f"F{len(self.failures)+1:04d}",
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "agent": agent,
            "description": description,
            "raw_output_preview": raw_output[:500] if raw_output else None,
            "expected": expected,
            "run_id": run_id,
            "severity": severity,
            "recoverable": recoverable,
            "metadata": metadata or {},
        }
        
        self.failures.append(failure_entry)
        self._save_failures()
        
        return failure_entry
    
    def analyze_output(self, 
                       output: str, 
                       agent: str,
                       expected_schema: Dict = None,
                       user_inputs: Dict = None,
                       run_id: str = None) -> List[Dict[str, Any]]:
        """
        Analyze an agent's output for potential failures.
        
        Args:
            output: The raw output from the agent
            agent: Agent name
            expected_schema: Expected JSON schema
            user_inputs: Original user inputs for grounding check
            run_id: Run identifier
            
        Returns:
            List of detected failures
        """
        detected_failures = []
        
        # 1. Check for JSON parse errors
        json_failure = self._check_json_parsing(output, agent, run_id)
        if json_failure:
            detected_failures.append(json_failure)
            return detected_failures  # Can't continue without valid JSON
        
        # Parse JSON for further analysis
        try:
            doc = self._extract_json(output)
        except:
            return detected_failures
        
        if not doc:
            return detected_failures
        
        # 2. Check for schema violations
        schema_failures = self._check_schema_violations(doc, agent, run_id)
        detected_failures.extend(schema_failures)
        
        # 3. Check for grounding failures
        if user_inputs:
            grounding_failures = self._check_grounding(doc, user_inputs, agent, run_id)
            detected_failures.extend(grounding_failures)
        
        # 4. Check for hallucinations
        hallucination_failures = self._check_hallucinations(doc, user_inputs, agent, run_id)
        detected_failures.extend(hallucination_failures)
        
        # 5. Check for truncation
        truncation_failure = self._check_truncation(output, agent, run_id)
        if truncation_failure:
            detected_failures.append(truncation_failure)
        
        # 6. Check for budget mismatches
        budget_failure = self._check_budget_consistency(doc, agent, run_id)
        if budget_failure:
            detected_failures.append(budget_failure)
        
        return detected_failures
    
    def _extract_json(self, raw: str) -> Optional[dict]:
        """Extract JSON from raw output."""
        if not raw:
            return None
        raw = raw.strip()
        
        # Remove markdown fences
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
        except:
            return None
    
    def _check_json_parsing(self, output: str, agent: str, run_id: str) -> Optional[Dict]:
        """Check for JSON parsing errors."""
        if not output:
            return self.log_failure(
                category=FailureCategory.JSON_PARSE_ERROR,
                agent=agent,
                description="Empty output received",
                run_id=run_id,
                severity="high",
                recoverable=False,
            )
        
        # Try to parse
        doc = self._extract_json(output)
        if doc is None:
            # Analyze the error type
            if "```" in output:
                error_type = "Markdown fences not properly handled"
            elif output.count("{") != output.count("}"):
                error_type = "Unbalanced braces"
            elif "..." in output or output.endswith("..."):
                error_type = "Output appears truncated"
            else:
                error_type = "Invalid JSON structure"
            
            return self.log_failure(
                category=FailureCategory.JSON_PARSE_ERROR,
                agent=agent,
                description=f"Failed to parse JSON: {error_type}",
                raw_output=output,
                run_id=run_id,
                severity="high",
                recoverable=False,
            )
        
        return None
    
    def _check_schema_violations(self, doc: dict, agent: str, run_id: str) -> List[Dict]:
        """Check for schema violations."""
        failures = []
        
        required_fields = [
            "strategy_text", "weaknesses", "kpis", "competitors",
            "tactics", "budget", "benchmarks", "assumptions"
        ]
        
        for field in required_fields:
            if field not in doc:
                failures.append(self.log_failure(
                    category=FailureCategory.MISSING_REQUIRED_FIELD,
                    agent=agent,
                    description=f"Missing required field: {field}",
                    run_id=run_id,
                    severity="medium",
                    recoverable=True,
                ))
        
        # Check specific field constraints
        if "weaknesses" in doc:
            weakness_count = len(doc["weaknesses"]) if isinstance(doc["weaknesses"], list) else 0
            if not (3 <= weakness_count <= 5):
                failures.append(self.log_failure(
                    category=FailureCategory.SCHEMA_VIOLATION,
                    agent=agent,
                    description=f"Weakness count {weakness_count} not in range [3,5]",
                    run_id=run_id,
                    severity="low",
                    recoverable=True,
                ))
        
        if "competitors" in doc:
            comp_count = len(doc["competitors"]) if isinstance(doc["competitors"], list) else 0
            if comp_count < 2:
                failures.append(self.log_failure(
                    category=FailureCategory.SCHEMA_VIOLATION,
                    agent=agent,
                    description=f"Only {comp_count} competitors (need at least 2)",
                    run_id=run_id,
                    severity="low",
                    recoverable=True,
                ))
        
        return failures
    
    def _check_grounding(self, doc: dict, user_inputs: Dict, agent: str, run_id: str) -> List[Dict]:
        """Check for grounding failures."""
        failures = []
        
        strategy_text = (doc.get("strategy_text") or "").lower()
        industry = (user_inputs.get("industry") or "").lower()
        audience = (user_inputs.get("target_audience") or "").lower()
        
        # Check industry mention
        if industry and industry != "general" and industry not in strategy_text:
            failures.append(self.log_failure(
                category=FailureCategory.GROUNDING_FAILURE,
                agent=agent,
                description=f"Strategy text does not mention industry: '{industry}'",
                expected=f"Should mention '{industry}'",
                run_id=run_id,
                severity="medium",
                recoverable=True,
            ))
        
        # Check audience mention
        if audience and audience != "general audience" and audience not in strategy_text:
            failures.append(self.log_failure(
                category=FailureCategory.GROUNDING_FAILURE,
                agent=agent,
                description=f"Strategy text does not mention target audience: '{audience}'",
                expected=f"Should mention '{audience}'",
                run_id=run_id,
                severity="medium",
                recoverable=True,
            ))
        
        return failures
    
    def _check_hallucinations(self, doc: dict, user_inputs: Dict, agent: str, run_id: str) -> List[Dict]:
        """Check for potential hallucinations."""
        failures = []
        
        # Check for unrealistic numbers
        budget = doc.get("budget", {})
        total = budget.get("total_usd", 0)
        
        # Flag extremely high or low budgets
        if total > 10_000_000:  # > $10M seems unrealistic
            failures.append(self.log_failure(
                category=FailureCategory.HALLUCINATION,
                agent=agent,
                description=f"Suspiciously high budget: ${total:,.0f}",
                run_id=run_id,
                severity="medium",
                recoverable=True,
                metadata={"budget_value": total},
            ))
        
        # Check for made-up competitor data
        competitors = doc.get("competitors", [])
        for comp in competitors:
            if isinstance(comp, dict):
                strength = (comp.get("strength") or "").lower()
                # Check for specific percentage claims without citation
                percentages = re.findall(r'(\d+(?:\.\d+)?)\s*%', strength)
                for pct in percentages:
                    pct_val = float(pct)
                    if pct_val > 90:
                        failures.append(self.log_failure(
                            category=FailureCategory.HALLUCINATION,
                            agent=agent,
                            description=f"Possibly hallucinated statistic: {pct}% in competitor analysis",
                            run_id=run_id,
                            severity="low",
                            recoverable=True,
                        ))
        
        return failures
    
    def _check_truncation(self, output: str, agent: str, run_id: str) -> Optional[Dict]:
        """Check for output truncation."""
        # Signs of truncation
        truncation_indicators = [
            output.rstrip().endswith("..."),
            output.count("{") > output.count("}"),
            output.count("[") > output.count("]"),
            len(output) > 10000 and not output.rstrip().endswith("}"),
        ]
        
        if any(truncation_indicators):
            return self.log_failure(
                category=FailureCategory.TRUNCATION,
                agent=agent,
                description="Output appears to be truncated",
                raw_output=output[-200:],  # Last 200 chars
                run_id=run_id,
                severity="high",
                recoverable=False,
            )
        
        return None
    
    def _check_budget_consistency(self, doc: dict, agent: str, run_id: str) -> Optional[Dict]:
        """Check budget total matches item sum."""
        budget = doc.get("budget", {})
        
        if not isinstance(budget, dict):
            return None
        
        total = budget.get("total_usd")
        items = budget.get("items", [])
        
        if total is None or not items:
            return None
        
        if not all(isinstance(item.get("usd"), (int, float)) for item in items):
            return self.log_failure(
                category=FailureCategory.SCHEMA_VIOLATION,
                agent=agent,
                description="Budget items missing 'usd' field or not numeric",
                run_id=run_id,
                severity="medium",
                recoverable=True,
            )
        
        item_sum = sum(item.get("usd", 0) for item in items)
        
        if abs(item_sum - total) > 1:
            return self.log_failure(
                category=FailureCategory.BUDGET_MISMATCH,
                agent=agent,
                description=f"Budget mismatch: total=${total:,.0f}, items sum=${item_sum:,.0f}",
                expected=f"total_usd should equal sum of items ({item_sum:,.0f})",
                run_id=run_id,
                severity="medium",
                recoverable=True,
                metadata={"total": total, "item_sum": item_sum, "difference": abs(total - item_sum)},
            )
        
        return None
    
    def _save_failures(self):
        """Save failures to log file."""
        with open(self.failure_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.failures, f, indent=2, ensure_ascii=False)
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get comprehensive failure statistics."""
        if not self.failures:
            return {"total_failures": 0, "message": "No failures logged"}
        
        stats = {
            "total_failures": len(self.failures),
            "by_category": defaultdict(int),
            "by_agent": defaultdict(int),
            "by_severity": defaultdict(int),
            "recovery_rate": 0,
            "category_details": {},
            "agent_details": {},
        }
        
        recoverable_count = 0
        
        for failure in self.failures:
            stats["by_category"][failure.get("category", "unknown")] += 1
            stats["by_agent"][failure.get("agent", "unknown")] += 1
            stats["by_severity"][failure.get("severity", "unknown")] += 1
            if failure.get("recoverable", False):
                recoverable_count += 1
        
        stats["recovery_rate"] = round(recoverable_count / len(self.failures) * 100, 1)
        
        # Convert defaultdicts to regular dicts
        stats["by_category"] = dict(stats["by_category"])
        stats["by_agent"] = dict(stats["by_agent"])
        stats["by_severity"] = dict(stats["by_severity"])
        
        # Category details
        for category in stats["by_category"]:
            cat_failures = [f for f in self.failures if f.get("category") == category]
            stats["category_details"][category] = {
                "count": len(cat_failures),
                "percent": round(len(cat_failures) / len(self.failures) * 100, 1),
                "agents_affected": list(set(f.get("agent") for f in cat_failures)),
                "example": cat_failures[0].get("description") if cat_failures else None,
            }
        
        # Agent details
        for agent in stats["by_agent"]:
            agent_failures = [f for f in self.failures if f.get("agent") == agent]
            stats["agent_details"][agent] = {
                "count": len(agent_failures),
                "percent": round(len(agent_failures) / len(self.failures) * 100, 1),
                "categories": list(set(f.get("category") for f in agent_failures)),
                "high_severity_count": sum(1 for f in agent_failures if f.get("severity") == "high"),
            }
        
        return stats
    
    def generate_failure_report(self) -> str:
        """Generate a markdown failure analysis report."""
        stats = self.get_failure_statistics()
        
        if stats.get("total_failures", 0) == 0:
            return "# Failure Analysis Report\n\nNo failures logged."
        
        report = f"""# Failure Analysis Report

**Generated:** {datetime.now().isoformat()}
**Total Failures Logged:** {stats['total_failures']}
**Recovery Rate:** {stats['recovery_rate']}%

## Summary by Category

| Category | Count | Percent | Agents Affected |
|----------|-------|---------|-----------------|
"""
        
        for category, details in stats.get("category_details", {}).items():
            agents = ", ".join(details.get("agents_affected", []))
            report += f"| {category} | {details['count']} | {details['percent']}% | {agents} |\n"
        
        report += f"""
## Summary by Agent

| Agent | Failures | Percent | High Severity |
|-------|----------|---------|---------------|
"""
        
        for agent, details in stats.get("agent_details", {}).items():
            report += f"| {agent} | {details['count']} | {details['percent']}% | {details['high_severity_count']} |\n"
        
        report += f"""
## Severity Distribution

| Severity | Count |
|----------|-------|
"""
        
        for severity, count in stats.get("by_severity", {}).items():
            report += f"| {severity} | {count} |\n"
        
        report += f"""
## Recent Failures (Last 10)

"""
        
        for failure in self.failures[-10:]:
            report += f"""### {failure.get('id')}: {failure.get('category')}
- **Agent:** {failure.get('agent')}
- **Severity:** {failure.get('severity')}
- **Description:** {failure.get('description')}
- **Recoverable:** {failure.get('recoverable')}

"""
        
        # Root cause analysis
        report += """## Root Cause Analysis

"""
        
        # Analyze patterns
        if stats["by_category"].get(FailureCategory.JSON_PARSE_ERROR, 0) > 0:
            report += """### JSON Parse Errors
Likely causes:
- Agent output includes markdown fences despite instructions
- Output truncation due to token limits
- Malformed JSON structure

**Recommendation:** Reinforce JSON-only output instructions, increase token limits

"""
        
        if stats["by_category"].get(FailureCategory.GROUNDING_FAILURE, 0) > 0:
            report += """### Grounding Failures
Likely causes:
- Agents not following grounding instructions
- Industry/audience mentions stripped during creative enhancement
- Context lost in sequential processing

**Recommendation:** Add explicit grounding validation after each agent

"""
        
        if stats["by_category"].get(FailureCategory.BUDGET_MISMATCH, 0) > 0:
            report += """### Budget Mismatches
Likely causes:
- Agents adding tactics without updating budget total
- Arithmetic errors in budget calculation
- Copy-paste errors from previous output

**Recommendation:** Add budget validation step before finalizing

"""
        
        # Save report
        report_file = self.output_dir / "failure_analysis_report.md"
        report_file.write_text(report, encoding='utf-8')
        print(f"[SAVED] Failure analysis report: {report_file}")
        
        return report
    
    def generate_visualizations(self):
        """Generate failure analysis visualizations."""
        stats = self.get_failure_statistics()
        
        if stats.get("total_failures", 0) == 0:
            print("[WARN] No failures to visualize")
            return
        
        # 1. Failures by category
        self._plot_failures_by_category(stats)
        
        # 2. Failures by agent
        self._plot_failures_by_agent(stats)
        
        # 3. Severity distribution
        self._plot_severity_distribution(stats)
    
    def _plot_failures_by_category(self, stats: Dict):
        """Plot failures by category."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = list(stats["by_category"].keys())
        counts = list(stats["by_category"].values())
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(categories)))
        
        bars = ax.barh(categories, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel("Count", fontsize=12, fontweight='bold')
        ax.set_ylabel("Failure Category", fontsize=12, fontweight='bold')
        ax.set_title("Failures by Category", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                   str(count), va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "failures_by_category.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] Created failures_by_category.png")
    
    def _plot_failures_by_agent(self, stats: Dict):
        """Plot failures by agent."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        agents = list(stats["by_agent"].keys())
        counts = list(stats["by_agent"].values())
        
        colors = ['#2E86AB', '#F18F01', '#6A994E'][:len(agents)]
        
        bars = ax.bar(agents, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel("Agent", fontsize=12, fontweight='bold')
        ax.set_ylabel("Failure Count", fontsize=12, fontweight='bold')
        ax.set_title("Failures by Agent", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   str(count), ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "failures_by_agent.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] Created failures_by_agent.png")
    
    def _plot_severity_distribution(self, stats: Dict):
        """Plot severity distribution."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        severities = list(stats["by_severity"].keys())
        counts = list(stats["by_severity"].values())
        
        colors = {'low': '#6A994E', 'medium': '#F18F01', 'high': '#E63946', 'critical': '#8B0000'}
        pie_colors = [colors.get(s, '#999999') for s in severities]
        
        wedges, texts, autotexts = ax.pie(counts, labels=severities, autopct='%1.1f%%',
                                          colors=pie_colors, startangle=90)
        
        ax.set_title("Failure Severity Distribution", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "failure_severity_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] Created failure_severity_distribution.png")


def analyze_failures(output_dir: str = "output"):
    """Convenience function to run failure analysis."""
    analyzer = FailureAnalyzer(output_dir=output_dir)
    
    # Generate report and visualizations
    report = analyzer.generate_failure_report()
    analyzer.generate_visualizations()
    
    return analyzer.get_failure_statistics()


if __name__ == "__main__":
    analyze_failures()