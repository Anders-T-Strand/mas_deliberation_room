#!/usr/bin/env python
"""
Enhanced Main Entry Point for AI Marketing Strategy System
Includes: Multi-run statistical validation, ablation studies, pipeline ordering experiments
"""

import sys
import warnings
import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import time

# Check for scipy (required for statistical tests)
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARNING] scipy not installed. Install with: pip install scipy")
    print("[WARNING] Statistical experiments (comparison, ablation, ordering) will be skipped.")

from mas_deliberation_room.evaluation import EvaluationHarness
from mas_deliberation_room.visualization_generator import VisualizationGenerator
from mas_deliberation_room.crew import MasDeliberationRoom

# Create output directory
os.makedirs('output', exist_ok=True)

# Default dataset/strategy locations
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY_FILE = PROJECT_ROOT / "datasets" / "ecommerce_fashion_strategy.txt"
DEFAULT_DATA_FILE = PROJECT_ROOT / "datasets" / "ecommerce_fashion.csv"
AGENT_MODE_RESULTS_FILE = PROJECT_ROOT / "output" / "agent_mode_results.json"
MULTI_RUN_RESULTS_FILE = PROJECT_ROOT / "output" / "multi_run_statistics.json"
ABLATION_RESULTS_FILE = PROJECT_ROOT / "output" / "ablation_results.json"

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Agent orderings for pipeline experiments
AGENT_ORDERINGS = {
    "default": ["openai", "claude", "gemini"],      # GPT → Claude → Gemini
    "reverse": ["gemini", "claude", "openai"],      # Gemini → Claude → GPT
    "claude_first": ["claude", "openai", "gemini"], # Claude → GPT → Gemini
    "claude_last": ["openai", "gemini", "claude"],  # GPT → Gemini → Claude
    "gemini_first": ["gemini", "openai", "claude"], # Gemini → GPT → Claude
    "openai_last": ["claude", "gemini", "openai"],  # Claude → Gemini → GPT
}

# Ablation configurations
ABLATION_CONFIGS = {
    "full_pipeline": {"agents": ["openai", "claude", "gemini"], "description": "Full 3-agent pipeline"},
    "two_agent_no_claude": {"agents": ["openai", "gemini"], "description": "GPT → Gemini (no Claude)"},
    "two_agent_no_gemini": {"agents": ["openai", "claude"], "description": "GPT → Claude (no Gemini)"},
    "homogeneous_gpt": {"agents": ["openai", "openai", "openai"], "description": "GPT → GPT → GPT"},
    "single_agent": {"agents": ["openai"], "description": "Single GPT agent"},
}

# Token limits by agent role
TOKEN_LIMITS = {
    "Data-Driven Marketing Strategist": 4000,
    "Creative Marketing Architect": 4096,
    "Marketing Operations Director": 8000,
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _resolve_with_project_root(file_path: Path, default_path: Path) -> Path:
    """Try multiple resolutions for file paths."""
    candidates = [
        file_path,
        PROJECT_ROOT / file_path,
        PROJECT_ROOT / "datasets" / file_path.name,
        default_path,
    ]
    for candidate in candidates:
        try:
            candidate_resolved = candidate.expanduser().resolve()
        except Exception:
            candidate_resolved = candidate
        if candidate_resolved.exists():
            return candidate_resolved
    return default_path


def read_strategy_file(file_path=DEFAULT_STRATEGY_FILE):
    """Read strategy file (txt, md, or docx)."""
    file_path = _resolve_with_project_root(Path(file_path), DEFAULT_STRATEGY_FILE)
    
    if file_path.suffix.lower() in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_path.suffix.lower() == '.docx':
        try:
            import subprocess
            result = subprocess.run(
                ['pandoc', str(file_path), '-t', 'plain'],
                capture_output=True, text=True, check=True
            )
            return result.stdout
        except subprocess.CalledProcessError:
            raise Exception("Failed to read .docx file. Ensure pandoc is installed.")
        except FileNotFoundError:
            raise Exception("Pandoc not found. Install with: sudo apt-get install pandoc")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def read_csv_data(file_path=DEFAULT_DATA_FILE):
    """Read and summarize CSV data."""
    file_path = _resolve_with_project_root(Path(file_path), DEFAULT_DATA_FILE)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    summary = {
        'file_name': file_path.name,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'date_range': None,
        'summary_stats': {}
    }
    
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    if date_columns:
        date_col = date_columns[0]
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            summary['date_range'] = f"{df[date_col].min()} to {df[date_col].max()}"
        except:
            pass
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols[:5]:
        summary['summary_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return {'dataframe': df, 'summary': summary}


def format_data_summary(data_info):
    """Format data summary for AI agent input."""
    summary = data_info['summary']
    
    output = f"Data File: {summary['file_name']}\n"
    output += f"Total Records: {summary['total_rows']}\n"
    output += f"Metrics Tracked: {summary['total_columns']} columns\n"
    
    if summary['date_range']:
        output += f"Date Range: {summary['date_range']}\n"
    
    output += f"\nAvailable Metrics:\n"
    for col in summary['columns'][:10]:
        output += f"  - {col}\n"
    
    if summary['summary_stats']:
        output += f"\nKey Statistics:\n"
        for col, stats in list(summary['summary_stats'].items())[:3]:
            output += f"  {col}:\n"
            output += f"    Mean: {stats['mean']:.2f}\n"
            output += f"    Range: {stats['min']:.2f} - {stats['max']:.2f}\n"
    
    return output


def record_agent_mode_result(agent_mode: str, eval_metrics: dict, execution_time: float,
                             run_id: int = None, seed: int = None, 
                             ablation: str = None, ordering: str = None):
    """Persist results with extended metadata for multi-run experiments."""
    result_entry = {
        "agent_mode": agent_mode,
        "timestamp": datetime.now().isoformat(),
        "execution_time_seconds": execution_time,
        "rubric_percent": eval_metrics.get("rubric", {}).get("percent"),
        "rubric_score": eval_metrics.get("rubric", {}).get("score_0_to_10"),
        "schema_valid": eval_metrics.get("schema", {}).get("valid"),
        "schema_issues": eval_metrics.get("schema", {}).get("issues", []),
        "refinement_delta": eval_metrics.get("refinement_delta"),
        "grounding_score": eval_metrics.get("grounding_score"),
        # Extended metadata for statistical analysis
        "run_id": run_id,
        "seed": seed,
        "ablation_config": ablation,
        "agent_ordering": ordering,
        "rubric_checks": eval_metrics.get("rubric", {}).get("checks", {}),
    }

    existing = []
    if AGENT_MODE_RESULTS_FILE.exists():
        try:
            existing = json.loads(AGENT_MODE_RESULTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    existing.append(result_entry)
    AGENT_MODE_RESULTS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    
    return result_entry


def apply_token_limits(crew):
    """Apply graduated token limits to crew agents."""
    for agent in crew.agents:
        agent_role = (agent.role.strip() if hasattr(agent, 'role') else "Unknown")
        limit = TOKEN_LIMITS.get(agent_role, 4000)
        setattr(agent, "max_tokens", limit)
        if hasattr(agent, "llm") and hasattr(agent.llm, "max_tokens"):
            agent.llm.max_tokens = limit
        print(f"[Token limit] {agent_role}: {limit} tokens")


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def calculate_statistics(results: List[Dict]) -> Dict[str, Any]:
    """Calculate comprehensive statistics from multiple runs."""
    if not results:
        return {}
    
    # Extract metrics
    rubric_scores = [r["rubric_percent"] for r in results if r.get("rubric_percent") is not None]
    exec_times = [r["execution_time_seconds"] for r in results if r.get("execution_time_seconds") is not None]
    
    stats = {}
    
    if rubric_scores:
        stats["rubric"] = {
            "mean": float(np.mean(rubric_scores)),
            "std": float(np.std(rubric_scores, ddof=1)) if len(rubric_scores) > 1 else 0.0,
            "min": float(np.min(rubric_scores)),
            "max": float(np.max(rubric_scores)),
            "median": float(np.median(rubric_scores)),
            "n": len(rubric_scores),
        }
        # 95% confidence interval
        if len(rubric_scores) > 1 and SCIPY_AVAILABLE:
            sem = scipy_stats.sem(rubric_scores)
            ci = scipy_stats.t.interval(0.95, len(rubric_scores)-1, 
                                        loc=np.mean(rubric_scores), scale=sem)
            stats["rubric"]["ci_95_lower"] = float(ci[0])
            stats["rubric"]["ci_95_upper"] = float(ci[1])
    
    if exec_times:
        stats["execution_time"] = {
            "mean": float(np.mean(exec_times)),
            "std": float(np.std(exec_times, ddof=1)) if len(exec_times) > 1 else 0.0,
            "min": float(np.min(exec_times)),
            "max": float(np.max(exec_times)),
            "median": float(np.median(exec_times)),
            "n": len(exec_times),
        }
    
    # Aggregate rubric check pass rates
    check_counts = {}
    for r in results:
        checks = r.get("rubric_checks", {})
        for check, passed in checks.items():
            if check not in check_counts:
                check_counts[check] = {"passed": 0, "total": 0}
            check_counts[check]["total"] += 1
            if passed:
                check_counts[check]["passed"] += 1
    
    stats["rubric_check_rates"] = {
        check: round(counts["passed"] / counts["total"] * 100, 1) 
        for check, counts in check_counts.items() if counts["total"] > 0
    }
    
    return stats


def compare_configurations(results_a: List[Dict], results_b: List[Dict], 
                          name_a: str = "A", name_b: str = "B") -> Dict[str, Any]:
    """Compare two configurations with statistical significance testing."""
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for statistical comparisons. Install with: pip install scipy")
    
    scores_a = [r["rubric_percent"] for r in results_a if r.get("rubric_percent") is not None]
    scores_b = [r["rubric_percent"] for r in results_b if r.get("rubric_percent") is not None]
    
    comparison = {
        "config_a": name_a,
        "config_b": name_b,
        "n_a": len(scores_a),
        "n_b": len(scores_b),
    }
    
    if len(scores_a) >= 2 and len(scores_b) >= 2:
        # Independent samples t-test
        t_stat, p_value = scipy_stats.ttest_ind(scores_a, scores_b)
        comparison["t_test"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_05": p_value < 0.05,
            "significant_at_01": p_value < 0.01,
        }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(scores_a)-1)*np.var(scores_a, ddof=1) + 
                              (len(scores_b)-1)*np.var(scores_b, ddof=1)) / 
                             (len(scores_a) + len(scores_b) - 2))
        cohens_d = (np.mean(scores_b) - np.mean(scores_a)) / pooled_std if pooled_std > 0 else 0
        comparison["effect_size"] = {
            "cohens_d": float(cohens_d),
            "interpretation": "large" if abs(cohens_d) >= 0.8 else "medium" if abs(cohens_d) >= 0.5 else "small"
        }
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat, u_pvalue = scipy_stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
        comparison["mann_whitney"] = {
            "u_statistic": float(u_stat),
            "p_value": float(u_pvalue),
        }
    
    comparison["means"] = {
        name_a: float(np.mean(scores_a)) if scores_a else None,
        name_b: float(np.mean(scores_b)) if scores_b else None,
    }
    comparison["difference"] = {
        "absolute": float(np.mean(scores_b) - np.mean(scores_a)) if scores_a and scores_b else None,
        "percent": float((np.mean(scores_b) - np.mean(scores_a)) / np.mean(scores_a) * 100) if scores_a and np.mean(scores_a) > 0 else None,
    }
    
    return comparison


# =============================================================================
# SINGLE RUN EXECUTION
# =============================================================================

def execute_single_run(inputs: Dict, mode: str = "multi", 
                       ablation: str = None, ordering: str = None,
                       run_id: int = None, seed: int = None,
                       save_individual_report: bool = False) -> Dict:
    """
    Execute a single run with specified configuration.
    
    Args:
        save_individual_report: If True, save individual JSON/MD reports per run.
                               Default False for multi-run experiments to avoid file clutter.
    """
    
    # Set random seed if provided
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        os.environ["RANDOM_SEED"] = str(seed)
    
    crew_builder = MasDeliberationRoom()
    
    # Select crew configuration
    if mode == "single":
        crew = crew_builder.single_agent_crew()
    elif ablation and ablation in ABLATION_CONFIGS:
        crew = crew_builder.crew(agent_mode="multi", ablation=ablation)
    elif ordering and ordering in AGENT_ORDERINGS:
        crew = crew_builder.crew(agent_mode="multi", ordering=ordering)
    else:
        crew = crew_builder.crew(agent_mode="multi")
    
    apply_token_limits(crew)
    
    # Execute
    t0 = time.time()
    result = crew.kickoff(inputs=inputs)
    exec_time = time.time() - t0
    
    # Evaluate
    harness = EvaluationHarness()
    final_output_path = Path("output/openai_strategy.json") if mode == "single" else Path("output/final_strategy.json")
    openai_output_path = None if mode == "single" else Path("output/openai_strategy.json")
    
    eval_metrics = harness.run_evaluation(
        crew_result=result,
        user_inputs=inputs,
        execution_time=exec_time,
        test_case_name=f"{mode}_run_{run_id or 0}",
        final_output_path=final_output_path,
        openai_output_path=openai_output_path,
        save_individual_report=save_individual_report,
    )
    
    # Record result
    result_entry = record_agent_mode_result(
        agent_mode=mode,
        eval_metrics=eval_metrics,
        execution_time=exec_time,
        run_id=run_id,
        seed=seed,
        ablation=ablation,
        ordering=ordering,
    )
    
    return result_entry


# =============================================================================
# MULTI-RUN STATISTICAL VALIDATION
# =============================================================================

def run_multiple_trials(n_runs: int = 10, mode: str = "multi",
                        strategy_file=None, data_file=None,
                        industry: str = "E-commerce", 
                        target_audience: str = "Women 25-45",
                        base_seed: int = 42) -> Dict[str, Any]:
    """
    Run n trials with different seeds and collect comprehensive statistics.
    
    This addresses Feedback Item #1: "Only a single run per configuration (major flaw)"
    """
    print("\n" + "="*70)
    print(f"MULTI-RUN STATISTICAL VALIDATION: {n_runs} trials in {mode.upper()} mode")
    print("="*70 + "\n")
    
    # Load data
    strategy_file = _resolve_with_project_root(
        Path(strategy_file) if strategy_file else DEFAULT_STRATEGY_FILE, 
        DEFAULT_STRATEGY_FILE
    )
    data_file = _resolve_with_project_root(
        Path(data_file) if data_file else DEFAULT_DATA_FILE,
        DEFAULT_DATA_FILE
    )
    
    strategy_content = read_strategy_file(strategy_file)
    data_info = read_csv_data(data_file)
    sales_data_summary = format_data_summary(data_info)
    
    inputs = {
        'module_name': 'marketing_strategy_report.md',
        'current_strategy': strategy_content,
        'industry': industry,
        'target_audience': target_audience,
        'sales_data': sales_data_summary
    }
    
    results = []
    for i in range(n_runs):
        seed = base_seed + i
        print(f"\n--- Trial {i+1}/{n_runs} (seed={seed}) ---")
        
        try:
            result = execute_single_run(
                inputs=inputs,
                mode=mode,
                run_id=i+1,
                seed=seed,
                save_individual_report=False  # Don't save per-run reports in multi-run mode
            )
            results.append(result)
            print(f"  Rubric: {result['rubric_percent']:.1f}%, Time: {result['execution_time_seconds']:.2f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"run_id": i+1, "seed": seed, "error": str(e)})
    
    # Calculate statistics
    valid_results = [r for r in results if "error" not in r]
    stats = calculate_statistics(valid_results)
    
    # Save multi-run results
    multi_run_output = {
        "mode": mode,
        "n_runs": n_runs,
        "n_successful": len(valid_results),
        "base_seed": base_seed,
        "timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "individual_runs": results,
    }
    
    MULTI_RUN_RESULTS_FILE.write_text(json.dumps(multi_run_output, indent=2), encoding="utf-8")
    
    # Print summary
    print("\n" + "="*70)
    print("MULTI-RUN STATISTICS SUMMARY")
    print("="*70)
    if "rubric" in stats:
        r = stats["rubric"]
        print(f"\nRubric Score:")
        print(f"  Mean ± Std: {r['mean']:.2f} ± {r['std']:.2f}")
        print(f"  Range: [{r['min']:.1f}, {r['max']:.1f}]")
        if "ci_95_lower" in r:
            print(f"  95% CI: [{r['ci_95_lower']:.2f}, {r['ci_95_upper']:.2f}]")
    
    if "execution_time" in stats:
        t = stats["execution_time"]
        print(f"\nExecution Time (seconds):")
        print(f"  Mean ± Std: {t['mean']:.2f} ± {t['std']:.2f}")
    
    print(f"\nResults saved to: {MULTI_RUN_RESULTS_FILE}")
    
    return multi_run_output


def run_comparison_experiment(n_runs: int = 10, base_seed: int = 42,
                              strategy_file=None, data_file=None,
                              industry: str = "E-commerce",
                              target_audience: str = "Women 25-45") -> Dict[str, Any]:
    """
    Run statistical comparison between single and multi-agent configurations.
    """
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON EXPERIMENT: Single vs Multi-Agent")
    print("="*70 + "\n")
    
    # Run single-agent trials
    print("\n=== Phase 1: Single-Agent Trials ===")
    single_results = run_multiple_trials(
        n_runs=n_runs, mode="single",
        strategy_file=strategy_file, data_file=data_file,
        industry=industry, target_audience=target_audience,
        base_seed=base_seed
    )
    
    # Run multi-agent trials
    print("\n=== Phase 2: Multi-Agent Trials ===")
    multi_results = run_multiple_trials(
        n_runs=n_runs, mode="multi",
        strategy_file=strategy_file, data_file=data_file,
        industry=industry, target_audience=target_audience,
        base_seed=base_seed + 100  # Offset seeds for multi
    )
    
    # Statistical comparison
    comparison = compare_configurations(
        single_results["individual_runs"],
        multi_results["individual_runs"],
        name_a="single",
        name_b="multi"
    )
    
    # Output
    output = {
        "experiment": "single_vs_multi_comparison",
        "n_runs_each": n_runs,
        "timestamp": datetime.now().isoformat(),
        "single_agent": single_results["statistics"],
        "multi_agent": multi_results["statistics"],
        "comparison": comparison,
    }
    
    comparison_file = PROJECT_ROOT / "output" / "statistical_comparison.json"
    comparison_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    
    # Print comparison summary
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON RESULTS")
    print("="*70)
    print(f"\nSingle-Agent Mean: {comparison['means']['single']:.2f}%")
    print(f"Multi-Agent Mean:  {comparison['means']['multi']:.2f}%")
    print(f"Difference: {comparison['difference']['absolute']:+.2f}% ({comparison['difference']['percent']:+.1f}% relative)")
    
    if "t_test" in comparison:
        t = comparison["t_test"]
        print(f"\nT-Test:")
        print(f"  t-statistic: {t['t_statistic']:.3f}")
        print(f"  p-value: {t['p_value']:.4f}")
        print(f"  Significant at α=0.05: {'YES' if t['significant_at_05'] else 'NO'}")
    
    if "effect_size" in comparison:
        e = comparison["effect_size"]
        print(f"\nEffect Size:")
        print(f"  Cohen's d: {e['cohens_d']:.3f} ({e['interpretation']})")
    
    print(f"\nFull results saved to: {comparison_file}")
    
    return output


# =============================================================================
# ABLATION STUDIES
# =============================================================================

def run_ablation_study(n_runs: int = 5, configs: List[str] = None,
                       strategy_file=None, data_file=None,
                       industry: str = "E-commerce",
                       target_audience: str = "Women 25-45",
                       base_seed: int = 42) -> Dict[str, Any]:
    """
    Run ablation studies comparing different agent configurations.
    
    This addresses Feedback Item #2: "No ablation studies"
    """
    if configs is None:
        configs = list(ABLATION_CONFIGS.keys())
    
    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)
    print(f"Configurations: {configs}")
    print(f"Runs per config: {n_runs}")
    print("="*70 + "\n")
    
    # Load data
    strategy_file = _resolve_with_project_root(
        Path(strategy_file) if strategy_file else DEFAULT_STRATEGY_FILE,
        DEFAULT_STRATEGY_FILE
    )
    data_file = _resolve_with_project_root(
        Path(data_file) if data_file else DEFAULT_DATA_FILE,
        DEFAULT_DATA_FILE
    )
    
    strategy_content = read_strategy_file(strategy_file)
    data_info = read_csv_data(data_file)
    sales_data_summary = format_data_summary(data_info)
    
    inputs = {
        'module_name': 'marketing_strategy_report.md',
        'current_strategy': strategy_content,
        'industry': industry,
        'target_audience': target_audience,
        'sales_data': sales_data_summary
    }
    
    all_results = {}
    
    for config_name in configs:
        if config_name not in ABLATION_CONFIGS:
            print(f"[WARN] Unknown config: {config_name}, skipping")
            continue
            
        config = ABLATION_CONFIGS[config_name]
        print(f"\n=== Ablation: {config_name} ({config['description']}) ===")
        
        results = []
        for i in range(n_runs):
            seed = base_seed + i
            print(f"  Trial {i+1}/{n_runs} (seed={seed})...", end=" ")
            
            try:
                mode = "single" if len(config["agents"]) == 1 else "multi"
                result = execute_single_run(
                    inputs=inputs,
                    mode=mode,
                    ablation=config_name,
                    run_id=i+1,
                    seed=seed,
                    save_individual_report=False  # Don't save per-run reports
                )
                results.append(result)
                print(f"Rubric: {result['rubric_percent']:.1f}%")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"run_id": i+1, "error": str(e)})
        
        valid_results = [r for r in results if "error" not in r]
        all_results[config_name] = {
            "description": config["description"],
            "agents": config["agents"],
            "n_runs": n_runs,
            "n_successful": len(valid_results),
            "statistics": calculate_statistics(valid_results),
            "individual_runs": results,
        }
    
    # Pairwise comparisons
    comparisons = []
    config_names = list(all_results.keys())
    for i, name_a in enumerate(config_names):
        for name_b in config_names[i+1:]:
            comp = compare_configurations(
                all_results[name_a]["individual_runs"],
                all_results[name_b]["individual_runs"],
                name_a=name_a,
                name_b=name_b
            )
            comparisons.append(comp)
    
    # Save results
    output = {
        "experiment": "ablation_study",
        "timestamp": datetime.now().isoformat(),
        "n_runs_per_config": n_runs,
        "results_by_config": all_results,
        "pairwise_comparisons": comparisons,
    }
    
    ABLATION_RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    
    # Print summary table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Configuration':<25} {'Mean':<10} {'Std':<10} {'N':<5}")
    print("-"*50)
    for name, data in all_results.items():
        stats = data["statistics"]
        if "rubric" in stats:
            r = stats["rubric"]
            print(f"{name:<25} {r['mean']:<10.2f} {r['std']:<10.2f} {r['n']:<5}")
    
    print(f"\nFull results saved to: {ABLATION_RESULTS_FILE}")
    
    return output


# =============================================================================
# PIPELINE ORDERING EXPERIMENTS
# =============================================================================

def run_ordering_experiment(n_runs: int = 5, orderings: List[str] = None,
                            strategy_file=None, data_file=None,
                            industry: str = "E-commerce",
                            target_audience: str = "Women 25-45",
                            base_seed: int = 42) -> Dict[str, Any]:
    """
    Test different agent orderings in the pipeline.
    
    This addresses Feedback Item #10: "Pipeline ordering effects unexplored"
    """
    if orderings is None:
        orderings = list(AGENT_ORDERINGS.keys())
    
    print("\n" + "="*70)
    print("PIPELINE ORDERING EXPERIMENT")
    print("="*70)
    print(f"Orderings to test: {orderings}")
    print(f"Runs per ordering: {n_runs}")
    print("="*70 + "\n")
    
    # Load data
    strategy_file = _resolve_with_project_root(
        Path(strategy_file) if strategy_file else DEFAULT_STRATEGY_FILE,
        DEFAULT_STRATEGY_FILE
    )
    data_file = _resolve_with_project_root(
        Path(data_file) if data_file else DEFAULT_DATA_FILE,
        DEFAULT_DATA_FILE
    )
    
    strategy_content = read_strategy_file(strategy_file)
    data_info = read_csv_data(data_file)
    sales_data_summary = format_data_summary(data_info)
    
    inputs = {
        'module_name': 'marketing_strategy_report.md',
        'current_strategy': strategy_content,
        'industry': industry,
        'target_audience': target_audience,
        'sales_data': sales_data_summary
    }
    
    all_results = {}
    
    for ordering_name in orderings:
        if ordering_name not in AGENT_ORDERINGS:
            print(f"[WARN] Unknown ordering: {ordering_name}, skipping")
            continue
        
        order = AGENT_ORDERINGS[ordering_name]
        print(f"\n=== Ordering: {ordering_name} ({' → '.join(order)}) ===")
        
        results = []
        for i in range(n_runs):
            seed = base_seed + i
            print(f"  Trial {i+1}/{n_runs} (seed={seed})...", end=" ")
            
            try:
                result = execute_single_run(
                    inputs=inputs,
                    mode="multi",
                    ordering=ordering_name,
                    run_id=i+1,
                    seed=seed,
                    save_individual_report=False  # Don't save per-run reports
                )
                results.append(result)
                print(f"Rubric: {result['rubric_percent']:.1f}%")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"run_id": i+1, "error": str(e)})
        
        valid_results = [r for r in results if "error" not in r]
        all_results[ordering_name] = {
            "agent_order": order,
            "n_runs": n_runs,
            "n_successful": len(valid_results),
            "statistics": calculate_statistics(valid_results),
            "individual_runs": results,
        }
    
    # Save results
    ordering_file = PROJECT_ROOT / "output" / "ordering_experiment.json"
    output = {
        "experiment": "pipeline_ordering",
        "timestamp": datetime.now().isoformat(),
        "n_runs_per_ordering": n_runs,
        "results_by_ordering": all_results,
    }
    ordering_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    
    # Print summary
    print("\n" + "="*70)
    print("ORDERING EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Ordering':<20} {'Agents':<30} {'Mean':<10} {'Std':<10}")
    print("-"*70)
    for name, data in all_results.items():
        stats = data["statistics"]
        agents_str = " → ".join(data["agent_order"])
        if "rubric" in stats:
            r = stats["rubric"]
            print(f"{name:<20} {agents_str:<30} {r['mean']:<10.2f} {r['std']:<10.2f}")
    
    print(f"\nFull results saved to: {ordering_file}")
    
    return output


# =============================================================================
# COMPREHENSIVE EXPERIMENT SUITE
# =============================================================================

def run_full_experiment_suite(strategy_file=None, data_file=None,
                              industry: str = "E-commerce",
                              target_audience: str = "Women 25-45",
                              n_runs: int = 10,
                              base_seed: int = 42):
    """
    Run the complete experiment suite including all analyses.
    
    This is the main entry point when running `crewai run`.
    
    Executes in order:
    1. Dataset validation
    2. Single vs Multi-agent comparison (statistical)
    3. Ablation studies
    4. Pipeline ordering experiments
    5. LLM-as-judge evaluation
    6. Failure analysis
    7. Generate all visualizations
    """
    import pandas as pd
    import sys
    
    print("\n" + "="*80)
    print("   COMPREHENSIVE EXPERIMENT SUITE")
    print("   AI Marketing Strategy Multi-Agent System Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Industry: {industry}")
    print(f"  - Target Audience: {target_audience}")
    print(f"  - Runs per experiment: {n_runs}")
    print(f"  - Base seed: {base_seed}")
    
    # Check for scipy
    if not SCIPY_AVAILABLE:
        print("\n" + "!"*70)
        print("WARNING: scipy is NOT installed!")
        print("Statistical experiments (comparison, ablation, ordering) will be SKIPPED.")
        print("To enable full statistical analysis, run: pip install scipy")
        print("!"*70)
    
    print("="*80 + "\n")
    
    start_time = time.time()
    results_summary = {}
    
    # Resolve file paths
    strat_file = _resolve_with_project_root(
        Path(strategy_file) if strategy_file else DEFAULT_STRATEGY_FILE,
        DEFAULT_STRATEGY_FILE
    )
    dat_file = _resolve_with_project_root(
        Path(data_file) if data_file else DEFAULT_DATA_FILE,
        DEFAULT_DATA_FILE
    )
    
    # Add current directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent))
    
    # ==========================================================================
    # PHASE 1: Dataset Validation
    # ==========================================================================
    print("\n" + "="*70)
    print("PHASE 1: DATASET VALIDATION")
    print("="*70 + "\n")
    
    try:
        from dataset_validation import DatasetValidator
        
        df = pd.read_csv(dat_file)
        validator = DatasetValidator(output_dir="output")
        validation_results = validator.validate_dataset(
            df, industry="fashion", dataset_name="ecommerce_fashion"
        )
        validator.generate_validation_report("ecommerce_fashion")
        
        results_summary["dataset_validation"] = {
            "status": "completed",
            "realism_score": validation_results.get("realism_score", {}).get("overall"),
        }
        print(f"\n[OK] Dataset validation complete. Realism score: {validation_results.get('realism_score', {}).get('overall', 'N/A')}%")
    except ImportError as e:
        print(f"[WARN] Skipping dataset validation (module not found): {e}")
        results_summary["dataset_validation"] = {"status": "skipped", "reason": str(e)}
    except Exception as e:
        print(f"[ERROR] Dataset validation error: {e}")
        results_summary["dataset_validation"] = {"status": "error", "error": str(e)}
    
    # ==========================================================================
    # PHASE 2: Single vs Multi-Agent Comparison
    # ==========================================================================
    print("\n" + "="*70)
    print("PHASE 2: SINGLE VS MULTI-AGENT COMPARISON")
    print("="*70 + "\n")
    
    if not SCIPY_AVAILABLE:
        print("[SKIP] Comparison experiment requires scipy. Install with: pip install scipy")
        results_summary["comparison"] = {"status": "skipped", "reason": "scipy not installed"}
    else:
        try:
            comparison_results = run_comparison_experiment(
                n_runs=n_runs,
                strategy_file=str(strat_file),
                data_file=str(dat_file),
                industry=industry,
                target_audience=target_audience,
                base_seed=base_seed
            )
            
            results_summary["comparison"] = {
                "status": "completed",
                "single_mean": comparison_results.get("single_agent", {}).get("rubric", {}).get("mean"),
                "multi_mean": comparison_results.get("multi_agent", {}).get("rubric", {}).get("mean"),
                "p_value": comparison_results.get("comparison", {}).get("t_test", {}).get("p_value"),
                "cohens_d": comparison_results.get("comparison", {}).get("effect_size", {}).get("cohens_d"),
            }
            print(f"\n[OK] Comparison experiment complete.")
        except Exception as e:
            print(f"[ERROR] Comparison experiment error: {e}")
            results_summary["comparison"] = {"status": "error", "error": str(e)}
    
    # ==========================================================================
    # PHASE 3: Ablation Studies
    # ==========================================================================
    print("\n" + "="*70)
    print("PHASE 3: ABLATION STUDIES")
    print("="*70 + "\n")
    
    if not SCIPY_AVAILABLE:
        print("[SKIP] Ablation studies require scipy. Install with: pip install scipy")
        results_summary["ablation"] = {"status": "skipped", "reason": "scipy not installed"}
    else:
        try:
            ablation_results = run_ablation_study(
                n_runs=n_runs,  # Full 10 runs for statistical validity
                strategy_file=str(strat_file),
                data_file=str(dat_file),
                industry=industry,
                target_audience=target_audience,
                base_seed=base_seed
            )
            
            results_summary["ablation"] = {
                "status": "completed",
                "configurations_tested": len(ablation_results.get("results_by_config", {})),
            }
            print(f"\n[OK] Ablation study complete.")
        except Exception as e:
            print(f"[ERROR] Ablation study error: {e}")
            results_summary["ablation"] = {"status": "error", "error": str(e)}
    
    # ==========================================================================
    # PHASE 4: Pipeline Ordering Experiments
    # ==========================================================================
    print("\n" + "="*70)
    print("PHASE 4: PIPELINE ORDERING EXPERIMENTS")
    print("="*70 + "\n")
    
    if not SCIPY_AVAILABLE:
        print("[SKIP] Ordering experiments require scipy. Install with: pip install scipy")
        results_summary["ordering"] = {"status": "skipped", "reason": "scipy not installed"}
    else:
        try:
            ordering_results = run_ordering_experiment(
                n_runs=n_runs,  # Full 10 runs for statistical validity
                strategy_file=str(strat_file),
                data_file=str(dat_file),
                industry=industry,
                target_audience=target_audience,
                base_seed=base_seed
            )
            
            results_summary["ordering"] = {
                "status": "completed",
                "orderings_tested": len(ordering_results.get("results_by_ordering", {})),
            }
            print(f"\n[OK] Ordering experiment complete.")
        except Exception as e:
            print(f"[ERROR] Ordering experiment error: {e}")
            results_summary["ordering"] = {"status": "error", "error": str(e)}
    
    # ==========================================================================
    # PHASE 5: LLM-as-Judge Evaluation
    # ==========================================================================
    print("\n" + "="*70)
    print("PHASE 5: LLM-AS-JUDGE EVALUATION")
    print("="*70 + "\n")
    
    try:
        from llm_judge import LLMJudge
        
        # Load the final strategy
        strategy_path = Path("output/final_strategy.json")
        if not strategy_path.exists():
            strategy_path = Path("output/openai_strategy.json")
        
        if strategy_path.exists():
            with open(strategy_path, 'r', encoding='utf-8') as f:
                content = f.read()
                try:
                    strategy_doc = json.loads(content)
                except:
                    strategy_doc = {"strategy_text": content}
            
            user_inputs = {
                "industry": industry,
                "target_audience": target_audience,
            }
            
            judge = LLMJudge(output_dir="output")
            judge_results = judge.evaluate_strategy(strategy_doc, user_inputs)
            judge.generate_evaluation_report()
            
            results_summary["llm_judge"] = {
                "status": "completed",
                "overall_score": judge_results.get("overall_score"),
                "scores": judge_results.get("scores"),
            }
            print(f"\n[OK] LLM-as-judge evaluation complete. Overall score: {judge_results.get('overall_score')}/10")
        else:
            print("[WARN] No strategy file found for LLM judge evaluation")
            results_summary["llm_judge"] = {"status": "skipped", "reason": "No strategy file"}
    except ImportError as e:
        print(f"[WARN] Skipping LLM-as-judge (module not found): {e}")
        results_summary["llm_judge"] = {"status": "skipped", "reason": str(e)}
    except Exception as e:
        print(f"[ERROR] LLM-as-judge error: {e}")
        results_summary["llm_judge"] = {"status": "error", "error": str(e)}
    
    # ==========================================================================
    # PHASE 6: Failure Analysis
    # ==========================================================================
    print("\n" + "="*70)
    print("PHASE 6: FAILURE ANALYSIS")
    print("="*70 + "\n")
    
    try:
        from failure_analysis import FailureAnalyzer
        
        analyzer = FailureAnalyzer(output_dir="output")
        failure_stats = analyzer.get_failure_statistics()
        analyzer.generate_failure_report()
        analyzer.generate_visualizations()
        
        results_summary["failure_analysis"] = {
            "status": "completed",
            "total_failures": failure_stats.get("total_failures", 0),
            "recovery_rate": failure_stats.get("recovery_rate"),
        }
        print(f"\n[OK] Failure analysis complete. Total failures logged: {failure_stats.get('total_failures', 0)}")
    except ImportError as e:
        print(f"[WARN] Skipping failure analysis (module not found): {e}")
        results_summary["failure_analysis"] = {"status": "skipped", "reason": str(e)}
    except Exception as e:
        print(f"[ERROR] Failure analysis error: {e}")
        results_summary["failure_analysis"] = {"status": "error", "error": str(e)}
    
    # ==========================================================================
    # PHASE 7: Generate All Visualizations
    # ==========================================================================
    print("\n" + "="*70)
    print("PHASE 7: GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    try:
        vg = VisualizationGenerator(output_dir="output")
        vg.generate_all_visualizations()
        
        results_summary["visualizations"] = {"status": "completed"}
        print(f"\n[OK] Visualization generation complete.")
    except Exception as e:
        print(f"[ERROR] Visualization error: {e}")
        results_summary["visualizations"] = {"status": "error", "error": str(e)}
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("   EXPERIMENT SUITE COMPLETE")
    print("="*80)
    print(f"\nTotal execution time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print("\nPhase Results:")
    print("-"*60)
    
    for phase, result in results_summary.items():
        status = result.get("status", "unknown")
        status_icon = "+" if status == "completed" else "!" if status == "skipped" else "x"
        print(f"  [{status_icon}] {phase}: {status}")
    
    # Save comprehensive results
    results_summary["total_time_seconds"] = total_time
    results_summary["timestamp"] = datetime.now().isoformat()
    results_summary["configuration"] = {
        "industry": industry,
        "target_audience": target_audience,
        "n_runs": n_runs,
        "base_seed": base_seed,
    }
    
    results_file = Path("output") / "experiment_suite_results.json"
    results_file.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    
    print(f"\nResults saved to: {results_file}")
    print("\nOutput files generated in: output/")
    print("  - Statistical comparisons: statistical_comparison.json")
    print("  - Ablation results: ablation_results.json")
    print("  - Ordering results: ordering_experiment.json")
    print("  - Dataset validation: dataset_validation_*.json")
    print("  - LLM judge: llm_judge_evaluations.json")
    print("  - Failure analysis: failure_analysis_report.md")
    print("  - Visualizations: *.png files")
    print("="*80 + "\n")
    
    return results_summary


# =============================================================================
# ORIGINAL RUN FUNCTIONS (maintained for backward compatibility)
# =============================================================================

def run_with_files(
    strategy_file=DEFAULT_STRATEGY_FILE,
    data_file=DEFAULT_DATA_FILE,
    industry="E-commerce",
    target_audience="Women 25-45",
    mode="both",
):
    """Original run function - maintained for backward compatibility."""
    strategy_file = _resolve_with_project_root(Path(strategy_file or DEFAULT_STRATEGY_FILE), DEFAULT_STRATEGY_FILE)
    data_file = _resolve_with_project_root(Path(data_file or DEFAULT_DATA_FILE), DEFAULT_DATA_FILE)
    mode = (mode or "multi").lower()
    
    if mode not in {"single", "multi", "both"}:
        raise ValueError("mode must be one of: single, multi, both")

    def _run_selected(selected_mode: str):
        print(f"\n=== Running in {selected_mode.upper()} mode ===")
        print("="*60)
        
        strategy_content = read_strategy_file(strategy_file)
        data_info = read_csv_data(data_file)
        sales_data_summary = format_data_summary(data_info)
        
        inputs = {
            'module_name': 'marketing_strategy_report.md',
            'current_strategy': strategy_content,
            'industry': industry or "General",
            'target_audience': target_audience or "General audience",
            'sales_data': sales_data_summary
        }
        
        result = execute_single_run(inputs=inputs, mode=selected_mode, save_individual_report=True)
        
        # Generate visualizations
        if AGENT_MODE_RESULTS_FILE.exists():
            try:
                vg = VisualizationGenerator()
                vg.generate_all_visualizations()
            except Exception as e:
                print(f"[WARN] Visualization generation failed: {e}")
        
        return result

    if mode == "both":
        _run_selected("single")
        _run_selected("multi")
    else:
        _run_selected(mode)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Marketing Strategy System with Statistical Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run FULL EXPERIMENT SUITE (default when using crewai run)
  crewai run
  python main.py
  
  # Quick mode with fewer runs (faster testing)
  python main.py --quick
  
  # Run specific experiments individually:
  python main.py --experiment comparison --runs 10
  python main.py --experiment ablation --runs 5
  python main.py --experiment ordering --runs 5
  python main.py --experiment validate-data
  python main.py --experiment llm-judge
  
  # Legacy: single run mode
  python main.py --experiment none --mode multi
        """
    )
    
    parser.add_argument("--mode", choices=["single", "multi", "both"], default="both",
                        help="Agent mode (default: both)")
    parser.add_argument("--experiment", choices=["none", "multi-run", "comparison", "ablation", "ordering",
                                                  "sensitivity", "validate-data", "failure-analysis", "llm-judge"],
                        default="none", help="Experiment type to run")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of runs for statistical experiments (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Path to strategy file")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data CSV file")
    parser.add_argument("--industry", type=str, default="E-commerce",
                        help="Industry context (default: E-commerce)")
    parser.add_argument("--audience", type=str, default="Women 25-45",
                        help="Target audience (default: Women 25-45)")
    parser.add_argument("--full-suite", action="store_true",
                        help="Run complete experiment suite (default when using crewai run)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer runs for faster testing")
    
    return parser.parse_args()


def run():
    """
    Main entry point with CLI support.
    
    When running via `crewai run`, this executes the full experiment suite by default.
    Use --experiment flag to run specific experiments individually.
    """
    args = parse_args()
    
    # Determine number of runs
    n_runs = args.runs
    if args.quick:
        n_runs = min(3, args.runs)
        print("[INFO] Quick mode: using 3 runs per experiment")
    
    if args.experiment == "multi-run":
        run_multiple_trials(
            n_runs=n_runs,
            mode=args.mode if args.mode != "both" else "multi",
            strategy_file=args.strategy,
            data_file=args.data,
            industry=args.industry,
            target_audience=args.audience,
            base_seed=args.seed
        )
    elif args.experiment == "comparison":
        run_comparison_experiment(
            n_runs=n_runs,
            strategy_file=args.strategy,
            data_file=args.data,
            industry=args.industry,
            target_audience=args.audience,
            base_seed=args.seed
        )
    elif args.experiment == "ablation":
        run_ablation_study(
            n_runs=n_runs,
            strategy_file=args.strategy,
            data_file=args.data,
            industry=args.industry,
            target_audience=args.audience,
            base_seed=args.seed
        )
    elif args.experiment == "ordering":
        run_ordering_experiment(
            n_runs=n_runs,
            strategy_file=args.strategy,
            data_file=args.data,
            industry=args.industry,
            target_audience=args.audience,
            base_seed=args.seed
        )
    elif args.experiment == "sensitivity":
        # Run sensitivity analysis
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from sensitivity_analysis import SensitivityAnalyzer
            print("\n" + "="*70)
            print("SENSITIVITY ANALYSIS")
            print("="*70 + "\n")
            analyzer = SensitivityAnalyzer(output_dir="output")
            analyzer.run_full_sensitivity_analysis(n_runs=n_runs, mode=args.mode)
        except ImportError as e:
            print(f"[ERROR] sensitivity_analysis.py not found in path: {e}")
    elif args.experiment == "validate-data":
        # Validate dataset
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from dataset_validation import DatasetValidator
            print("\n" + "="*70)
            print("DATASET VALIDATION")
            print("="*70 + "\n")
            data_file = args.data or DEFAULT_DATA_FILE
            df = pd.read_csv(data_file)
            validator = DatasetValidator(output_dir="output")
            validator.validate_dataset(df, industry="fashion", dataset_name="ecommerce_fashion")
            validator.generate_validation_report("ecommerce_fashion")
        except ImportError as e:
            print(f"[ERROR] dataset_validation.py not found in path: {e}")
    elif args.experiment == "failure-analysis":
        # Run failure analysis
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from failure_analysis import FailureAnalyzer
            print("\n" + "="*70)
            print("FAILURE ANALYSIS")
            print("="*70 + "\n")
            analyzer = FailureAnalyzer(output_dir="output")
            stats = analyzer.get_failure_statistics()
            print(json.dumps(stats, indent=2))
            analyzer.generate_failure_report()
            analyzer.generate_visualizations()
        except ImportError as e:
            print(f"[ERROR] failure_analysis.py not found in path: {e}")
    elif args.experiment == "llm-judge":
        # Run LLM-as-judge evaluation
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from llm_judge import LLMJudge
            print("\n" + "="*70)
            print("LLM-AS-JUDGE EVALUATION")
            print("="*70 + "\n")
            
            # Load the final strategy
            strategy_path = Path("output/final_strategy.json")
            if not strategy_path.exists():
                strategy_path = Path("output/openai_strategy.json")
            
            if strategy_path.exists():
                with open(strategy_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    try:
                        strategy_doc = json.loads(content)
                    except:
                        strategy_doc = {"strategy_text": content}
                
                user_inputs = {
                    "industry": args.industry,
                    "target_audience": args.audience,
                }
                
                judge = LLMJudge(output_dir="output")
                results = judge.evaluate_strategy(strategy_doc, user_inputs)
                print(json.dumps(results, indent=2))
                judge.generate_evaluation_report()
            else:
                print("[ERROR] No strategy file found to evaluate. Run a strategy generation first.")
        except ImportError as e:
            print(f"[ERROR] llm_judge.py not found in path: {e}")
    else:
        # DEFAULT: Run the full experiment suite
        # This is what happens when you type `crewai run`
        run_full_experiment_suite(
            strategy_file=args.strategy,
            data_file=args.data,
            industry=args.industry,
            target_audience=args.audience,
            n_runs=n_runs,
            base_seed=args.seed
        )


if __name__ == "__main__":
    run()