#!/usr/bin/env python
"""
Enhanced Visualization Generator for Marketing Strategy System

Improvements:
- Statistical annotations on figures (Feedback Item #14)
- Multi-run comparison support (Feedback Item #1)
- Formalized divergence visualization (Feedback Item #6)
- Ablation study visualizations (Feedback Item #2)
"""

import matplotlib
matplotlib.use("Agg")

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Optional scipy for statistics
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


class DiversityMetrics:
    """Calculate diversity metrics for agent outputs."""
    
    @staticmethod
    def calculate_vocabulary_diversity(text1: str, text2: str) -> float:
        """Calculate vocabulary overlap ratio (lower = more diverse)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return 1.0 - (intersection / union) if union > 0 else 0.0
    
    @staticmethod
    def calculate_edit_ratio(original: str, modified: str) -> float:
        """Estimate how much content changed."""
        words_orig = original.lower().split()
        words_mod = modified.lower().split()
        if not words_orig:
            return 1.0
        common = len(set(words_orig) & set(words_mod))
        return 1.0 - (common / len(words_orig))
    
    @staticmethod
    def count_strategic_elements(text: str) -> Dict[str, int]:
        """Count strategic elements in text."""
        text_lower = text.lower()
        return {
            "kpi_mentions": text_lower.count("kpi") + text_lower.count("metric") + text_lower.count("target"),
            "competitor_mentions": text_lower.count("competitor") + text_lower.count("vs") + text_lower.count("versus"),
            "tactic_mentions": text_lower.count("tactic") + text_lower.count("strategy") + text_lower.count("campaign"),
            "budget_mentions": text_lower.count("$") + text_lower.count("budget") + text_lower.count("cost"),
            "channel_mentions": sum([
                text_lower.count(ch) for ch in ["instagram", "tiktok", "email", "sms", "google ads", "facebook"]
            ])
        }


class VisualizationGenerator:
    """Generate comprehensive visualizations with statistical annotations."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load various result files
        self.results_file = self.output_dir / "agent_mode_results.json"
        self.multi_run_file = self.output_dir / "multi_run_statistics.json"
        self.comparison_file = self.output_dir / "statistical_comparison.json"
        self.ablation_file = self.output_dir / "ablation_results.json"
        self.ordering_file = self.output_dir / "ordering_experiment.json"
        
        self.results: List[Dict] = []
        self.multi_run_data: Optional[Dict] = None
        self.comparison_data: Optional[Dict] = None
        self.ablation_data: Optional[Dict] = None
        self.ordering_data: Optional[Dict] = None
        
        self._load_data()
        self.latest_by_mode = self._latest_by_mode()
        self.diversity_calc = DiversityMetrics()

    def _load_data(self):
        """Load all available data files."""
        if self.results_file.exists():
            with open(self.results_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.results = raw if isinstance(raw, list) else []
        
        if self.multi_run_file.exists():
            with open(self.multi_run_file, "r", encoding="utf-8") as f:
                self.multi_run_data = json.load(f)
        
        if self.comparison_file.exists():
            with open(self.comparison_file, "r", encoding="utf-8") as f:
                self.comparison_data = json.load(f)
        
        if self.ablation_file.exists():
            with open(self.ablation_file, "r", encoding="utf-8") as f:
                self.ablation_data = json.load(f)
        
        if self.ordering_file.exists():
            with open(self.ordering_file, "r", encoding="utf-8") as f:
                self.ordering_data = json.load(f)

    def _latest_by_mode(self) -> Dict[str, Dict]:
        """Pick the most recent result per agent_mode."""
        latest: Dict[str, Dict] = {}
        for item in self.results:
            mode = item.get("agent_mode")
            ts = item.get("timestamp", "")
            if not mode:
                continue
            if mode not in latest or ts > latest[mode].get("timestamp", ""):
                latest[mode] = item
        return latest

    def _load_strategy_file(self, filename: str) -> Optional[str]:
        """Load strategy output file."""
        path = self.output_dir / filename
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                try:
                    doc = json.loads(text)
                    if isinstance(doc, dict) and doc.get("strategy_text"):
                        return doc["strategy_text"]
                except Exception:
                    pass
            return text
        return None

    def _add_significance_annotation(self, ax, p_value: float, x_pos: float, y_pos: float):
        """Add significance stars to a plot."""
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        else:
            stars = "ns"
        
        ax.annotate(stars, xy=(x_pos, y_pos), ha='center', fontsize=14, fontweight='bold')

    # =========================================================================
    # MULTI-RUN STATISTICAL VISUALIZATIONS
    # =========================================================================

    def create_multi_run_distribution(self):
        """
        Create distribution plot for multi-run results.
        Shows violin/box plots with individual points.
        """
        if not self.multi_run_data:
            print("[WARN] Skipping multi-run distribution (no multi-run data).")
            return
        
        runs = self.multi_run_data.get("individual_runs", [])
        valid_runs = [r for r in runs if "error" not in r and r.get("rubric_percent") is not None]
        
        if len(valid_runs) < 3:
            print("[WARN] Skipping multi-run distribution (need at least 3 successful runs).")
            return
        
        scores = [r["rubric_percent"] for r in valid_runs]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Violin plot
        parts = ax.violinplot([scores], positions=[1], showmeans=True, showmedians=True)
        parts['bodies'][0].set_facecolor('#2E86AB')
        parts['bodies'][0].set_alpha(0.7)
        
        # Overlay individual points
        jitter = np.random.normal(0, 0.04, len(scores))
        ax.scatter([1 + j for j in jitter], scores, alpha=0.6, c='#E63946', s=50, zorder=5)
        
        # Statistics annotation
        stats = self.multi_run_data.get("statistics", {}).get("rubric", {})
        mean = stats.get("mean", np.mean(scores))
        std = stats.get("std", np.std(scores))
        
        ax.axhline(y=mean, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean:.1f}%')
        ax.fill_between([0.5, 1.5], mean-std, mean+std, alpha=0.2, color='green', label=f'±1 SD: {std:.1f}')
        
        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([1])
        ax.set_xticklabels([self.multi_run_data.get("mode", "Multi").capitalize()])
        ax.set_ylabel("Rubric Score (%)", fontsize=12, fontweight='bold')
        ax.set_title(f"Score Distribution Across {len(valid_runs)} Runs\n(Mean ± SD: {mean:.1f} ± {std:.1f}%)", 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "multi_run_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created multi_run_distribution.png")

    def create_statistical_comparison_chart(self):
        """
        Create bar chart with error bars and significance testing.
        Compares single vs multi-agent with confidence intervals.
        """
        if not self.comparison_data:
            print("[WARN] Skipping statistical comparison (no comparison data).")
            return
        
        single_stats = self.comparison_data.get("single_agent", {}).get("rubric", {})
        multi_stats = self.comparison_data.get("multi_agent", {}).get("rubric", {})
        comparison = self.comparison_data.get("comparison", {})
        
        if not single_stats or not multi_stats:
            print("[WARN] Skipping statistical comparison (incomplete statistics).")
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        labels = ['Single Agent', 'Multi Agent']
        means = [single_stats.get("mean", 0), multi_stats.get("mean", 0)]
        stds = [single_stats.get("std", 0), multi_stats.get("std", 0)]
        
        colors = ['#2E86AB', '#F18F01']
        x = np.arange(len(labels))
        
        bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, 
                      edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add individual values as text
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 2,
                   f'{mean:.1f}%\n(±{std:.1f})', ha='center', fontsize=11, fontweight='bold')
        
        # Add significance annotation if available
        t_test = comparison.get("t_test", {})
        if t_test:
            p_value = t_test.get("p_value", 1.0)
            max_height = max(means) + max(stds) + 10
            
            # Draw significance bracket
            ax.plot([0, 0, 1, 1], [max_height-2, max_height, max_height, max_height-2], 
                   'k-', linewidth=1.5)
            
            self._add_significance_annotation(ax, p_value, 0.5, max_height + 2)
            
            # Add p-value text
            ax.text(0.5, max_height + 6, f'p = {p_value:.4f}', ha='center', fontsize=10)
        
        # Effect size annotation
        effect = comparison.get("effect_size", {})
        if effect:
            cohens_d = effect.get("cohens_d", 0)
            interp = effect.get("interpretation", "")
            ax.text(0.98, 0.02, f"Cohen's d = {cohens_d:.2f} ({interp})",
                   transform=ax.transAxes, ha='right', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_ylabel("Rubric Score (%)", fontsize=12, fontweight='bold')
        ax.set_title("Single vs Multi-Agent: Statistical Comparison\n(Error bars show ±1 SD)", 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, max(means) + max(stds) + 20)
        ax.axhline(y=85, color='green', linestyle='--', alpha=0.5, label='Target: 85%')
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "statistical_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created statistical_comparison.png")

    # =========================================================================
    # ABLATION STUDY VISUALIZATIONS
    # =========================================================================

    def create_ablation_comparison(self):
        """Create bar chart comparing ablation configurations."""
        if not self.ablation_data:
            print("[WARN] Skipping ablation comparison (no ablation data).")
            return
        
        results = self.ablation_data.get("results_by_config", {})
        if not results:
            print("[WARN] Skipping ablation comparison (empty results).")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        configs = list(results.keys())
        means = []
        stds = []
        
        for config in configs:
            stats = results[config].get("statistics", {}).get("rubric", {})
            means.append(stats.get("mean", 0))
            stds.append(stats.get("std", 0))
        
        x = np.arange(len(configs))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(configs)))
        
        bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                   f'{mean:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        # Format config names for display
        display_names = [c.replace('_', '\n') for c in configs]
        
        ax.set_ylabel("Rubric Score (%)", fontsize=12, fontweight='bold')
        ax.set_title("Ablation Study: Configuration Comparison\n(Error bars show ±1 SD)", 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=10, rotation=0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "ablation_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created ablation_comparison.png")

    # =========================================================================
    # ORDERING EXPERIMENT VISUALIZATIONS
    # =========================================================================

    def create_ordering_comparison(self):
        """Create chart comparing different agent orderings."""
        if not self.ordering_data:
            print("[WARN] Skipping ordering comparison (no ordering data).")
            return
        
        results = self.ordering_data.get("results_by_ordering", {})
        if not results:
            print("[WARN] Skipping ordering comparison (empty results).")
            return
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        orderings = list(results.keys())
        means = []
        stds = []
        agent_orders = []
        
        for ordering in orderings:
            stats = results[ordering].get("statistics", {}).get("rubric", {})
            means.append(stats.get("mean", 0))
            stds.append(stats.get("std", 0))
            agent_orders.append(" → ".join(results[ordering].get("agent_order", [])))
        
        x = np.arange(len(orderings))
        colors = plt.cm.Set2(np.linspace(0, 1, len(orderings)))
        
        bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                   f'{mean:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        ax.set_ylabel("Rubric Score (%)", fontsize=12, fontweight='bold')
        ax.set_title("Pipeline Ordering Experiment\n(Testing agent sequence effects)", 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{o}\n({ao})" for o, ao in zip(orderings, agent_orders)], 
                          fontsize=9, rotation=0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "ordering_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created ordering_comparison.png")

    # =========================================================================
    # ORIGINAL VISUALIZATIONS (ENHANCED)
    # =========================================================================

    def create_quality_comparison(self):
        """Bar chart comparing rubric percent for single vs multi."""
        data = [
            (mode.capitalize(), res.get("rubric_percent"))
            for mode, res in self.latest_by_mode.items()
            if res.get("rubric_percent") is not None
        ]
        if not data:
            print("[WARN] Skipping quality comparison (no rubric data).")
            return

        labels, values = zip(*sorted(data))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#2E86AB", "#F18F01"]
        bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_ylim(0, 110)
        ax.set_ylabel("Rubric Score (%)", fontsize=12, fontweight='bold')
        ax.set_title("Quality Comparison: Single vs Multi-Agent", fontsize=14, fontweight='bold')
        ax.axhline(y=85, color='green', linestyle='--', alpha=0.5, label='Target: 85%')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 2, 
                   f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight='bold')
        
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "quality_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created quality_comparison.png")

    def create_execution_time_chart(self):
        """Bar chart comparing execution time."""
        data = [
            (mode.capitalize(), res.get("execution_time_seconds"))
            for mode, res in self.latest_by_mode.items()
            if res.get("execution_time_seconds") is not None
        ]
        if not data:
            print("[WARN] Skipping execution time comparison (no timing data).")
            return

        labels, values = zip(*sorted(data))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#6A994E", "#A23B72"]
        bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel("Execution Time (seconds)", fontsize=12, fontweight='bold')
        ax.set_title("Speed Comparison: Single vs Multi-Agent", fontsize=14, fontweight='bold')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, 
                   f"{val:.1f}s", ha="center", va="bottom", fontsize=11, fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "execution_time_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created execution_time_comparison.png")

    def create_diversity_lift_chart(self):
        """Show diversity impact of multi-agent vs single."""
        single = self.latest_by_mode.get("single")
        multi = self.latest_by_mode.get("multi")
        if not single or not multi:
            print("[WARN] Skipping diversity lift (need both single and multi runs).")
            return

        metrics = {}
        
        if single.get("rubric_percent") is not None and multi.get("rubric_percent") is not None:
            metrics["Rubric % Lift"] = multi["rubric_percent"] - single["rubric_percent"]

        ref = multi.get("refinement_delta") or {}
        if isinstance(ref, dict):
            if ref.get("kpis_delta") is not None:
                metrics["KPI Count Delta"] = ref["kpis_delta"]
            if ref.get("tactics_delta") is not None:
                metrics["Tactics Count Delta"] = ref["tactics_delta"]
            if ref.get("edit_ratio") is not None:
                metrics["Content Edit Ratio"] = ref["edit_ratio"] * 100

        if not metrics:
            print("[WARN] Skipping diversity lift (no comparison metrics found).")
            return

        labels = list(metrics.keys())
        values = list(metrics.values())
        colors = ['#2E86AB' if v >= 0 else '#E63946' for v in values]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(labels, values, color=colors, edgecolor='black', linewidth=1.5)
        ax.axvline(0, color="black", linewidth=1.5)
        
        for bar, val in zip(bars, values):
            x_pos = val + (2 if val >= 0 else -2)
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:+.1f}",
                   va="center", ha="left" if val >= 0 else "right", fontsize=10, fontweight='bold')
        
        ax.set_title("Diversity Impact: Multi vs Single Agent", fontsize=14, fontweight='bold')
        ax.set_xlabel("Change (Multi - Single)", fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "diversity_lift.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created diversity_lift.png")

    def create_content_diversity_heatmap(self):
        """Create heatmap showing content evolution across agent stages."""
        openai_text = self._load_strategy_file("openai_strategy.json")
        claude_text = self._load_strategy_file("claude_strategy.json")
        gemini_text = self._load_strategy_file("final_strategy.json")
        
        if not all([openai_text, claude_text, gemini_text]):
            print("[WARN] Skipping content diversity heatmap (missing multi-agent outputs).")
            return
        
        openai_to_claude = self.diversity_calc.calculate_edit_ratio(openai_text, claude_text)
        claude_to_gemini = self.diversity_calc.calculate_edit_ratio(claude_text, gemini_text)
        openai_to_gemini = self.diversity_calc.calculate_edit_ratio(openai_text, gemini_text)
        
        data = np.array([
            [0, openai_to_claude, openai_to_gemini],
            [openai_to_claude, 0, claude_to_gemini],
            [openai_to_gemini, claude_to_gemini, 0]
        ])
        
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        agents = ['Phase 1:\nOpenAI', 'Phase 2:\nClaude', 'Phase 3:\nGemini']
        ax.set_xticks(np.arange(len(agents)))
        ax.set_yticks(np.arange(len(agents)))
        ax.set_xticklabels(agents, fontsize=10)
        ax.set_yticklabels(agents, fontsize=10)

        for i in range(len(agents)):
            for j in range(len(agents)):
                if i != j:
                    percentage = data[i, j] * 100
                    ax.text(j, i, f'{percentage:.0f}%',
                           ha="center", va="center", color="black", fontweight='bold', fontsize=12)

        ax.set_title("Content Evolution Across Phases\n(Jaccard Distance)", fontsize=13, fontweight='bold')
        cbar = fig.colorbar(im, ax=ax, label='Content Divergence')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "content_diversity_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created content_diversity_heatmap.png")

    def create_comprehensive_comparison_table(self):
        """Create visual comparison table."""
        single = self.latest_by_mode.get("single") or {}
        multi = self.latest_by_mode.get("multi") or {}
        
        if not single or not multi:
            print("[WARN] Skipping comparison table (need both modes).")
            return
        
        metrics = {
            "Quality Score (%)": (
                single.get("rubric_percent", 0),
                multi.get("rubric_percent", 0)
            ),
            "Execution Time (s)": (
                single.get("execution_time_seconds", 0),
                multi.get("execution_time_seconds", 0)
            ),
            "Schema Valid": (
                "Yes" if single.get("schema_valid") else "No",
                "Yes" if multi.get("schema_valid") else "No"
            ),
        }
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [["Metric", "Single Agent", "Multi Agent", "Δ Change"]]
        
        for metric, (single_val, multi_val) in metrics.items():
            if isinstance(single_val, (int, float)) and isinstance(multi_val, (int, float)):
                delta = multi_val - single_val
                delta_str = f"{delta:+.1f}"
                single_str = f"{single_val:.1f}"
                multi_str = f"{multi_val:.1f}"
            else:
                single_str = str(single_val)
                multi_str = str(multi_val)
                delta_str = "—"
            
            table_data.append([metric, single_str, multi_str, delta_str])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.3, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        for i in range(4):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title("Comparison: Single vs Multi-Agent", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_table.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created comparison_table.png")

    # =========================================================================
    # MAIN GENERATION METHOD
    # =========================================================================

    def generate_all_visualizations(self):
        """Generate all available visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70 + "\n")

        viz_methods = [
            # Standard comparisons
            ("Quality Comparison", self.create_quality_comparison),
            ("Execution Time", self.create_execution_time_chart),
            ("Diversity Lift", self.create_diversity_lift_chart),
            ("Content Diversity Heatmap", self.create_content_diversity_heatmap),
            ("Comparison Table", self.create_comprehensive_comparison_table),
            
            # Statistical/multi-run visualizations
            ("Multi-Run Distribution", self.create_multi_run_distribution),
            ("Statistical Comparison", self.create_statistical_comparison_chart),
            
            # Experiment visualizations
            ("Ablation Comparison", self.create_ablation_comparison),
            ("Ordering Comparison", self.create_ordering_comparison),
        ]
        
        for name, method in viz_methods:
            try:
                print(f"[VIZ] Generating {name}...")
                method()
            except Exception as e:
                print(f"[ERROR] Error generating {name}: {e}")

        print("\n" + "=" * 70)
        print(f"[SUCCESS] Visualization generation complete!")
        print(f"[INFO] Files saved in: {self.output_dir.absolute()}/")
        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    try:
        generator = VisualizationGenerator()
        generator.generate_all_visualizations()
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        raise


if __name__ == "__main__":
    main()