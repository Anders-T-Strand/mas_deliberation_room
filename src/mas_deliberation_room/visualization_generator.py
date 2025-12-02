#!/usr/bin/env python
"""
Enhanced Visualization Generator for Marketing Strategy System
Creates comprehensive charts comparing single vs multi-agent runs with diversity metrics.
"""

import matplotlib

# Use a non-interactive backend to avoid Tcl/Tk errors in threaded/CLI runs
matplotlib.use("Agg")

import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


class DiversityMetrics:
    # Calculate diversity metrics for agent outputs.
    
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
        """Estimate how much content changed (simple word-based)."""
        words_orig = original.lower().split()
        words_mod = modified.lower().split()
        if not words_orig:
            return 1.0
        
        # Simple Levenshtein-like ratio at word level
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
    """Generate comprehensive visualizations for agent-mode comparisons."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results_file = self.output_dir / "agent_mode_results.json"
        
        if not self.results_file.exists():
            raise FileNotFoundError(
                "agent_mode_results.json not found. Run both single and multi-agent modes first."
            )

        with open(self.results_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.results: List[Dict] = raw if isinstance(raw, list) else []
        self.latest_by_mode = self._latest_by_mode()
        self.diversity_calc = DiversityMetrics()

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
        """
        Load a strategy output file if it exists (supports .txt or .json with strategy_text).
        Returns plain string content or None.
        """
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
        # Try json fallback for legacy filenames
        alt_json = path.with_suffix(".json")
        if not path.suffix and alt_json.exists():
            try:
                doc = json.loads(alt_json.read_text(encoding="utf-8"))
                if isinstance(doc, dict) and doc.get("strategy_text"):
                    return doc["strategy_text"]
            except Exception:
                return None
        return None

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
        """Show how diversity (multi-agent) changes quality/content vs single."""
        single = self.latest_by_mode.get("single")
        multi = self.latest_by_mode.get("multi")
        if not single or not multi:
            print("[WARN] Skipping diversity lift (need both single and multi runs).")
            return

        metrics = {}
        
        # Quality lift
        if single.get("rubric_percent") is not None and multi.get("rubric_percent") is not None:
            metrics["Rubric % Lift"] = multi["rubric_percent"] - single["rubric_percent"]

        # Refinement deltas
        ref = multi.get("refinement_delta") or {}
        if isinstance(ref, dict):
            if ref.get("kpis_delta") is not None:
                metrics["KPI Count Delta"] = ref["kpis_delta"]
            if ref.get("tactics_delta") is not None:
                metrics["Tactics Count Delta"] = ref["tactics_delta"]
            if ref.get("edit_ratio") is not None:
                metrics["Content Edit Ratio"] = ref["edit_ratio"] * 100  # Convert to %

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

    def create_strategic_element_comparison(self):
        """Compare strategic elements between single and multi-agent outputs."""
        single_text = self._load_strategy_file("openai_strategy.txt") or self._load_strategy_file("openai_strategy.json")
        multi_text = self._load_strategy_file("final_strategy.txt") or self._load_strategy_file("final_strategy.json")
        
        if not single_text or not multi_text:
            print("[WARN] Skipping strategic element comparison (missing output files).")
            return
        
        single_elements = self.diversity_calc.count_strategic_elements(single_text)
        multi_elements = self.diversity_calc.count_strategic_elements(multi_text)
        
        categories = list(single_elements.keys())
        single_vals = [single_elements[c] for c in categories]
        multi_vals = [multi_elements[c] for c in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        bars1 = ax.bar(x - width/2, single_vals, width, label='Single Agent', 
                       color='#2E86AB', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, multi_vals, width, label='Multi Agent', 
                       color='#F18F01', edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Mention Count', fontsize=12, fontweight='bold')
        ax.set_title('Strategic Element Coverage: Single vs Multi-Agent', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ").title() for c in categories], rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "strategic_elements.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created strategic_elements.png")

    def create_content_diversity_heatmap(self):
        """Create heatmap showing content evolution across agent stages."""
        # Load all three stages for multi-agent
        openai_text = self._load_strategy_file("openai_strategy.txt") or self._load_strategy_file("openai_strategy.json")
        claude_text = self._load_strategy_file("claude_strategy.txt") or self._load_strategy_file("claude_strategy.json")
        gemini_text = self._load_strategy_file("final_strategy.txt") or self._load_strategy_file("final_strategy.json")
        
        if not all([openai_text, claude_text, gemini_text]):
            print("[WARN] Skipping content diversity heatmap (missing multi-agent outputs).")
            return
        
        # Calculate edit ratios
        openai_to_claude = self.diversity_calc.calculate_edit_ratio(openai_text, claude_text)
        claude_to_gemini = self.diversity_calc.calculate_edit_ratio(claude_text, gemini_text)
        openai_to_gemini = self.diversity_calc.calculate_edit_ratio(openai_text, gemini_text)
        
        # Calculate vocabulary diversity
        vocab_oc = self.diversity_calc.calculate_vocabulary_diversity(openai_text, claude_text)
        vocab_cg = self.diversity_calc.calculate_vocabulary_diversity(claude_text, gemini_text)
        vocab_og = self.diversity_calc.calculate_vocabulary_diversity(openai_text, gemini_text)
        
        # Create heatmap data
        data = np.array([
            [0, openai_to_claude, openai_to_gemini],
            [openai_to_claude, 0, claude_to_gemini],
            [openai_to_gemini, claude_to_gemini, 0]
        ])
        
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        agents = ['Phase 1:\nOpenAI Analyst\n(Baseline)', 'Phase 2:\nClaude Creative\n(Enhancement)', 'Phase 3:\nGemini Operations\n(Finalization)']
        ax.set_xticks(np.arange(len(agents)))
        ax.set_yticks(np.arange(len(agents)))
        ax.set_xticklabels(agents, fontsize=10)
        ax.set_yticklabels(agents, fontsize=10)

        # Annotate cells with percentages
        for i in range(len(agents)):
            for j in range(len(agents)):
                if i != j:
                    percentage = data[i, j] * 100
                    text = ax.text(j, i, f'{percentage:.0f}%',
                                 ha="center", va="center", color="black", fontweight='bold', fontsize=12)

        ax.set_title("Multi-Agent Deliberation: Content Evolution Across Phases\n(Shows how each agent transforms the strategy)",
                    fontsize=13, fontweight='bold', pad=15)
        cbar = fig.colorbar(im, ax=ax, label='Content Change (%)')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%\n(identical)', '25%', '50%', '75%', '100%\n(different)'])
        plt.tight_layout()
        plt.savefig(self.output_dir / "content_diversity_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created content_diversity_heatmap.png")

    def create_efficiency_tradeoff_scatter(self):
        """Scatter plot showing quality vs speed tradeoff."""
        data_points = []
        for mode, res in self.latest_by_mode.items():
            quality = res.get("rubric_percent")
            time = res.get("execution_time_seconds")
            if quality is not None and time is not None:
                data_points.append({
                    "mode": mode.capitalize(),
                    "quality": quality,
                    "time": time,
                    "efficiency": quality / time  # Quality per second
                })
        
        if len(data_points) < 2:
            print("[WARN] Skipping efficiency tradeoff (need both modes).")
            return
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for point in data_points:
            color = '#2E86AB' if point["mode"] == "Single" else '#F18F01'
            marker = 'o' if point["mode"] == "Single" else 's'
            size = point["efficiency"] * 100  # Scale for visibility
            
            ax.scatter(point["time"], point["quality"], 
                      s=size*10, c=color, marker=marker, 
                      edgecolors='black', linewidths=2, alpha=0.7,
                      label=f"{point['mode']} (eff: {point['efficiency']:.2f})")
            
            # Annotate
            ax.annotate(point["mode"], 
                       xy=(point["time"], point["quality"]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        ax.set_xlabel("Execution Time (seconds)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Quality Score (%)", fontsize=12, fontweight='bold')
        ax.set_title("Quality vs Speed Tradeoff\n(Bubble size = Efficiency)", 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "efficiency_tradeoff.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created efficiency_tradeoff.png")

    def create_comprehensive_comparison_table(self):
        """Create a visual comparison table of key metrics."""
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
                "[Y]" if single.get("schema_valid") else "[N]",
                "[Y]" if multi.get("schema_valid") else "[N]"
            ),
            "KPI Delta": (
                0,
                (multi.get("refinement_delta") or {}).get("kpis_delta", 0)
            ),
            "Tactics Delta": (
                0,
                (multi.get("refinement_delta") or {}).get("tactics_delta", 0)
            )
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [["Metric", "Single Agent", "Multi Agent", "Δ Change"]]
        
        for metric, (single_val, multi_val) in metrics.items():
            if isinstance(single_val, (int, float)) and isinstance(multi_val, (int, float)):
                delta = multi_val - single_val
                delta_str = f"{delta:+.1f}" if isinstance(delta, float) else f"{delta:+d}"
                single_str = f"{single_val:.1f}" if isinstance(single_val, float) else str(single_val)
                multi_str = f"{multi_val:.1f}" if isinstance(multi_val, float) else str(multi_val)
            else:
                single_str = str(single_val)
                multi_str = str(multi_val)
                delta_str = "—"
            
            table_data.append([metric, single_str, multi_str, delta_str])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.3, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color code changes
        for i in range(1, len(table_data)):
            delta_cell = table[(i, 3)]
            if table_data[i][3] not in ["—", "[Y]", "[N]"]:
                try:
                    val = float(table_data[i][3].replace("+", ""))
                    if val > 0:
                        delta_cell.set_facecolor('#90EE90')
                    elif val < 0:
                        delta_cell.set_facecolor('#FFB6C1')
                except:
                    pass
        
        plt.title("Comprehensive Comparison: Single vs Multi-Agent", 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / "comparison_table.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("[OK] Created comparison_table.png")

    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "=" * 70)
        print("GENERATING ENHANCED VISUALIZATIONS")
        print("=" * 70 + "\n")

        viz_methods = [
            ("Quality Comparison", self.create_quality_comparison),
            ("Execution Time", self.create_execution_time_chart),
            ("Diversity Lift", self.create_diversity_lift_chart),
            ("Strategic Elements", self.create_strategic_element_comparison),
            ("Content Diversity Heatmap", self.create_content_diversity_heatmap),
            ("Efficiency Tradeoff", self.create_efficiency_tradeoff_scatter),
            ("Comparison Table", self.create_comprehensive_comparison_table),
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
    except FileNotFoundError as e:
        print(f"\n[ERROR] Error: {e}")
        print("\n[INFO] To generate visualizations:")
        print("   1. Run: python main.py (will run multi-agent by default)")
        print("   2. Run: python main.py --mode single (for single-agent)")
        print("   3. Then run this script again\n")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
