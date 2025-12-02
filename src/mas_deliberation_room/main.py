#!/usr/bin/env python

import sys
import warnings
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from mas_deliberation_room.evaluation import EvaluationHarness
from mas_deliberation_room.visualization_generator import VisualizationGenerator
import time


from mas_deliberation_room.crew import MasDeliberationRoom

# from mas_deliberation_room.model_validator import validate_models

# # Run model validation before initializing Crew
# validate_models("src/mas_deliberation_room/agents.yaml")


# Create output directory
os.makedirs('output', exist_ok=True)

# Default dataset/strategy locations so tests don't need to pass paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STRATEGY_FILE = PROJECT_ROOT / "datasets" / "ecommerce_fashion_strategy.txt"
DEFAULT_DATA_FILE = PROJECT_ROOT / "datasets" / "ecommerce_fashion.csv"
AGENT_MODE_RESULTS_FILE = PROJECT_ROOT / "output" / "agent_mode_results.json"


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def _resolve_with_project_root(file_path: Path, default_path: Path) -> Path:
    """
    Try multiple resolutions so users can pass relative paths without breaking.
    Order: as-given, relative to PROJECT_ROOT, relative to PROJECT_ROOT/datasets, then default.
    """
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
    
    file_path = _resolve_with_project_root(Path(file_path), DEFAULT_STRATEGY_FILE)
    
    # Handle text files
    if file_path.suffix.lower() in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Handle docx files
    elif file_path.suffix.lower() == '.docx':
        try:
            import subprocess
            # Use pandoc to convert docx to text
            result = subprocess.run(
                ['pandoc', str(file_path), '-t', 'plain'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError:
            raise Exception("Failed to read .docx file. Ensure pandoc is installed.")
        except FileNotFoundError:
            raise Exception("Pandoc not found. Install with: sudo apt-get install pandoc")
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .txt, .md, or .docx")


def read_csv_data(file_path=DEFAULT_DATA_FILE):
   
    file_path = _resolve_with_project_root(Path(file_path), DEFAULT_DATA_FILE)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    if file_path.suffix.lower() != '.csv':
        raise ValueError(f"Expected .csv file, got: {file_path.suffix}")
    
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Generate summary
    summary = {
        'file_name': file_path.name,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'date_range': None,
        'summary_stats': {}
    }
    
    # Try to detect date column and get range
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    if date_columns:
        date_col = date_columns[0]
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            summary['date_range'] = f"{df[date_col].min()} to {df[date_col].max()}"
        except:
            pass
    
    # Get basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
        summary['summary_stats'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return {
        'dataframe': df,
        'summary': summary
    }


def format_data_summary(data_info):
    """Format data summary for AI agent input"""
    summary = data_info['summary']
    
    output = f"Data File: {summary['file_name']}\n"
    output += f"Total Records: {summary['total_rows']}\n"
    output += f"Metrics Tracked: {summary['total_columns']} columns\n"
    
    if summary['date_range']:
        output += f"Date Range: {summary['date_range']}\n"
    
    output += f"\nAvailable Metrics:\n"
    for col in summary['columns'][:10]:  # Show first 10 columns
        output += f"  - {col}\n"
    
    if summary['summary_stats']:
        output += f"\nKey Statistics:\n"
        for col, stats in list(summary['summary_stats'].items())[:3]:  # Show 3 metrics
            output += f"  {col}:\n"
            output += f"    Mean: {stats['mean']:.2f}\n"
            output += f"    Range: {stats['min']:.2f} - {stats['max']:.2f}\n"
    
    return output


def record_agent_mode_result(agent_mode: str, eval_metrics: dict, execution_time: float):
    """Persist a lightweight summary for single vs multi runs."""
    result_entry = {
        "agent_mode": agent_mode,
        "timestamp": datetime.now().isoformat(),
        "execution_time_seconds": execution_time,
        "rubric_percent": eval_metrics.get("rubric", {}).get("percent"),
        "rubric_score": eval_metrics.get("rubric", {}).get("score_0_to_7"),
        "schema_valid": eval_metrics.get("schema", {}).get("valid"),
        "schema_issues": eval_metrics.get("schema", {}).get("issues", []),
        "refinement_delta": eval_metrics.get("refinement_delta"),
    }

    existing = []
    if AGENT_MODE_RESULTS_FILE.exists():
        try:
            existing = json.loads(AGENT_MODE_RESULTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    existing.append(result_entry)
    AGENT_MODE_RESULTS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def run_interactive():
    """
    Interactive mode - prompt user for file paths
    """
    print("="*60)
    print("AI MARKETING STRATEGY BOARD MEETING")
    print("="*60)
    print()
    
    # Get strategy file
    print("STEP 1: Current Marketing Strategy")
    print("-" * 40)
    print("Please provide your current marketing strategy file")
    print("Supported formats: .txt, .md, .docx")
    print()
    
    strategy_file = input("Enter path to strategy file (press Enter for default): ").strip() or DEFAULT_STRATEGY_FILE
    
    try:
        strategy_content = read_strategy_file(strategy_file)
        print(f"Successfully loaded strategy file ({len(strategy_content)} characters)")
    except Exception as e:
        print(f"Error reading strategy file: {e}")
        return
    
    print()
    
    # Get data file
    print("STEP 2: Performance Data")
    print("-" * 40)
    print("Please provide your performance data CSV file")
    print()
    
    data_file = input("Enter path to CSV data file (press Enter for default): ").strip() or DEFAULT_DATA_FILE
    
    try:
        data_info = read_csv_data(data_file)
        print(f"Successfully loaded data file ({data_info['summary']['total_rows']} rows)")
        sales_data_summary = format_data_summary(data_info)
    except Exception as e:
        print(f"âœ— Error reading data file: {e}")
        return
    
    print()
    
    # Get additional context
    print("STEP 3: Additional Context")
    print("-" * 40)
    
    industry = input("Industry (optional, press Enter to skip): ").strip() or "General"
    target_audience = input("Target Audience (optional, press Enter to skip): ").strip() or "General audience"
    
    print()
    print("="*60)
    print("STARTING AI ANALYSIS...")
    print("="*60)
    print()
    
    # Prepare inputs for CrewAI
    inputs = {
        'module_name': 'marketing_strategy_report.md',
        'current_strategy': strategy_content,
        'industry': industry,
        'target_audience': target_audience,
        'sales_data': sales_data_summary
    }
    
    # Run analysis
    print("🤖 Starting AI analysis...\n")
    try:
        # Create crew
        crew = MasDeliberationRoom().crew()

        # --- TOKEN LIMITER (max 3000 tokens per agent) ---
        for agent in crew.agents:
            # Works across LLM backends CrewAI wraps; if the LLM object exists, set there too.
            setattr(agent, "max_tokens", 3000)
            if hasattr(agent, "llm") and hasattr(agent.llm, "max_tokens"):
                agent.llm.max_tokens = 3000
        print("🔒 Token limit set to 3000 per agent\n")

        # Time the run for evaluation
        import time
        _t0 = time.time()
        result = crew.kickoff(inputs=inputs)
        exec_time = time.time() - _t0

        print("\n✅ Analysis complete! Check output/marketing_strategy_report.md")

        # --- EVALUATION ---
        harness = EvaluationHarness()
        # Point evaluation at JSON outputs to ensure structured parsing
        eval_metrics = harness.run_evaluation(
            crew_result=result,
            user_inputs=inputs,
            execution_time=exec_time,
            test_case_name="manual_run",
            final_output_path=Path("output/final_strategy.json"),
            openai_output_path=Path("output/openai_strategy.json"),
        )
        record_agent_mode_result("multi", eval_metrics, exec_time)

        # --- VISUALIZATION ---
        if AGENT_MODE_RESULTS_FILE.exists():
            vg = VisualizationGenerator()
            vg.generate_all_visualizations()
        else:
            print("ℹ️ Skipping visualizations (agent_mode_results.json not found). "
                "Run single and multi-agent modes to populate comparison charts.")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        raise



def run_with_files(
    strategy_file="..//datasets/ecommerce_fashion_strategy.txt",
    data_file="..//datasets/ecommerce_fashion.csv",
    industry="E-commerce",
    target_audience="Women 25-45",
    mode="both",
):
    strategy_file = _resolve_with_project_root(Path(strategy_file or DEFAULT_STRATEGY_FILE), DEFAULT_STRATEGY_FILE)
    data_file = _resolve_with_project_root(Path(data_file or DEFAULT_DATA_FILE), DEFAULT_DATA_FILE)
    mode = (mode or "multi").lower()
    if mode not in {"single", "multi", "both"}:
        raise ValueError("mode must be one of: single, multi, both")

    def _run_selected(selected_mode: str):
        print(f"\n=== Running in {selected_mode.upper()} mode ===")
        print("="*60)
        print("AI MARKETING STRATEGY BOARD MEETING")
        print("="*60)
        print()
        
        # Load strategy
        print(f"Loading strategy from: {strategy_file}")
        strategy_content = read_strategy_file(strategy_file)
        print(f"Loaded {len(strategy_content)} characters\n")
        
        # Load data
        print(f"Loading data from: {data_file}")
        data_info = read_csv_data(data_file)
        print(f"Loaded {data_info['summary']['total_rows']} rows\n")
        sales_data_summary = format_data_summary(data_info)
        
        # Prepare inputs
        inputs = {
            'module_name': 'marketing_strategy_report.md',
            'current_strategy': strategy_content,
            'industry': industry or "General",
            'target_audience': target_audience or "General audience",
            'sales_data': sales_data_summary
        }
        
        # Run analysis
        print("🤖 Starting AI analysis...\n")
        try:
            # Create crew
            crew_builder = MasDeliberationRoom()
            crew = crew_builder.single_agent_crew() if selected_mode == "single" else crew_builder.crew()

            # --- TOKEN LIMITER (max 3000 tokens per agent) ---
            for agent in crew.agents:
                # Works across LLM backends CrewAI wraps; if the LLM object exists, set there too.
                setattr(agent, "max_tokens", 3000)
                if hasattr(agent, "llm") and hasattr(agent.llm, "max_tokens"):
                    agent.llm.max_tokens = 3000
            print("🔒 Token limit set to 3000 per agent\n")

            # Time the run for evaluation
            import time
            _t0 = time.time()
            result = crew.kickoff(inputs=inputs)
            exec_time = time.time() - _t0

            print("\n✅ Analysis complete! Check output/marketing_strategy_report.md")

            # --- EVALUATION ---
            harness = EvaluationHarness()
            # Use JSON outputs so evaluation can parse structured content
            final_output_path = Path("output/openai_strategy.json") if selected_mode == "single" else Path("output/final_strategy.json")
            openai_output_path = None if selected_mode == "single" else Path("output/openai_strategy.json")
            eval_metrics = harness.run_evaluation(
                crew_result=result,
                user_inputs=inputs,
                execution_time=exec_time,
                test_case_name=f"{selected_mode}_run",
                final_output_path=final_output_path,
                openai_output_path=openai_output_path,
            )
            record_agent_mode_result(selected_mode, eval_metrics, exec_time)

            # --- VISUALIZATION ---
            if AGENT_MODE_RESULTS_FILE.exists():
                vg = VisualizationGenerator()
                vg.generate_all_visualizations()
            else:
                print("ℹ️ Skipping visualizations (agent_mode_results.json not found). "
                    "Run single and multi-agent modes to populate comparison charts.")

        except Exception as e:
            print(f"\n❌ Error during analysis in {selected_mode} mode: {e}")
            raise

    if mode == "both":
        _run_selected("single")
        _run_selected("multi")
    else:
        _run_selected(mode)



def run():
    
    if len(sys.argv) >= 3:
        strategy_file = sys.argv[1]
        data_file = sys.argv[2]
        industry = sys.argv[3] if len(sys.argv) > 3 else None
        target_audience = sys.argv[4] if len(sys.argv) > 4 else None
        mode = sys.argv[5] if len(sys.argv) > 5 else "multi"
        
        run_with_files(strategy_file, data_file, industry, target_audience, mode)
    else:
        # Allow environment override when no CLI args are provided
        run_with_files(mode=os.environ.get("MAS_RUN_MODE", "both"))


if __name__ == "__main__":
    # Example usage in comments:
    # python main.py /path/to/strategy.txt /path/to/data.csv "E-commerce" "Women 25-45"
    # or just run: python main.py (for interactive mode)
    run()
