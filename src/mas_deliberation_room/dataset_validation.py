#!/usr/bin/env python
"""
Dataset Validation Module

Addresses Feedback Item #3: Synthetic dataset justification is insufficient
- Validates that synthetic data resembles real-world e-commerce distributions
- Provides statistical comparison and summary metrics
- Generates distribution histograms and correlation analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Optional scipy for statistical tests
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# INDUSTRY BENCHMARKS
# =============================================================================

# Real-world e-commerce benchmarks for comparison
ECOMMERCE_BENCHMARKS = {
    "fashion": {
        "cac": {"min": 30, "max": 100, "mean": 55, "std": 20},
        "conversion_rate": {"min": 1.0, "max": 5.0, "mean": 2.8, "std": 1.0},
        "aov": {"min": 50, "max": 300, "mean": 120, "std": 50},
        "cltv": {"min": 150, "max": 800, "mean": 400, "std": 150},
        "repeat_rate": {"min": 20, "max": 70, "mean": 45, "std": 15},
        "email_open_rate": {"min": 15, "max": 35, "mean": 22, "std": 5},
        "instagram_engagement": {"min": 1.0, "max": 6.0, "mean": 3.5, "std": 1.2},
    },
    "general_ecommerce": {
        "cac": {"min": 20, "max": 150, "mean": 60, "std": 30},
        "conversion_rate": {"min": 0.5, "max": 4.0, "mean": 2.0, "std": 0.8},
        "aov": {"min": 30, "max": 500, "mean": 100, "std": 80},
        "cltv": {"min": 100, "max": 1000, "mean": 350, "std": 200},
    }
}

# Expected correlations in real e-commerce data
EXPECTED_CORRELATIONS = {
    ("revenue", "orders"): {"min": 0.7, "max": 1.0, "expected": 0.9},
    ("revenue", "marketing_spend"): {"min": 0.3, "max": 0.8, "expected": 0.5},
    ("conversion_rate", "revenue"): {"min": 0.2, "max": 0.7, "expected": 0.4},
    ("cac", "marketing_spend"): {"min": 0.1, "max": 0.6, "expected": 0.3},
    ("instagram_engagement_rate", "instagram_followers"): {"min": -0.3, "max": 0.3, "expected": 0.0},
}


class DatasetValidator:
    """
    Validates synthetic datasets against real-world benchmarks.
    
    Provides:
    - Summary statistics comparison
    - Distribution normality tests
    - Correlation analysis
    - Seasonal pattern detection
    - Visualization generation
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.validation_results: Dict[str, Any] = {}
    
    def validate_dataset(self, df: pd.DataFrame, 
                         industry: str = "fashion",
                         dataset_name: str = "synthetic_data") -> Dict[str, Any]:
        """
        Run comprehensive validation on a dataset.
        
        Args:
            df: DataFrame to validate
            industry: Industry for benchmark comparison ("fashion", "general_ecommerce")
            dataset_name: Name for reports
            
        Returns:
            Dictionary with all validation results
        """
        print(f"\n{'='*70}")
        print(f"DATASET VALIDATION: {dataset_name}")
        print(f"{'='*70}\n")
        
        self.validation_results = {
            "dataset_name": dataset_name,
            "industry": industry,
            "timestamp": datetime.now().isoformat(),
            "basic_info": {},
            "summary_statistics": {},
            "benchmark_comparison": {},
            "normality_tests": {},
            "correlation_analysis": {},
            "seasonal_patterns": {},
            "realism_score": {},
        }
        
        # Basic info
        self._analyze_basic_info(df)
        
        # Summary statistics
        self._compute_summary_statistics(df)
        
        # Benchmark comparison
        self._compare_to_benchmarks(df, industry)
        
        # Normality tests
        self._run_normality_tests(df)
        
        # Correlation analysis
        self._analyze_correlations(df)
        
        # Seasonal patterns
        self._detect_seasonal_patterns(df)
        
        # Overall realism score
        self._compute_realism_score()
        
        # Generate visualizations
        self._generate_visualizations(df, dataset_name)
        
        # Save results
        self._save_results(dataset_name)
        
        return self.validation_results
    
    def _analyze_basic_info(self, df: pd.DataFrame):
        """Analyze basic dataset information."""
        info = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage_kb": df.memory_usage(deep=True).sum() / 1024,
        }
        
        # Date range if date column exists
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if date_cols:
            try:
                dates = pd.to_datetime(df[date_cols[0]])
                info["date_range"] = {
                    "start": str(dates.min()),
                    "end": str(dates.max()),
                    "days": (dates.max() - dates.min()).days + 1,
                }
            except:
                pass
        
        self.validation_results["basic_info"] = info
        print(f"[INFO] Dataset: {info['n_rows']} rows × {info['n_columns']} columns")
    
    def _compute_summary_statistics(self, df: pd.DataFrame):
        """Compute comprehensive summary statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) == 0:
                continue
                
            col_stats = {
                "count": int(len(data)),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "median": float(data.median()),
                "q25": float(data.quantile(0.25)),
                "q75": float(data.quantile(0.75)),
                "iqr": float(data.quantile(0.75) - data.quantile(0.25)),
                "skewness": float(data.skew()),
                "kurtosis": float(data.kurtosis()),
                "cv": float(data.std() / data.mean()) if data.mean() != 0 else 0,
            }
            stats[col] = col_stats
        
        self.validation_results["summary_statistics"] = stats
        print(f"[INFO] Computed statistics for {len(stats)} numeric columns")
    
    def _compare_to_benchmarks(self, df: pd.DataFrame, industry: str):
        """Compare dataset to industry benchmarks."""
        benchmarks = ECOMMERCE_BENCHMARKS.get(industry, ECOMMERCE_BENCHMARKS["general_ecommerce"])
        
        comparisons = {}
        
        # Map column names to benchmark keys
        column_mapping = {
            "cac": ["cac", "customer_acquisition_cost"],
            "conversion_rate": ["conversion_rate", "cvr", "conversions"],
            "aov": ["avg_order_value", "aov", "average_order_value"],
            "cltv": ["cltv", "ltv", "customer_lifetime_value"],
            "email_open_rate": ["email_open_rate", "open_rate"],
            "instagram_engagement": ["instagram_engagement_rate", "engagement_rate"],
        }
        
        for benchmark_key, benchmark_values in benchmarks.items():
            # Find matching column
            matched_col = None
            for possible_col in column_mapping.get(benchmark_key, [benchmark_key]):
                if possible_col in df.columns:
                    matched_col = possible_col
                    break
            
            if matched_col is None:
                continue
            
            data = df[matched_col].dropna()
            if len(data) == 0:
                continue
            
            data_mean = float(data.mean())
            data_std = float(data.std())
            
            # Compare to benchmark
            benchmark_mean = benchmark_values["mean"]
            benchmark_std = benchmark_values["std"]
            
            # Calculate z-score of data mean relative to benchmark
            z_score = (data_mean - benchmark_mean) / benchmark_std if benchmark_std > 0 else 0
            
            # Check if within realistic range
            within_range = benchmark_values["min"] <= data_mean <= benchmark_values["max"]
            
            comparisons[benchmark_key] = {
                "column": matched_col,
                "data_mean": round(data_mean, 2),
                "data_std": round(data_std, 2),
                "benchmark_mean": benchmark_mean,
                "benchmark_std": benchmark_std,
                "benchmark_range": [benchmark_values["min"], benchmark_values["max"]],
                "z_score": round(z_score, 2),
                "within_realistic_range": within_range,
                "realism_assessment": "REALISTIC" if within_range else "OUTSIDE_BENCHMARK",
            }
        
        self.validation_results["benchmark_comparison"] = comparisons
        
        realistic_count = sum(1 for c in comparisons.values() if c["within_realistic_range"])
        print(f"[INFO] Benchmark comparison: {realistic_count}/{len(comparisons)} metrics within realistic range")
    
    def _run_normality_tests(self, df: pd.DataFrame):
        """Run normality tests on numeric columns."""
        if not SCIPY_AVAILABLE:
            print("[WARN] scipy not available, skipping normality tests")
            return
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        tests = {}
        for col in numeric_cols[:10]:  # Limit to first 10 columns
            data = df[col].dropna()
            if len(data) < 8:  # Need minimum samples for Shapiro
                continue
            
            # Shapiro-Wilk test (best for small samples)
            if len(data) <= 5000:
                try:
                    stat, p_value = scipy_stats.shapiro(data[:5000])
                    tests[col] = {
                        "test": "shapiro-wilk",
                        "statistic": round(float(stat), 4),
                        "p_value": round(float(p_value), 4),
                        "is_normal": p_value > 0.05,
                        "interpretation": "Normal distribution" if p_value > 0.05 else "Non-normal distribution",
                    }
                except:
                    pass
        
        self.validation_results["normality_tests"] = tests
        
        normal_count = sum(1 for t in tests.values() if t.get("is_normal", False))
        print(f"[INFO] Normality tests: {normal_count}/{len(tests)} columns appear normally distributed")
    
    def _analyze_correlations(self, df: pd.DataFrame):
        """Analyze correlations and compare to expected patterns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Compute correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Store as nested dict for JSON serialization
        corr_dict = {}
        for col1 in corr_matrix.columns:
            corr_dict[col1] = {}
            for col2 in corr_matrix.columns:
                corr_dict[col1][col2] = round(float(corr_matrix.loc[col1, col2]), 3)
        
        # Check expected correlations
        expected_checks = []
        for (col1, col2), expected in EXPECTED_CORRELATIONS.items():
            if col1 in df.columns and col2 in df.columns:
                actual_corr = float(corr_matrix.loc[col1, col2])
                within_expected = expected["min"] <= actual_corr <= expected["max"]
                
                expected_checks.append({
                    "columns": [col1, col2],
                    "actual_correlation": round(actual_corr, 3),
                    "expected_range": [expected["min"], expected["max"]],
                    "expected_typical": expected["expected"],
                    "realistic": within_expected,
                })
        
        self.validation_results["correlation_analysis"] = {
            "correlation_matrix": corr_dict,
            "expected_correlation_checks": expected_checks,
        }
        
        if expected_checks:
            realistic_count = sum(1 for c in expected_checks if c["realistic"])
            print(f"[INFO] Correlation patterns: {realistic_count}/{len(expected_checks)} match expected relationships")
    
    def _detect_seasonal_patterns(self, df: pd.DataFrame):
        """Detect seasonal patterns in time series data."""
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if not date_cols:
            self.validation_results["seasonal_patterns"] = {"detected": False, "reason": "No date column found"}
            return
        
        try:
            df_copy = df.copy()
            df_copy['_date'] = pd.to_datetime(df_copy[date_cols[0]])
            df_copy = df_copy.sort_values('_date')
            
            # Check for revenue seasonality
            if 'revenue' in df.columns:
                df_copy['_month'] = df_copy['_date'].dt.month
                df_copy['_dayofweek'] = df_copy['_date'].dt.dayofweek
                
                monthly_avg = df_copy.groupby('_month')['revenue'].mean()
                daily_avg = df_copy.groupby('_dayofweek')['revenue'].mean()
                
                # Calculate coefficient of variation for seasonality strength
                monthly_cv = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() > 0 else 0
                daily_cv = daily_avg.std() / daily_avg.mean() if daily_avg.mean() > 0 else 0
                
                self.validation_results["seasonal_patterns"] = {
                    "detected": True,
                    "monthly_pattern": {
                        "cv": round(float(monthly_cv), 3),
                        "strength": "strong" if monthly_cv > 0.2 else "moderate" if monthly_cv > 0.1 else "weak",
                        "monthly_averages": {int(k): round(float(v), 2) for k, v in monthly_avg.items()},
                    },
                    "daily_pattern": {
                        "cv": round(float(daily_cv), 3),
                        "strength": "strong" if daily_cv > 0.15 else "moderate" if daily_cv > 0.05 else "weak",
                        "daily_averages": {int(k): round(float(v), 2) for k, v in daily_avg.items()},
                    },
                    "interpretation": "Real e-commerce data typically shows moderate to strong seasonal patterns"
                }
                print(f"[INFO] Seasonal patterns: Monthly CV={monthly_cv:.3f}, Daily CV={daily_cv:.3f}")
            else:
                self.validation_results["seasonal_patterns"] = {"detected": False, "reason": "No revenue column"}
                
        except Exception as e:
            self.validation_results["seasonal_patterns"] = {"detected": False, "error": str(e)}
    
    def _compute_realism_score(self):
        """Compute overall realism score for the dataset."""
        scores = []
        weights = []
        
        # Benchmark comparison score (weight: 40%)
        benchmark = self.validation_results.get("benchmark_comparison", {})
        if benchmark:
            realistic = sum(1 for c in benchmark.values() if c.get("within_realistic_range", False))
            benchmark_score = realistic / len(benchmark) if benchmark else 0
            scores.append(benchmark_score)
            weights.append(0.4)
        
        # Correlation pattern score (weight: 30%)
        corr_checks = self.validation_results.get("correlation_analysis", {}).get("expected_correlation_checks", [])
        if corr_checks:
            realistic = sum(1 for c in corr_checks if c.get("realistic", False))
            corr_score = realistic / len(corr_checks)
            scores.append(corr_score)
            weights.append(0.3)
        
        # Seasonal pattern score (weight: 20%)
        seasonal = self.validation_results.get("seasonal_patterns", {})
        if seasonal.get("detected"):
            monthly_cv = seasonal.get("monthly_pattern", {}).get("cv", 0)
            # Real data typically has 10-30% monthly variation
            if 0.1 <= monthly_cv <= 0.3:
                seasonal_score = 1.0
            elif 0.05 <= monthly_cv <= 0.4:
                seasonal_score = 0.7
            else:
                seasonal_score = 0.3
            scores.append(seasonal_score)
            weights.append(0.2)
        
        # Distribution shape score (weight: 10%)
        normality = self.validation_results.get("normality_tests", {})
        if normality:
            # Mix of normal and non-normal is realistic for e-commerce
            normal_count = sum(1 for t in normality.values() if t.get("is_normal", False))
            normal_ratio = normal_count / len(normality)
            # 30-70% normal is realistic
            if 0.3 <= normal_ratio <= 0.7:
                dist_score = 1.0
            else:
                dist_score = 0.6
            scores.append(dist_score)
            weights.append(0.1)
        
        # Weighted average
        if scores:
            total_weight = sum(weights)
            overall_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            overall_score = 0.5  # Default if no metrics available
        
        self.validation_results["realism_score"] = {
            "overall": round(overall_score * 100, 1),
            "interpretation": self._interpret_realism_score(overall_score),
            "component_scores": {
                "benchmark_alignment": round(scores[0] * 100, 1) if len(scores) > 0 else None,
                "correlation_patterns": round(scores[1] * 100, 1) if len(scores) > 1 else None,
                "seasonal_patterns": round(scores[2] * 100, 1) if len(scores) > 2 else None,
                "distribution_shapes": round(scores[3] * 100, 1) if len(scores) > 3 else None,
            }
        }
        
        print(f"\n[RESULT] Overall Realism Score: {overall_score*100:.1f}% - {self._interpret_realism_score(overall_score)}")
    
    def _interpret_realism_score(self, score: float) -> str:
        """Interpret realism score."""
        if score >= 0.8:
            return "HIGHLY REALISTIC - Dataset closely matches real-world patterns"
        elif score >= 0.6:
            return "MODERATELY REALISTIC - Dataset shows reasonable real-world characteristics"
        elif score >= 0.4:
            return "PARTIALLY REALISTIC - Some deviations from real-world patterns"
        else:
            return "LOW REALISM - Significant deviations from real-world patterns"
    
    def _generate_visualizations(self, df: pd.DataFrame, dataset_name: str):
        """Generate validation visualizations."""
        print("\n[VIZ] Generating validation visualizations...")
        
        # 1. Distribution histograms for key metrics
        self._plot_distributions(df, dataset_name)
        
        # 2. Correlation heatmap
        self._plot_correlation_heatmap(df, dataset_name)
        
        # 3. Benchmark comparison chart
        self._plot_benchmark_comparison(dataset_name)
        
        # 4. Time series if available
        self._plot_time_series(df, dataset_name)
    
    def _plot_distributions(self, df: pd.DataFrame, dataset_name: str):
        """Plot distribution histograms for key metrics."""
        key_cols = ['revenue', 'cac', 'conversion_rate', 'avg_order_value', 'cltv', 'orders']
        available_cols = [c for c in key_cols if c in df.columns]
        
        if not available_cols:
            return
        
        n_cols = min(len(available_cols), 6)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(available_cols[:6]):
            ax = axes[i]
            data = df[col].dropna()
            
            ax.hist(data, bins=30, color='#2E86AB', edgecolor='black', alpha=0.7)
            ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f}')
            ax.axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.1f}')
            
            ax.set_xlabel(col.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
            ax.legend(fontsize=8)
        
        # Hide unused subplots
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Key Metric Distributions - {dataset_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"dataset_distributions_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Created dataset_distributions_{dataset_name}.png")
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame, dataset_name: str):
        """Plot correlation heatmap."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:12]  # Limit for readability
        
        if len(numeric_cols) < 2:
            return
        
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax,
                    cbar_kws={'label': 'Correlation'})
        
        ax.set_title(f'Correlation Matrix - {dataset_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"dataset_correlations_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Created dataset_correlations_{dataset_name}.png")
    
    def _plot_benchmark_comparison(self, dataset_name: str):
        """Plot benchmark comparison chart."""
        comparisons = self.validation_results.get("benchmark_comparison", {})
        if not comparisons:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = list(comparisons.keys())
        data_means = [comparisons[m]["data_mean"] for m in metrics]
        benchmark_means = [comparisons[m]["benchmark_mean"] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, data_means, width, label='Dataset', color='#2E86AB', edgecolor='black')
        bars2 = ax.bar(x + width/2, benchmark_means, width, label='Industry Benchmark', color='#F18F01', edgecolor='black')
        
        # Add benchmark ranges as error bars
        for i, m in enumerate(metrics):
            benchmark_range = comparisons[m]["benchmark_range"]
            ax.plot([x[i] + width/2, x[i] + width/2], benchmark_range, 'k-', linewidth=2)
            ax.plot([x[i] + width/2 - 0.05, x[i] + width/2 + 0.05], [benchmark_range[0]]*2, 'k-', linewidth=2)
            ax.plot([x[i] + width/2 - 0.05, x[i] + width/2 + 0.05], [benchmark_range[1]]*2, 'k-', linewidth=2)
        
        ax.set_ylabel('Value')
        ax.set_title(f'Dataset vs Industry Benchmarks - {dataset_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"dataset_benchmark_comparison_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Created dataset_benchmark_comparison_{dataset_name}.png")
    
    def _plot_time_series(self, df: pd.DataFrame, dataset_name: str):
        """Plot time series if date column available."""
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if not date_cols or 'revenue' not in df.columns:
            return
        
        try:
            df_plot = df.copy()
            df_plot['_date'] = pd.to_datetime(df_plot[date_cols[0]])
            df_plot = df_plot.sort_values('_date')
            
            fig, ax = plt.subplots(figsize=(14, 5))
            
            ax.plot(df_plot['_date'], df_plot['revenue'], color='#2E86AB', linewidth=1.5)
            ax.fill_between(df_plot['_date'], df_plot['revenue'], alpha=0.3, color='#2E86AB')
            
            # Add trend line
            x_numeric = np.arange(len(df_plot))
            z = np.polyfit(x_numeric, df_plot['revenue'], 1)
            p = np.poly1d(z)
            ax.plot(df_plot['_date'], p(x_numeric), 'r--', label='Trend', linewidth=2)
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Revenue')
            ax.set_title(f'Revenue Time Series - {dataset_name}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"dataset_timeseries_{dataset_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] Created dataset_timeseries_{dataset_name}.png")
        except Exception as e:
            print(f"[WARN] Could not create time series plot: {e}")
    
    def _save_results(self, dataset_name: str):
        """Save validation results to JSON."""
        output_file = self.output_dir / f"dataset_validation_{dataset_name}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVED] Validation results: {output_file}")
    
    def generate_validation_report(self, dataset_name: str) -> str:
        """Generate a markdown validation report."""
        results = self.validation_results
        
        report = f"""# Dataset Validation Report: {dataset_name}

**Generated:** {results.get('timestamp', 'N/A')}
**Industry:** {results.get('industry', 'N/A')}

## Overall Realism Score

**Score: {results.get('realism_score', {}).get('overall', 'N/A')}%**

{results.get('realism_score', {}).get('interpretation', '')}

### Component Scores
"""
        
        components = results.get('realism_score', {}).get('component_scores', {})
        for component, score in components.items():
            if score is not None:
                report += f"- {component.replace('_', ' ').title()}: {score}%\n"
        
        report += f"""
## Basic Information

- Rows: {results.get('basic_info', {}).get('n_rows', 'N/A')}
- Columns: {results.get('basic_info', {}).get('n_columns', 'N/A')}
- Date Range: {results.get('basic_info', {}).get('date_range', {}).get('start', 'N/A')} to {results.get('basic_info', {}).get('date_range', {}).get('end', 'N/A')}

## Benchmark Comparison

| Metric | Dataset Mean | Benchmark Mean | Within Range |
|--------|-------------|----------------|--------------|
"""
        
        for metric, data in results.get('benchmark_comparison', {}).items():
            status = "✅" if data.get('within_realistic_range') else "❌"
            report += f"| {metric} | {data.get('data_mean')} | {data.get('benchmark_mean')} | {status} |\n"
        
        report += f"""
## Seasonal Patterns

{json.dumps(results.get('seasonal_patterns', {}), indent=2)}

## Conclusion

This synthetic dataset {"demonstrates good alignment with" if results.get('realism_score', {}).get('overall', 0) >= 60 else "shows some deviations from"} real-world e-commerce patterns. The validation provides confidence that experimental results using this data are {"likely generalizable" if results.get('realism_score', {}).get('overall', 0) >= 60 else "should be interpreted with caution regarding generalizability"}.
"""
        
        report_file = self.output_dir / f"dataset_validation_report_{dataset_name}.md"
        report_file.write_text(report, encoding='utf-8')
        print(f"[SAVED] Validation report: {report_file}")
        
        return report


def validate_project_dataset(csv_path: str = None, output_dir: str = "output"):
    """Convenience function to validate the project's dataset."""
    if csv_path is None:
        # Try default locations
        possible_paths = [
            Path("datasets/ecommerce_fashion.csv"),
            Path("../datasets/ecommerce_fashion.csv"),
            Path("ecommerce_fashion.csv"),
        ]
        for p in possible_paths:
            if p.exists():
                csv_path = str(p)
                break
    
    if csv_path is None or not Path(csv_path).exists():
        print("[ERROR] Dataset file not found")
        return None
    
    df = pd.read_csv(csv_path)
    validator = DatasetValidator(output_dir=output_dir)
    results = validator.validate_dataset(df, industry="fashion", dataset_name="ecommerce_fashion")
    validator.generate_validation_report("ecommerce_fashion")
    
    return results


if __name__ == "__main__":
    validate_project_dataset()