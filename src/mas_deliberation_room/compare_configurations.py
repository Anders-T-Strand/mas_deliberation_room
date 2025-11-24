#!/usr/bin/env python
"""
Configuration Comparison Tool
Compares single-agent vs multi-agent performance

Usage:
    python compare_configurations.py single_agent_baseline.json multi_agent_results.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np


class ConfigurationComparator:
    """Compare performance between different AI configurations"""
    
    def __init__(self, baseline_path: str, comparison_path: str):
        with open(baseline_path, 'r') as f:
            self.baseline = json.load(f)
        
        with open(comparison_path, 'r') as f:
            self.comparison = json.load(f)
        
        self.baseline_name = Path(baseline_path).stem.replace('_', ' ').title()
        self.comparison_name = Path(comparison_path).stem.replace('_', ' ').title()
    
    def compare_performance(self) -> Dict[str, Any]:
        """Compare execution time and efficiency"""
        baseline_perf = self.baseline['performance_comparison']
        comparison_perf = self.comparison['performance_comparison']
        
        results = {
            'by_scenario': {},
            'overall': {}
        }
        
        # Compare each scenario
        for scenario in baseline_perf.keys():
            if scenario in comparison_perf:
                baseline_time = baseline_perf[scenario]['avg_execution_time']
                comparison_time = comparison_perf[scenario]['avg_execution_time']
                
                speedup = baseline_time / comparison_time if comparison_time > 0 else 0
                pct_change = ((comparison_time - baseline_time) / baseline_time) * 100
                
                results['by_scenario'][scenario] = {
                    'baseline_time': baseline_time,
                    'comparison_time': comparison_time,
                    'speedup_factor': speedup,
                    'percent_change': pct_change,
                    'faster': comparison_time < baseline_time
                }
        
        # Overall statistics
        all_baseline_times = [v['avg_execution_time'] for v in baseline_perf.values()]
        all_comparison_times = [v['avg_execution_time'] for v in comparison_perf.values()]
        
        results['overall'] = {
            'baseline_avg': np.mean(all_baseline_times),
            'comparison_avg': np.mean(all_comparison_times),
            'overall_speedup': np.mean(all_baseline_times) / np.mean(all_comparison_times),
            'overall_percent_change': ((np.mean(all_comparison_times) - np.mean(all_baseline_times)) / 
                                      np.mean(all_baseline_times)) * 100
        }
        
        return results
    
    def compare_quality(self) -> Dict[str, Any]:
        """Compare output quality metrics"""
        baseline_qual = self.baseline['quality_comparison']
        comparison_qual = self.comparison['quality_comparison']
        
        results = {
            'by_scenario': {},
            'overall': {}
        }
        
        # Compare each scenario
        for scenario in baseline_qual.keys():
            if scenario in comparison_qual:
                baseline_score = baseline_qual[scenario]['avg_quality_score']
                comparison_score = comparison_qual[scenario]['avg_quality_score']
                
                improvement = comparison_score - baseline_score
                pct_improvement = (improvement / baseline_score) * 100 if baseline_score > 0 else 0
                
                results['by_scenario'][scenario] = {
                    'baseline_quality': baseline_score,
                    'comparison_quality': comparison_score,
                    'absolute_improvement': improvement,
                    'percent_improvement': pct_improvement,
                    'better': comparison_score > baseline_score,
                    'baseline_breakdown': {
                        'kpi': baseline_qual[scenario]['avg_kpi_score'],
                        'specificity': baseline_qual[scenario]['avg_specificity_score'],
                        'actionability': baseline_qual[scenario]['avg_actionability_score']
                    },
                    'comparison_breakdown': {
                        'kpi': comparison_qual[scenario]['avg_kpi_score'],
                        'specificity': comparison_qual[scenario]['avg_specificity_score'],
                        'actionability': comparison_qual[scenario]['avg_actionability_score']
                    }
                }
        
        # Overall statistics
        all_baseline_qual = [v['avg_quality_score'] for v in baseline_qual.values()]
        all_comparison_qual = [v['avg_quality_score'] for v in comparison_qual.values()]
        
        results['overall'] = {
            'baseline_avg': np.mean(all_baseline_qual),
            'comparison_avg': np.mean(all_comparison_qual),
            'overall_improvement': np.mean(all_comparison_qual) - np.mean(all_baseline_qual),
            'overall_percent_improvement': ((np.mean(all_comparison_qual) - np.mean(all_baseline_qual)) / 
                                           np.mean(all_baseline_qual)) * 100
        }
        
        return results
    
    def analyze_tradeoffs(self) -> Dict[str, Any]:
        """Analyze speed vs quality tradeoffs"""
        perf_results = self.compare_performance()
        qual_results = self.compare_quality()
        
        tradeoffs = []
        
        for scenario in perf_results['by_scenario'].keys():
            perf = perf_results['by_scenario'][scenario]
            qual = qual_results['by_scenario'][scenario]
            
            # Classify tradeoff
            if perf['faster'] and qual['better']:
                tradeoff_type = "WIN-WIN: Faster AND Better"
            elif not perf['faster'] and not qual['better']:
                tradeoff_type = "LOSE-LOSE: Slower AND Worse"
            elif perf['faster'] and not qual['better']:
                tradeoff_type = "Speed over Quality"
            else:  # slower but better
                tradeoff_type = "Quality over Speed"
            
            # Calculate efficiency score (quality per second)
            baseline_efficiency = qual['baseline_quality'] / perf['baseline_time']
            comparison_efficiency = qual['comparison_quality'] / perf['comparison_time']
            
            tradeoffs.append({
                'scenario': scenario,
                'tradeoff_type': tradeoff_type,
                'time_change_pct': perf['percent_change'],
                'quality_change_pct': qual['percent_improvement'],
                'baseline_efficiency': baseline_efficiency,
                'comparison_efficiency': comparison_efficiency,
                'efficiency_improvement': ((comparison_efficiency - baseline_efficiency) / 
                                         baseline_efficiency) * 100
            })
        
        # Overall tradeoff assessment
        avg_time_change = perf_results['overall']['overall_percent_change']
        avg_qual_change = qual_results['overall']['overall_percent_improvement']
        
        if avg_time_change < 0 and avg_qual_change > 0:
            overall_assessment = "CLEAR WIN: Faster with better quality"
        elif avg_time_change > 0 and avg_qual_change < 0:
            overall_assessment = "CLEAR LOSS: Slower with worse quality"
        elif avg_time_change < 0 and avg_qual_change < 0:
            overall_assessment = "Speed gains, but quality suffers"
        else:
            overall_assessment = "Quality gains, but takes longer"
        
        return {
            'by_scenario': tradeoffs,
            'overall_assessment': overall_assessment,
            'avg_time_change': avg_time_change,
            'avg_quality_change': avg_qual_change
        }
    
    def generate_report(self, output_path: str = "output/configuration_comparison.txt"):
        """Generate comprehensive comparison report"""
        
        perf = self.compare_performance()
        qual = self.compare_quality()
        tradeoffs = self.analyze_tradeoffs()
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CONFIGURATION COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Baseline:   {self.baseline_name}\n")
            f.write(f"Comparison: {self.comparison_name}\n\n")
            
            # Performance Comparison
            f.write("="*80 + "\n")
            f.write("PERFORMANCE COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Scenario':<30} {'Baseline':<15} {'Comparison':<15} {'Change':<15}\n")
            f.write("-"*80 + "\n")
            
            for scenario, data in perf['by_scenario'].items():
                f.write(f"{scenario:<30} "
                       f"{data['baseline_time']:<15.2f} "
                       f"{data['comparison_time']:<15.2f} "
                       f"{data['percent_change']:>+14.1f}%\n")
            
            f.write("-"*80 + "\n")
            f.write(f"{'AVERAGE':<30} "
                   f"{perf['overall']['baseline_avg']:<15.2f} "
                   f"{perf['overall']['comparison_avg']:<15.2f} "
                   f"{perf['overall']['overall_percent_change']:>+14.1f}%\n\n")
            
            # Quality Comparison
            f.write("="*80 + "\n")
            f.write("QUALITY COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'Scenario':<30} {'Baseline':<15} {'Comparison':<15} {'Change':<15}\n")
            f.write("-"*80 + "\n")
            
            for scenario, data in qual['by_scenario'].items():
                f.write(f"{scenario:<30} "
                       f"{data['baseline_quality']:<15.2f} "
                       f"{data['comparison_quality']:<15.2f} "
                       f"{data['percent_improvement']:>+14.1f}%\n")
            
            f.write("-"*80 + "\n")
            f.write(f"{'AVERAGE':<30} "
                   f"{qual['overall']['baseline_avg']:<15.2f} "
                   f"{qual['overall']['comparison_avg']:<15.2f} "
                   f"{qual['overall']['overall_percent_improvement']:>+14.1f}%\n\n")
            
            # Detailed Quality Breakdown
            f.write("="*80 + "\n")
            f.write("QUALITY METRICS BREAKDOWN\n")
            f.write("="*80 + "\n\n")
            
            for scenario, data in qual['by_scenario'].items():
                f.write(f"{scenario}:\n")
                f.write(f"  KPI Score:         {data['baseline_breakdown']['kpi']:.2f} → "
                       f"{data['comparison_breakdown']['kpi']:.2f} "
                       f"({data['comparison_breakdown']['kpi'] - data['baseline_breakdown']['kpi']:+.2f})\n")
                f.write(f"  Specificity:       {data['baseline_breakdown']['specificity']:.2f} → "
                       f"{data['comparison_breakdown']['specificity']:.2f} "
                       f"({data['comparison_breakdown']['specificity'] - data['baseline_breakdown']['specificity']:+.2f})\n")
                f.write(f"  Actionability:     {data['baseline_breakdown']['actionability']:.2f} → "
                       f"{data['comparison_breakdown']['actionability']:.2f} "
                       f"({data['comparison_breakdown']['actionability'] - data['baseline_breakdown']['actionability']:+.2f})\n\n")
            
            # Tradeoff Analysis
            f.write("="*80 + "\n")
            f.write("TRADEOFF ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Overall Assessment: {tradeoffs['overall_assessment']}\n")
            f.write(f"  Average Time Change:    {tradeoffs['avg_time_change']:+.1f}%\n")
            f.write(f"  Average Quality Change: {tradeoffs['avg_quality_change']:+.1f}%\n\n")
            
            f.write(f"{'Scenario':<30} {'Tradeoff Type':<25} {'Efficiency Δ':<15}\n")
            f.write("-"*80 + "\n")
            
            for trade in tradeoffs['by_scenario']:
                f.write(f"{trade['scenario']:<30} "
                       f"{trade['tradeoff_type']:<25} "
                       f"{trade['efficiency_improvement']:>+14.1f}%\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("="*80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*80 + "\n\n")
            
            recommendations = self._generate_recommendations(perf, qual, tradeoffs)
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Comparison report saved to {output_path}")
        return output_path
    
    def _generate_recommendations(self, perf, qual, tradeoffs) -> list:
        """Generate actionable recommendations"""
        recs = []
        
        # Performance recommendation
        if perf['overall']['overall_percent_change'] < -10:
            recs.append(f"{self.comparison_name} is significantly faster ({perf['overall']['overall_percent_change']:.1f}% reduction). "
                       f"Consider using for time-critical scenarios.")
        elif perf['overall']['overall_percent_change'] > 10:
            recs.append(f"{self.comparison_name} is slower ({perf['overall']['overall_percent_change']:+.1f}%). "
                       f"Evaluate if quality gains justify the time cost.")
        
        # Quality recommendation
        if qual['overall']['overall_percent_improvement'] > 10:
            recs.append(f"{self.comparison_name} produces significantly better quality ({qual['overall']['overall_percent_improvement']:+.1f}%). "
                       f"Strong candidate for production use.")
        elif qual['overall']['overall_percent_improvement'] < -5:
            recs.append(f"Warning: {self.comparison_name} shows quality degradation ({qual['overall']['overall_percent_improvement']:.1f}%). "
                       f"Investigate cause before deployment.")
        
        # Tradeoff recommendation
        if tradeoffs['overall_assessment'].startswith("CLEAR WIN"):
            recs.append(f"Clear winner: {self.comparison_name} improves both speed and quality. Recommended for adoption.")
        elif tradeoffs['overall_assessment'].startswith("CLEAR LOSS"):
            recs.append(f"Regression detected: {self.comparison_name} underperforms on both metrics. Not recommended.")
        else:
            recs.append(f"Tradeoff exists: Evaluate business priorities (speed vs quality) to choose configuration.")
        
        # Scenario-specific recommendations
        win_scenarios = [t for t in tradeoffs['by_scenario'] if t['tradeoff_type'] == "WIN-WIN: Faster AND Better"]
        if win_scenarios:
            recs.append(f"Best performance in: {', '.join([t['scenario'] for t in win_scenarios])}")
        
        return recs
    
    def generate_visualization_data(self) -> Dict:
        """Generate data structure for visualization"""
        perf = self.compare_performance()
        qual = self.compare_quality()
        
        viz_data = {
            'scenarios': list(perf['by_scenario'].keys()),
            'baseline_times': [v['baseline_time'] for v in perf['by_scenario'].values()],
            'comparison_times': [v['comparison_time'] for v in perf['by_scenario'].values()],
            'baseline_quality': [v['baseline_quality'] for v in qual['by_scenario'].values()],
            'comparison_quality': [v['comparison_quality'] for v in qual['by_scenario'].values()]
        }
        
        return viz_data


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_configurations.py <baseline.json> <comparison.json>")
        print("\nExample:")
        print("  python compare_configurations.py single_agent_baseline.json multi_agent_results.json")
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    comparison_path = sys.argv[2]
    
    if not Path(baseline_path).exists():
        print(f"Error: Baseline file not found: {baseline_path}")
        sys.exit(1)
    
    if not Path(comparison_path).exists():
        print(f"Error: Comparison file not found: {comparison_path}")
        sys.exit(1)
    
    print("="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    print()
    
    comparator = ConfigurationComparator(baseline_path, comparison_path)
    
    # Generate comprehensive report
    report_path = comparator.generate_report()
    
    # Display key findings
    perf = comparator.compare_performance()
    qual = comparator.compare_quality()
    tradeoffs = comparator.analyze_tradeoffs()
    
    print("\nKEY FINDINGS:")
    print("-"*80)
    print(f"Performance: {perf['overall']['overall_percent_change']:+.1f}% time change")
    print(f"Quality:     {qual['overall']['overall_percent_improvement']:+.1f}% quality change")
    print(f"Assessment:  {tradeoffs['overall_assessment']}")
    print()
    print(f"Full report: {report_path}")
    print("="*80)


if __name__ == "__main__":
    main()
