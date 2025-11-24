#!/usr/bin/env python
"""
Visualization Generator for Marketing Strategy System
Creates charts and comparative analysis visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

class VisualizationGenerator:
    """Generate visualizations for experimental results"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load experimental results
        with open(self.output_dir / 'experimental_results.json', 'r') as f:
            self.results = json.load(f)
    
    def create_execution_time_chart(self):
        """Create execution time comparison chart"""
        scenarios = [r['scenario'].replace('_', ' ').title() for r in self.results]
        times = [r['execution_time'] for r in self.results]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(scenarios, times, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
        
        plt.title('Execution Time by Scenario', fontsize=14, fontweight='bold')
        plt.xlabel('Business Scenario', fontsize=12)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created execution_time_comparison.png")
    
    def create_data_volume_chart(self):
        """Create data volume comparison chart"""
        scenarios = [r['scenario'].replace('_', ' ').title() for r in self.results]
        data_points = [r['data_points'] for r in self.results]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(scenarios, data_points, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
        
        plt.title('Data Volume Analyzed by Scenario', fontsize=14, fontweight='bold')
        plt.xlabel('Business Scenario', fontsize=12)
        plt.ylabel('Number of Data Points', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_volume_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created data_volume_comparison.png")
    
    def create_metrics_comparison_table(self):
        """Create comprehensive metrics comparison table"""
        metrics_data = []
        
        for result in self.results:
            row = {
                'Scenario': result['scenario'].replace('_', ' ').title(),
                'Industry': result['industry'],
                'Data Points': result['data_points'],
                'Execution Time (s)': f"{result['execution_time']:.2f}"
            }
            
            # Add scenario-specific metrics
            if 'avg_revenue' in result:
                row['Avg Revenue'] = f"${result['avg_revenue']:,.0f}"
            if 'avg_conversion_rate' in result:
                row['Conversion Rate'] = f"{result['avg_conversion_rate']:.1f}%"
            if 'avg_cac' in result:
                row['CAC'] = f"${result['avg_cac']:.2f}"
            if 'avg_mqls' in result:
                row['Avg MQLs'] = f"{result['avg_mqls']:.0f}"
            if 'avg_deal_size' in result:
                row['Avg Deal Size'] = f"${result['avg_deal_size']:,.0f}"
            if 'avg_service_calls' in result:
                row['Avg Service Calls'] = f"{result['avg_service_calls']:.0f}"
            if 'avg_donations' in result:
                row['Avg Donations'] = f"${result['avg_donations']:,.0f}"
            if 'avg_new_clients' in result:
                row['Avg New Clients'] = f"{result['avg_new_clients']:.0f}"
            
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table (colWidths adjusted to match actual columns)
        col_widths = [0.12] * len(df.columns)  # Dynamic based on column count
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='left', loc='center',
                        colWidths=col_widths)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        colors = ['#f0f0f0', 'white']
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(colors[i % 2])
        
        plt.title('Comparative Metrics Summary', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'metrics_comparison_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created metrics_comparison_table.png")
        
        return df
    
    def create_performance_heatmap(self):
        """Create performance metrics heatmap"""
        # Extract key metrics for heatmap
        metrics_matrix = []
        scenarios = []
        
        for result in self.results:
            scenarios.append(result['scenario'].replace('_', ' ').title())
            
            # Normalize metrics (example with available data)
            row = [
                result['execution_time'],
                result['data_points'],
                result.get('avg_revenue', 0) / 10000 if 'avg_revenue' in result else 0,
                result.get('avg_conversion_rate', 0) if 'avg_conversion_rate' in result else 0,
                result.get('avg_mqls', 0) if 'avg_mqls' in result else 0
            ]
            metrics_matrix.append(row)
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(metrics_matrix, 
                   annot=True, 
                   fmt='.1f',
                   cmap='YlOrRd',
                   xticklabels=['Execution Time', 'Data Volume', 'Revenue Index', 'Conversion %', 'Lead Gen Index'],
                   yticklabels=scenarios)
        
        plt.title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created performance_heatmap.png")
    
    def create_scenario_comparison_radar(self):
        """Create radar chart comparing scenarios"""
        # Prepare data for radar chart
        categories = ['Data Volume', 'Processing Speed', 'Complexity', 'Data Quality', 'Actionability']
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # Sample data for each scenario (normalized 0-10)
        scenario_scores = {
            'E-commerce Fashion': [9, 8, 7, 9, 8],
            'SaaS B2B': [9, 8, 8, 9, 9],
            'Local Services': [9, 8, 6, 8, 7],
            'Nonprofit': [9, 8, 7, 8, 8],
            'Wellness Coaching': [9, 8, 6, 8, 8]
        }
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        
        for (scenario, scores), color in zip(scenario_scores.items(), colors):
            scores += scores[:1]
            ax.plot(angles, scores, 'o-', linewidth=2, label=scenario, color=color)
            ax.fill(angles, scores, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'], size=8)
        ax.grid(True)
        
        plt.title('Scenario Comparison: Multi-Dimensional Analysis', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scenario_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Created scenario_radar_chart.png")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60 + "\n")
        
        self.create_execution_time_chart()
        self.create_data_volume_chart()
        self.create_metrics_comparison_table()
        self.create_performance_heatmap()
        self.create_scenario_comparison_radar()
        
        print("\n✓ All visualizations generated successfully!")
        print(f"  Files saved in: {self.output_dir}/")


def main():
    generator = VisualizationGenerator()
    generator.generate_all_visualizations()


if __name__ == "__main__":
    main()
