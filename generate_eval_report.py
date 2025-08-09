#!/usr/bin/env python3
"""Generate HTML report from evaluation results following lightweight eval-driven development principles."""

import json
import os
from datetime import datetime
from pathlib import Path


def generate_html_report(json_file_path):
    """Generate a visual HTML report from evaluation JSON results."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract timestamp from filename
    filename = os.path.basename(json_file_path)
    timestamp = filename.replace('quick_eval_', '').replace('.json', '')
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report - {timestamp}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .status-pass {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .test-result {{
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #e74c3c;
        }}
        .test-result.passed {{
            border-left-color: #27ae60;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ecf0f1;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #2c3e50;
        }}
        .recommendation {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
        }}
        .tier-indicator {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
        }}
        .tier-ready {{
            background: #d4edda;
            color: #155724;
        }}
        .tier-not-ready {{
            background: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Autonomous Coding Agent Evaluation Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Summary Metrics</h2>
        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-value">{data['summary']['total_tests']}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: #27ae60">{data['summary']['passed']}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: #e74c3c">{data['summary']['failed']}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['summary']['success_rate']:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['summary']['average_execution_time']:.1f}s</div>
                <div class="metric-label">Avg Execution Time</div>
            </div>
        </div>
        
        <h2>🏆 Backspace Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Streaming Compliance</td>
                <td>{data['backspace_metrics']['streaming_compliance'].get('score', 0)}</td>
                <td><span class="{'status-pass' if data['backspace_metrics']['streaming_compliance'].get('score', 0) >= 80 else 'status-fail'}">
                    {'✓ Ready' if data['backspace_metrics']['streaming_compliance'].get('score', 0) >= 80 else '✗ Not Ready'}
                </span></td>
            </tr>
            <tr>
                <td>PR Quality Score</td>
                <td>{data['backspace_metrics']['pr_quality_score']}</td>
                <td><span class="{'status-pass' if data['backspace_metrics']['pr_quality_score'] >= 60 else 'status-fail'}">
                    {'✓ Ready' if data['backspace_metrics']['pr_quality_score'] >= 60 else '✗ Not Ready'}
                </span></td>
            </tr>
            <tr>
                <td>Specification Adherence</td>
                <td>{data['backspace_metrics']['specification_adherence']['overall_adherence']}%</td>
                <td><span class="{'status-pass' if data['backspace_metrics']['specification_adherence']['overall_adherence'] >= 80 else 'status-fail'}">
                    {'✓ Ready' if data['backspace_metrics']['specification_adherence']['overall_adherence'] >= 80 else '✗ Not Ready'}
                </span></td>
            </tr>
            <tr>
                <td><strong>Production Ready</strong></td>
                <td colspan="2"><span class="{'status-pass' if data['backspace_metrics']['ready_for_production'] else 'status-fail'}">
                    {'✓ YES' if data['backspace_metrics']['ready_for_production'] else '✗ NO'}
                </span></td>
            </tr>
        </table>
        
        <h2>🧪 Test Results</h2>
"""
    
    # Add individual test results
    for result in data['results']:
        status_class = 'passed' if result['status'] == 'passed' else ''
        html_content += f"""
        <div class="test-result {status_class}">
            <h3>{result['test_id']}</h3>
            <table>
                <tr>
                    <td><strong>Status:</strong></td>
                    <td><span class="status-{result['status']}">{result['status'].upper()}</span></td>
                </tr>
                <tr>
                    <td><strong>Execution Time:</strong></td>
                    <td>{result['execution_time']:.2f} seconds</td>
                </tr>
                <tr>
                    <td><strong>PR URL:</strong></td>
                    <td>{result['pr_url'] if result['pr_url'] else 'None'}</td>
                </tr>
                {f'<tr><td><strong>Error:</strong></td><td>{result["error"]}</td></tr>' if result.get('error') else ''}
            </table>
        </div>
"""
    
    # Add tier performance
    html_content += """
        <h2>📈 Tier Performance</h2>
        <table>
            <tr>
                <th>Tier</th>
                <th>Tests</th>
                <th>Success Rate</th>
                <th>Avg Time</th>
                <th>Status</th>
            </tr>
"""
    
    for tier, perf in data.get('tier_performance', {}).items():
        ready_class = 'tier-ready' if perf['ready'] else 'tier-not-ready'
        ready_text = '✓ Ready' if perf['ready'] else '✗ Not Ready'
        html_content += f"""
            <tr>
                <td>{tier.upper()}</td>
                <td>{perf['count']}</td>
                <td>{perf['success_rate']:.1f}%</td>
                <td>{perf['average_time']:.1f}s</td>
                <td><span class="{ready_class}">{ready_text}</span></td>
            </tr>
"""
    
    html_content += """
        </table>
        
        <h2>💡 Recommendations</h2>
"""
    
    # Add recommendations
    for rec in data['recommendations']:
        html_content += f"""
        <div class="recommendation">
            {rec}
        </div>
"""
    
    html_content += """
    </div>
</body>
</html>
"""
    
    return html_content


def main():
    """Generate HTML reports for all evaluation results."""
    eval_dir = Path("evaluation_results")
    
    if not eval_dir.exists():
        print("No evaluation results directory found.")
        return
    
    # Find all JSON evaluation files
    json_files = list(eval_dir.glob("*.json"))
    
    if not json_files:
        print("No evaluation results found.")
        return
    
    print(f"Found {len(json_files)} evaluation result(s)")
    
    # Generate HTML report for each JSON file
    for json_file in json_files:
        html_content = generate_html_report(json_file)
        
        # Save HTML report
        html_file = json_file.with_suffix('.html')
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"Generated HTML report: {html_file}")
        print(f"Open in browser: file://{html_file.absolute()}")


if __name__ == "__main__":
    main()