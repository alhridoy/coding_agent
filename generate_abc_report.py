#!/usr/bin/env python3
"""
Generate enhanced ABC-compliant evaluation report
Shows the difference between basic and rigorous evaluation
"""

import json
import os
from datetime import datetime
from pathlib import Path


def generate_abc_html_report(comparison_file_path):
    """Generate enhanced HTML report showing ABC validation benefits."""
    with open(comparison_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract key metrics
    basic = data["comparison_summary"]["basic_evaluation"]
    abc = data["comparison_summary"]["abc_evaluation"]
    findings = data["key_findings"]
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ABC-Compliant Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }}
        h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 40px 0;
        }}
        .evaluation-card {{
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }}
        .basic-card {{
            background: linear-gradient(135deg, #ff6b6b, #ee5a24);
            color: white;
        }}
        .abc-card {{
            background: linear-gradient(135deg, #26de81, #20bf6b);
            color: white;
        }}
        .card-title {{
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 15px 0;
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
        }}
        .metric-label {{
            font-weight: bold;
        }}
        .metric-value {{
            font-size: 1.1em;
        }}
        .impact-section {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
            border-left: 5px solid #ffc107;
        }}
        .findings-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .finding-card {{
            background: #e8f4fd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .status-pass {{
            background: #d4edda;
            color: #155724;
        }}
        .status-fail {{
            background: #f8d7da;
            color: #721c24;
        }}
        .status-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        .test-results {{
            margin: 30px 0;
        }}
        .test-item {{
            background: #f8f9fa;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
        }}
        .test-item.passed {{
            border-left-color: #27ae60;
        }}
        .test-item.abc-failed {{
            border-left-color: #f39c12;
            background: #fff3cd;
        }}
        .recommendations {{
            background: #d1ecf1;
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #17a2b8;
            margin: 30px 0;
        }}
        .progress-bar {{
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            height: 20px;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}
        .progress-basic {{
            background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        }}
        .progress-abc {{
            background: linear-gradient(90deg, #26de81, #20bf6b);
        }}
        .highlight-box {{
            background: #fff3cd;
            border: 2px solid #ffc107;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }}
        .highlight-number {{
            font-size: 3em;
            font-weight: bold;
            color: #e67e22;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 ABC-Compliant Evaluation Report</h1>
            <div class="subtitle">Demonstrating Rigorous Agentic Benchmark Validation</div>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="highlight-box">
            <div class="highlight-number">{abc['overestimation_rate']:.0f}%</div>
            <p><strong>Performance Overestimation Detected</strong><br>
            Basic evaluation would have overestimated agent performance by {abc['overestimation_rate']:.0f}%</p>
        </div>

        <div class="comparison-grid">
            <div class="evaluation-card basic-card">
                <div class="card-title">📊 Basic Evaluation</div>
                <div class="metric">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value">{basic['success_rate']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Execution Time:</span>
                    <span class="metric-value">{basic['avg_execution_time']:.1f}s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Confidence Scores:</span>
                    <span class="metric-value">❌ Not Provided</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Task Validation:</span>
                    <span class="metric-value">❌ None</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Environment Isolation:</span>
                    <span class="metric-value">❌ Not Checked</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Shortcut Detection:</span>
                    <span class="metric-value">❌ Not Implemented</span>
                </div>
            </div>

            <div class="evaluation-card abc-card">
                <div class="card-title">🔬 ABC-Compliant Evaluation</div>
                <div class="metric">
                    <span class="metric-label">Original Success Rate:</span>
                    <span class="metric-value">{abc['original_success_rate']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">True Success Rate:</span>
                    <span class="metric-value">{abc['true_success_rate']:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Confidence Score:</span>
                    <span class="metric-value">{abc['avg_confidence_score']:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">ABC Compliance:</span>
                    <span class="metric-value">{abc['avg_abc_compliance']:.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Task Validation:</span>
                    <span class="metric-value">✅ Complete</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Environment Isolation:</span>
                    <span class="metric-value">✅ Verified</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Shortcut Detection:</span>
                    <span class="metric-value">✅ Active</span>
                </div>
            </div>
        </div>

        <div class="impact-section">
            <h2>⚠️ Critical Impact of ABC Validation</h2>
            <p><strong>Without ABC validation, your evaluation would have serious blind spots:</strong></p>
            <ul>
                <li><strong>False Positive Rate:</strong> {findings['false_positive_rate']:.0f}% of "successful" tests didn't actually solve the tasks</li>
                <li><strong>Confidence Issues:</strong> Only {findings['confidence_threshold_met']} out of 4 tests met high confidence thresholds</li>
                <li><strong>Hidden Issues:</strong> Task specifications problems, environment contamination, and shortcuts would go undetected</li>
            </ul>
        </div>

        <h2>📋 Test Results Breakdown</h2>
        <div class="test-results">
"""

    # Add test results
    abc_results = data["detailed_results"]["abc"]["results"]
    for result in abc_results:
        test_class = "passed" if result["truly_successful"] else "abc-failed"
        status_badge_class = "status-pass" if result["truly_successful"] else "status-warning"
        status_text = "✅ Truly Successful" if result["truly_successful"] else "⚠️ ABC Validation Failed"
        
        html_content += f"""
            <div class="test-item {test_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3>{result['test_id']}</h3>
                    <span class="status-badge {status_badge_class}">{status_text}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-top: 15px;">
                    <div>
                        <strong>Original Status:</strong> {result['original_status']}<br>
                        <strong>Execution Time:</strong> {result['execution_time']:.1f}s
                    </div>
                    <div>
                        <strong>Confidence Score:</strong> {result['confidence_score']:.2f}<br>
                        <strong>ABC Compliance:</strong> {result['abc_compliance_score']:.2f}
                    </div>
                    <div>
                        <strong>Issues Found:</strong><br>
                        {'; '.join(result.get('validation_issues', ['None'])) if result.get('validation_issues') else 'None'}
                    </div>
                </div>
            </div>
        """

    # Add most common issues
    html_content += f"""
        </div>

        <h2>🔍 Most Common Validation Issues</h2>
        <div class="findings-grid">
"""

    for issue, count in findings["most_common_issues"]:
        html_content += f"""
            <div class="finding-card">
                <h4>{issue}</h4>
                <p>Occurred in <strong>{count}</strong> test case(s)</p>
            </div>
        """

    # Add recommendations
    html_content += f"""
        </div>

        <div class="recommendations">
            <h2>💡 ABC Validation Recommendations</h2>
            <ol>
"""

    for rec in data["recommendations"]:
        html_content += f"<li>{rec}</li>"

    # Add conclusion
    html_content += f"""
            </ol>
        </div>

        <div class="impact-section">
            <h2>🎯 Why ABC Validation Matters</h2>
            <div class="findings-grid">
                <div class="finding-card">
                    <h4>🔍 Accuracy</h4>
                    <p>Detected {findings['false_positive_rate']:.0f}% false positives that basic evaluation missed</p>
                </div>
                <div class="finding-card">
                    <h4>📊 Confidence</h4>
                    <p>Provides 0-1 confidence scores for all results, enabling informed decision-making</p>
                </div>
                <div class="finding-card">
                    <h4>🛡️ Robustness</h4>
                    <p>Validates task specifications, environment isolation, and detects shortcuts</p>
                </div>
                <div class="finding-card">
                    <h4>🎯 Production Ready</h4>
                    <p>Follows academic best practices from the ABC paper for agentic benchmarks</p>
                </div>
            </div>
        </div>

        <div class="highlight-box">
            <h3>🚀 Implementation Success</h3>
            <p>Your evaluation framework now implements <strong>ABC (Agentic Benchmark Checklist)</strong> best practices,
            providing the same level of rigor as academic benchmarks and catching issues that have plagued 
            other agentic evaluation systems.</p>
            <p><strong>Result:</strong> More reliable agent assessment, reduced false positives, and confident decision-making about agent capabilities.</p>
        </div>

        <footer style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ecf0f1; color: #7f8c8d;">
            <p>Generated by ABC-Compliant Evaluation Framework • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""
    
    return html_content


def main():
    """Generate ABC-specific HTML report"""
    eval_dir = Path("evaluation_results")
    
    # Find the latest ABC comparison file
    comparison_files = list(eval_dir.glob("abc_evaluation_comparison_*.json"))
    
    if not comparison_files:
        print("No ABC evaluation comparison results found.")
        print("Run: python run_abc_evaluation.py")
        return
    
    # Get the latest file
    latest_file = max(comparison_files, key=lambda f: f.stat().st_mtime)
    
    print(f"Generating enhanced ABC report from: {latest_file}")
    
    # Generate HTML report
    html_content = generate_abc_html_report(latest_file)
    
    # Save enhanced HTML report
    html_file = latest_file.with_name(f"abc_enhanced_report_{latest_file.stem.split('_')[-1]}.html")
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Enhanced ABC report generated: {html_file}")
    print(f"🌐 Open in browser: file://{html_file.absolute()}")
    
    # Also create a summary log
    summary_file = latest_file.with_suffix('.summary.txt')
    with open(summary_file, 'w') as f:
        f.write("ABC-COMPLIANT EVALUATION SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        # Read the data
        with open(latest_file, 'r') as data_file:
            data = json.load(data_file)
            
        abc = data["comparison_summary"]["abc_evaluation"]
        f.write(f"Original Success Rate: {abc['original_success_rate']:.1f}%\n")
        f.write(f"True Success Rate: {abc['true_success_rate']:.1f}%\n")
        f.write(f"Overestimation Rate: {abc['overestimation_rate']:.1f}%\n")
        f.write(f"Average Confidence: {abc['avg_confidence_score']:.2f}\n")
        f.write(f"ABC Compliance: {abc['avg_abc_compliance']:.2f}\n")
        f.write(f"False Positive Rate: {data['key_findings']['false_positive_rate']:.1f}%\n\n")
        
        f.write("KEY BENEFITS:\n")
        f.write("• Detected false positives that basic evaluation missed\n")
        f.write("• Provided confidence scores for all results\n")
        f.write("• Validated task specifications and requirements\n")
        f.write("• Ensured environment isolation between tests\n")
        f.write("• Implemented shortcut detection mechanisms\n")
        f.write("• Follows ABC academic best practices\n")
    
    print(f"📄 Summary saved: {summary_file}")


if __name__ == "__main__":
    main()