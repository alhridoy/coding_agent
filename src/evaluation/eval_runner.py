"""
Main evaluation runner script for autonomous coding agents
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

from .backspace_test_runner import BackspaceTestRunner
from .eval_framework import EvaluationFramework
from .tiered_datasets import TieredDatasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main evaluation entry point"""
    
    parser = argparse.ArgumentParser(description="Autonomous Coding Agent Evaluation Runner")
    
    parser.add_argument(
        "--mode",
        choices=["quick", "comprehensive", "progressive", "performance", "security"],
        default="quick",
        help="Evaluation mode to run"
    )
    
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API base URL for the autonomous coding agent"
    )
    
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--tier",
        choices=["tier1_basic", "tier2_intermediate", "tier3_advanced", "tier4_expert", "performance", "security"],
        help="Run specific tier tests only"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum concurrent test executions"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Initialize test runner
    test_runner = BackspaceTestRunner(
        api_base_url=args.api_url,
        output_dir=args.output_dir
    )
    
    logger.info(f"Starting evaluation in {args.mode} mode")
    logger.info(f"API URL: {args.api_url}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        if args.tier:
            # Run specific tier
            await run_tier_evaluation(test_runner, args.tier, args.max_concurrent)
        elif args.mode == "quick":
            await run_quick_evaluation(test_runner)
        elif args.mode == "comprehensive":
            await run_comprehensive_evaluation(test_runner)
        elif args.mode == "progressive":
            await run_progressive_evaluation(test_runner)
        elif args.mode == "performance":
            await run_performance_evaluation(test_runner)
        elif args.mode == "security":
            await run_security_evaluation(test_runner)
            
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
        
    logger.info("Evaluation completed successfully")


async def run_quick_evaluation(test_runner: BackspaceTestRunner):
    """Run quick evaluation"""
    
    logger.info("Running quick evaluation...")
    
    report = await test_runner.run_quick_evaluation()
    
    # Print summary
    print_evaluation_summary(report, "Quick Evaluation")


async def run_comprehensive_evaluation(test_runner: BackspaceTestRunner):
    """Run comprehensive evaluation"""
    
    logger.info("Running comprehensive evaluation...")
    
    report = await test_runner.run_comprehensive_evaluation()
    
    # Print summary
    print_evaluation_summary(report, "Comprehensive Evaluation")
    
    # Print regression analysis if available
    if "regression_analysis" in report:
        print_regression_summary(report["regression_analysis"])


async def run_progressive_evaluation(test_runner: BackspaceTestRunner):
    """Run progressive evaluation"""
    
    logger.info("Running progressive evaluation...")
    
    report = await test_runner.run_progressive_evaluation()
    
    # Print tier-by-tier results
    print_progressive_summary(report)


async def run_performance_evaluation(test_runner: BackspaceTestRunner):
    """Run performance evaluation"""
    
    logger.info("Running performance evaluation...")
    
    report = await test_runner.run_performance_benchmark()
    
    # Print performance summary
    print_performance_summary(report)


async def run_security_evaluation(test_runner: BackspaceTestRunner):
    """Run security evaluation"""
    
    logger.info("Running security evaluation...")
    
    report = await test_runner.run_security_evaluation()
    
    # Print security summary
    print_security_summary(report)


async def run_tier_evaluation(test_runner: BackspaceTestRunner, tier: str, max_concurrent: int):
    """Run evaluation for specific tier"""
    
    logger.info(f"Running {tier} evaluation...")
    
    # Get test cases for the tier
    datasets = TieredDatasets()
    test_cases = datasets.get_dataset(tier)
    
    if not test_cases:
        logger.error(f"No test cases found for tier: {tier}")
        return
        
    # Run tests
    results = await test_runner.eval_suite.framework.run_parallel_tests(test_cases, max_concurrent)
    
    # Generate report
    report = test_runner.eval_suite._generate_backspace_report(results)
    
    # Print summary
    print_evaluation_summary(report, f"{tier.title()} Evaluation")


def print_evaluation_summary(report: dict, title: str):
    """Print evaluation summary"""
    
    print(f"\n{'='*60}")
    print(f"{title} Results")
    print('='*60)
    
    summary = report.get("summary", {})
    
    print(f"Total Tests: {summary.get('total_tests', 0)}")
    print(f"Passed: {summary.get('passed', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
    print(f"Average Execution Time: {summary.get('average_execution_time', 0):.1f}s")
    
    # Backspace metrics
    backspace_metrics = report.get("backspace_metrics", {})
    if backspace_metrics:
        print(f"\nBackspace Metrics:")
        print(f"  Streaming Compliance: {backspace_metrics.get('streaming_compliance', {}).get('score', 0):.2f}")
        print(f"  PR Quality Score: {backspace_metrics.get('pr_quality_score', 0):.1f}")
        print(f"  Production Ready: {backspace_metrics.get('ready_for_production', False)}")
        
    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
            
    # Tier performance
    tier_performance = report.get("tier_performance", {})
    if tier_performance:
        print(f"\nTier Performance:")
        for tier, metrics in tier_performance.items():
            ready_status = "✅" if metrics.get("ready", False) else "❌"
            print(f"  {tier}: {metrics.get('success_rate', 0):.1f}% success {ready_status}")


def print_regression_summary(regression_report: dict):
    """Print regression analysis summary"""
    
    print(f"\n{'='*60}")
    print("Regression Analysis")
    print('='*60)
    
    summary = regression_report.get("summary", {})
    
    print(f"Tests Analyzed: {summary.get('total_tests_analyzed', 0)}")
    print(f"Regressions Detected: {summary.get('regressions_detected', 0)}")
    print(f"Regression Rate: {summary.get('regression_rate', 0):.1%}")
    
    severity_breakdown = summary.get("severity_breakdown", {})
    if any(severity_breakdown.values()):
        print(f"\nSeverity Breakdown:")
        print(f"  Critical: {severity_breakdown.get('critical', 0)}")
        print(f"  Major: {severity_breakdown.get('major', 0)}")
        print(f"  Minor: {severity_breakdown.get('minor', 0)}")
        
    # Overall health
    health = regression_report.get("overall_health", {})
    if health:
        status_icon = "🟢" if health.get("status") == "healthy" else "🟡" if health.get("status") == "warning" else "🔴"
        print(f"\nOverall Health: {status_icon} {health.get('status', 'unknown').upper()}")
        print(f"  Reason: {health.get('reason', 'No information')}")


def print_progressive_summary(report: dict):
    """Print progressive evaluation summary"""
    
    print(f"\n{'='*60}")
    print("Progressive Evaluation Results")
    print('='*60)
    
    tier_results = report.get("tier_results", {})
    
    for tier_name, metrics in tier_results.items():
        status = "✅ PASS" if metrics.get("meets_threshold", False) else "❌ FAIL"
        print(f"\n{tier_name}:")
        print(f"  Success Rate: {metrics.get('success_rate', 0):.1f}% {status}")
        print(f"  Tests: {metrics.get('successful_tests', 0)}/{metrics.get('total_tests', 0)}")
        print(f"  Avg Time: {metrics.get('average_execution_time', 0):.1f}s")
        print(f"  Quality Score: {metrics.get('average_quality_score', 0):.1f}")
        
    # Progression analysis
    progression = report.get("progression_analysis", {})
    production_ready = progression.get("ready_for_production", {})
    
    print(f"\nProgression Analysis:")
    print(f"  Completed Tiers: {progression.get('completed_tiers', 0)}")
    print(f"  Overall Success: {'✅' if progression.get('overall_success', False) else '❌'}")
    
    ready_icon = "🚀" if production_ready.get("ready", False) else "⚠️"
    print(f"  Production Ready: {ready_icon} {production_ready.get('ready', False)}")
    print(f"  Confidence: {production_ready.get('confidence_level', 'unknown')}")
    
    # Recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


def print_performance_summary(report: dict):
    """Print performance evaluation summary"""
    
    print(f"\n{'='*60}")
    print("Performance Benchmark Results")
    print('='*60)
    
    overall_metrics = report.get("overall_metrics", {})
    
    print(f"Average Execution Time: {overall_metrics.get('avg_execution_time', 0):.1f}s")
    print(f"Success Rate: {overall_metrics.get('success_rate', 0):.1%}")
    print(f"Performance Stability: {overall_metrics.get('performance_stability', 0):.1%}")
    
    # Test performance details
    test_performance = report.get("test_performance", {})
    if test_performance:
        print(f"\nIndividual Test Performance:")
        for test_id, metrics in test_performance.items():
            consistency = "🟢" if metrics.get("consistency_score", 0) > 0.8 else "🟡" if metrics.get("consistency_score", 0) > 0.6 else "🔴"
            print(f"  {test_id}:")
            print(f"    Avg Time: {metrics.get('avg_time', 0):.1f}s")
            print(f"    Success Rate: {metrics.get('success_rate', 0):.1%}")
            print(f"    Consistency: {consistency} {metrics.get('consistency_score', 0):.1%}")


def print_security_summary(report: dict):
    """Print security evaluation summary"""
    
    print(f"\n{'='*60}")
    print("Security Evaluation Results")
    print('='*60)
    
    print(f"Average Security Score: {report.get('average_security_score', 0):.1f}/100")
    print(f"Security Grade: {report.get('security_grade', 'N/A')}")
    print(f"Total Security Issues: {report.get('total_security_issues', 0)}")
    
    risk_distribution = report.get("risk_distribution", {})
    if risk_distribution:
        print(f"\nRisk Distribution:")
        for risk_level, count in risk_distribution.items():
            risk_icon = "🔴" if risk_level == "high_risk" else "🟡" if risk_level == "medium_risk" else "🟢"
            print(f"  {risk_level}: {risk_icon} {count}")
            
    # Security recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        print(f"\nSecurity Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


if __name__ == "__main__":
    asyncio.run(main())