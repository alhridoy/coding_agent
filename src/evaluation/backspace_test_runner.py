"""
Test runner specifically for Backspace evaluation scenarios
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from .eval_framework import EvaluationFramework, TestCase, EvaluationResult
from .backspace_eval_suite import BackspaceEvaluationSuite
from .enhanced_evaluators import CodeQualityEvaluator, PRQualityEvaluator, WorkflowEvaluator
from .regression_detection import RegressionDetector
from .tiered_datasets import TieredDatasets

logger = logging.getLogger(__name__)


class BackspaceTestRunner:
    """Comprehensive test runner for Backspace evaluation scenarios"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", output_dir: str = "evaluation_results"):
        self.api_base_url = api_base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.eval_suite = BackspaceEvaluationSuite(api_base_url)
        self.datasets = TieredDatasets()
        self.regression_detector = RegressionDetector()
        
        # Initialize evaluators
        self.code_evaluator = CodeQualityEvaluator()
        self.pr_evaluator = PRQualityEvaluator()
        self.workflow_evaluator = WorkflowEvaluator()
        
        # Add evaluators to framework
        self.eval_suite.framework.add_evaluator(self._evaluate_code_quality)
        self.eval_suite.framework.add_evaluator(self._evaluate_pr_quality)
        self.eval_suite.framework.add_evaluator(self._evaluate_workflow)
        
    async def _evaluate_code_quality(self, result: EvaluationResult):
        """Evaluate code quality"""
        await self.code_evaluator.evaluate(result)
        
    async def _evaluate_pr_quality(self, result: EvaluationResult):
        """Evaluate PR quality"""
        await self.pr_evaluator.evaluate(result)
        
    async def _evaluate_workflow(self, result: EvaluationResult):
        """Evaluate workflow execution"""
        workflow_metrics = await self.workflow_evaluator.evaluate(result)
        result.artifacts.update(workflow_metrics)
        
    async def run_quick_evaluation(self) -> Dict[str, Any]:
        """Run quick evaluation with basic test cases"""
        
        logger.info("Starting quick Backspace evaluation")
        
        # Get a small set of basic tests
        basic_tests = self.datasets.get_random_sample("tier1_basic", 3)
        
        # Run tests
        results = await self.eval_suite.framework.run_test_suite(basic_tests)
        
        # Generate report
        report = self.eval_suite._generate_backspace_report(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"quick_eval_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Quick evaluation completed. Report saved to {report_file}")
        
        return report
        
    async def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation across all tiers"""
        
        logger.info("Starting comprehensive Backspace evaluation")
        
        # Get test cases from all tiers
        all_tests = []
        all_tests.extend(self.datasets.get_dataset("tier1_basic"))
        all_tests.extend(self.datasets.get_dataset("tier2_intermediate"))
        all_tests.extend(self.datasets.get_random_sample("tier3_advanced", 2))  # Limited for time
        all_tests.extend(self.datasets.get_dataset("performance"))
        
        logger.info(f"Running {len(all_tests)} test cases")
        
        # Run tests with concurrency
        results = await self.eval_suite.framework.run_parallel_tests(all_tests, max_concurrent=2)
        
        # Generate comprehensive report
        report = self.eval_suite._generate_backspace_report(results)
        
        # Add detailed analysis
        report["detailed_analysis"] = await self._generate_detailed_analysis(results)
        
        # Detect regressions if baselines exist
        regression_results = self.regression_detector.detect_regressions(results)
        if regression_results:
            report["regression_analysis"] = self.regression_detector.generate_regression_report(regression_results)
            
        # Update baselines with current results
        self.regression_detector.update_baseline(results, version=f"eval_{int(time.time())}")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"comprehensive_eval_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Comprehensive evaluation completed. Report saved to {report_file}")
        
        return report
        
    async def run_progressive_evaluation(self) -> Dict[str, Any]:
        """Run progressive evaluation starting from basic to advanced"""
        
        logger.info("Starting progressive Backspace evaluation")
        
        progressive_report = {
            "tier_results": {},
            "progression_analysis": {},
            "recommendations": []
        }
        
        # Define tier progression
        tiers = [
            ("tier1_basic", 3),
            ("tier2_intermediate", 2), 
            ("tier3_advanced", 1)
        ]
        
        overall_success = True
        
        for tier_name, sample_size in tiers:
            logger.info(f"Evaluating {tier_name} (sample size: {sample_size})")
            
            # Get test cases for this tier
            test_cases = self.datasets.get_random_sample(tier_name, sample_size)
            
            # Run tests for this tier
            tier_results = await self.eval_suite.framework.run_test_suite(test_cases)
            
            # Analyze tier performance
            tier_analysis = self._analyze_tier_performance(tier_results, tier_name)
            progressive_report["tier_results"][tier_name] = tier_analysis
            
            # Check if we should continue to next tier
            if tier_analysis["success_rate"] < 0.7:  # 70% threshold
                logger.warning(f"Low success rate in {tier_name}: {tier_analysis['success_rate']:.2f}")
                progressive_report["recommendations"].append(
                    f"Focus on improving {tier_name} before advancing to harder tiers"
                )
                overall_success = False
                break
                
        # Generate progression analysis
        progressive_report["progression_analysis"] = {
            "completed_tiers": len(progressive_report["tier_results"]),
            "overall_success": overall_success,
            "ready_for_production": self._assess_production_readiness(progressive_report["tier_results"])
        }
        
        # Save progressive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"progressive_eval_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(progressive_report, f, indent=2, default=str)
            
        logger.info(f"Progressive evaluation completed. Report saved to {report_file}")
        
        return progressive_report
        
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run performance-focused benchmark tests"""
        
        logger.info("Starting performance benchmark")
        
        # Get performance test cases
        perf_tests = self.datasets.get_dataset("performance")
        
        # Run tests multiple times for statistical significance
        all_results = []
        
        for run in range(3):  # 3 runs for averaging
            logger.info(f"Performance run {run + 1}/3")
            
            run_results = await self.eval_suite.framework.run_test_suite(perf_tests)
            all_results.extend(run_results)
            
        # Analyze performance metrics
        perf_analysis = self._analyze_performance_metrics(all_results)
        
        # Save performance results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"performance_benchmark_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(perf_analysis, f, indent=2, default=str)
            
        logger.info(f"Performance benchmark completed. Report saved to {report_file}")
        
        return perf_analysis
        
    async def run_security_evaluation(self) -> Dict[str, Any]:
        """Run security-focused evaluation"""
        
        logger.info("Starting security evaluation")
        
        # Get security test cases
        security_tests = self.datasets.get_dataset("security")
        
        # Run security tests
        results = await self.eval_suite.framework.run_test_suite(security_tests)
        
        # Analyze security aspects
        security_analysis = await self._analyze_security_results(results)
        
        # Save security results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"security_eval_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(security_analysis, f, indent=2, default=str)
            
        logger.info(f"Security evaluation completed. Report saved to {report_file}")
        
        return security_analysis
        
    def _analyze_tier_performance(self, results: List[EvaluationResult], tier_name: str) -> Dict[str, Any]:
        """Analyze performance for a specific tier"""
        
        if not results:
            return {"success_rate": 0, "tier": tier_name, "error": "No results"}
            
        successful = [r for r in results if r.pr_url is not None]
        success_rate = len(successful) / len(results)
        
        avg_time = sum(r.metrics.execution_time for r in results) / len(results)
        avg_quality = sum(r.metrics.code_quality_score for r in results) / len(results)
        total_errors = sum(r.metrics.error_count for r in results)
        
        tier_metadata = self.datasets.get_dataset_metadata(tier_name)
        expected_success_rate = tier_metadata.success_rate_threshold if tier_metadata else 0.8
        
        return {
            "tier": tier_name,
            "total_tests": len(results),
            "successful_tests": len(successful),
            "success_rate": success_rate,
            "meets_threshold": success_rate >= expected_success_rate,
            "average_execution_time": avg_time,
            "average_quality_score": avg_quality,
            "total_errors": total_errors,
            "expected_threshold": expected_success_rate
        }
        
    def _assess_production_readiness(self, tier_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if system is ready for production based on tier performance"""
        
        if not tier_results:
            return {"ready": False, "reason": "No evaluation data"}
            
        # Check basic tier performance
        basic_performance = tier_results.get("tier1_basic", {})
        intermediate_performance = tier_results.get("tier2_intermediate", {})
        
        basic_success = basic_performance.get("success_rate", 0) >= 0.90
        intermediate_success = intermediate_performance.get("success_rate", 0) >= 0.75
        
        if not basic_success:
            return {
                "ready": False,
                "reason": "Basic functionality not reliable enough for production",
                "basic_success_rate": basic_performance.get("success_rate", 0)
            }
            
        if not intermediate_success and intermediate_performance:
            return {
                "ready": False,
                "reason": "Intermediate functionality needs improvement",
                "intermediate_success_rate": intermediate_performance.get("success_rate", 0)
            }
            
        return {
            "ready": True,
            "reason": "System meets production readiness criteria",
            "confidence_level": "high" if basic_success and intermediate_success else "medium"
        }
        
    async def _generate_detailed_analysis(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate detailed analysis of results"""
        
        analysis = {
            "code_quality_analysis": await self._analyze_code_quality_trends(results),
            "pr_quality_analysis": await self._analyze_pr_quality_trends(results),
            "workflow_efficiency": await self._analyze_workflow_efficiency(results),
            "error_patterns": self._analyze_error_patterns(results),
            "performance_insights": self._analyze_performance_insights(results)
        }
        
        return analysis
        
    async def _analyze_code_quality_trends(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze code quality trends"""
        
        quality_scores = [r.metrics.code_quality_score for r in results if r.metrics.code_quality_score > 0]
        
        if not quality_scores:
            return {"trend": "no_data", "average": 0}
            
        return {
            "average_score": sum(quality_scores) / len(quality_scores),
            "min_score": min(quality_scores),
            "max_score": max(quality_scores),
            "trend": "stable",  # Would need historical data for actual trend
            "distribution": {
                "excellent": len([s for s in quality_scores if s >= 90]),
                "good": len([s for s in quality_scores if 70 <= s < 90]),
                "fair": len([s for s in quality_scores if 50 <= s < 70]),
                "poor": len([s for s in quality_scores if s < 50])
            }
        }
        
    async def _analyze_pr_quality_trends(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze PR quality trends"""
        
        pr_scores = [r.metrics.pr_quality_score for r in results if r.metrics.pr_quality_score > 0]
        
        if not pr_scores:
            return {"trend": "no_data", "average": 0}
            
        return {
            "average_score": sum(pr_scores) / len(pr_scores),
            "prs_created": len([r for r in results if r.pr_url]),
            "pr_success_rate": len([r for r in results if r.pr_url]) / len(results),
            "quality_distribution": {
                "high": len([s for s in pr_scores if s >= 80]),
                "medium": len([s for s in pr_scores if 60 <= s < 80]),
                "low": len([s for s in pr_scores if s < 60])
            }
        }
        
    async def _analyze_workflow_efficiency(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze workflow efficiency"""
        
        execution_times = [r.metrics.execution_time for r in results]
        
        return {
            "average_time": sum(execution_times) / len(execution_times),
            "median_time": sorted(execution_times)[len(execution_times) // 2],
            "time_distribution": {
                "fast": len([t for t in execution_times if t < 300]),  # Under 5 min
                "moderate": len([t for t in execution_times if 300 <= t < 600]),  # 5-10 min
                "slow": len([t for t in execution_times if t >= 600])  # Over 10 min
            },
            "efficiency_score": len([t for t in execution_times if t < 300]) / len(execution_times) * 100
        }
        
    def _analyze_error_patterns(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze common error patterns"""
        
        error_results = [r for r in results if r.metrics.error_count > 0]
        total_errors = sum(r.metrics.error_count for r in results)
        
        # Analyze error messages for patterns
        error_types = {}
        for result in error_results:
            if result.error_message:
                # Simple categorization based on error message
                if "timeout" in result.error_message.lower():
                    error_types["timeout"] = error_types.get("timeout", 0) + 1
                elif "auth" in result.error_message.lower():
                    error_types["authentication"] = error_types.get("authentication", 0) + 1
                elif "network" in result.error_message.lower():
                    error_types["network"] = error_types.get("network", 0) + 1
                else:
                    error_types["other"] = error_types.get("other", 0) + 1
                    
        return {
            "total_errors": total_errors,
            "error_rate": total_errors / len(results),
            "tests_with_errors": len(error_results),
            "error_distribution": error_types,
            "most_common_error": max(error_types.items(), key=lambda x: x[1]) if error_types else None
        }
        
    def _analyze_performance_insights(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze performance insights"""
        
        # Group by difficulty
        difficulty_groups = {}
        for result in results:
            difficulty = result.test_case.difficulty
            if difficulty not in difficulty_groups:
                difficulty_groups[difficulty] = []
            difficulty_groups[difficulty].append(result)
            
        difficulty_analysis = {}
        for difficulty, group_results in difficulty_groups.items():
            success_rate = len([r for r in group_results if r.pr_url]) / len(group_results)
            avg_time = sum(r.metrics.execution_time for r in group_results) / len(group_results)
            
            difficulty_analysis[difficulty] = {
                "count": len(group_results),
                "success_rate": success_rate,
                "average_time": avg_time
            }
            
        return {
            "difficulty_analysis": difficulty_analysis,
            "scalability_insights": {
                "handles_simple_tasks": difficulty_analysis.get("easy", {}).get("success_rate", 0) > 0.9,
                "handles_complex_tasks": difficulty_analysis.get("hard", {}).get("success_rate", 0) > 0.6,
                "performance_degradation": self._calculate_performance_degradation(difficulty_analysis)
            }
        }
        
    def _calculate_performance_degradation(self, difficulty_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance degradation across difficulty levels"""
        
        easy_success = difficulty_analysis.get("easy", {}).get("success_rate", 0)
        hard_success = difficulty_analysis.get("hard", {}).get("success_rate", 0)
        
        if easy_success == 0:
            return {"degradation": "unknown", "reason": "No easy test data"}
            
        degradation = (easy_success - hard_success) / easy_success if hard_success > 0 else 1.0
        
        return {
            "degradation_factor": degradation,
            "severity": "high" if degradation > 0.5 else "moderate" if degradation > 0.3 else "low",
            "easy_success_rate": easy_success,
            "hard_success_rate": hard_success
        }
        
    def _analyze_performance_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze performance metrics from multiple runs"""
        
        # Group by test case
        test_groups = {}
        for result in results:
            test_id = result.test_case.id
            if test_id not in test_groups:
                test_groups[test_id] = []
            test_groups[test_id].append(result)
            
        performance_analysis = {}
        
        for test_id, test_results in test_groups.items():
            times = [r.metrics.execution_time for r in test_results]
            success_rates = [1 if r.pr_url else 0 for r in test_results]
            
            performance_analysis[test_id] = {
                "runs": len(test_results),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "time_variance": max(times) - min(times),
                "success_rate": sum(success_rates) / len(success_rates),
                "consistency_score": 1 - (max(times) - min(times)) / max(times) if max(times) > 0 else 0
            }
            
        return {
            "test_performance": performance_analysis,
            "overall_metrics": {
                "avg_execution_time": sum(r.metrics.execution_time for r in results) / len(results),
                "success_rate": len([r for r in results if r.pr_url]) / len(results),
                "performance_stability": sum(analysis["consistency_score"] for analysis in performance_analysis.values()) / len(performance_analysis)
            }
        }
        
    async def _analyze_security_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze security evaluation results"""
        
        from .enhanced_evaluators import SecurityEvaluator
        
        security_evaluator = SecurityEvaluator()
        security_results = []
        
        for result in results:
            security_eval = await security_evaluator.evaluate(result)
            security_results.append(security_eval)
            
        # Aggregate security metrics
        avg_security_score = sum(sr["security_score"] for sr in security_results) / len(security_results)
        
        all_issues = []
        for sr in security_results:
            all_issues.extend(sr["issues_found"])
            
        risk_distribution = {}
        for issue in all_issues:
            risk_level = issue["risk_level"]
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
            
        return {
            "average_security_score": avg_security_score,
            "total_security_issues": len(all_issues),
            "risk_distribution": risk_distribution,
            "security_grade": "A" if avg_security_score >= 90 else "B" if avg_security_score >= 80 else "C" if avg_security_score >= 70 else "D",
            "recommendations": self._generate_security_recommendations(security_results)
        }
        
    def _generate_security_recommendations(self, security_results: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations"""
        
        recommendations = []
        
        high_risk_count = sum(len([issue for issue in sr["issues_found"] if issue["risk_level"] == "high_risk"]) for sr in security_results)
        
        if high_risk_count > 0:
            recommendations.append(f"Address {high_risk_count} high-risk security issues immediately")
            
        avg_score = sum(sr["security_score"] for sr in security_results) / len(security_results)
        
        if avg_score < 80:
            recommendations.append("Improve overall security practices in code generation")
            
        if not recommendations:
            recommendations.append("Security posture is good - maintain current practices")
            
        return recommendations