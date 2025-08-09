"""
Backspace-specific evaluation suite for autonomous coding agents
"""

import asyncio
import logging
from typing import List, Dict, Any
from .eval_framework import EvaluationFramework, TestCase, EvaluationResult
from .tiered_datasets import TieredDatasets

logger = logging.getLogger(__name__)


class BackspaceEvaluationSuite:
    """Evaluation suite specifically designed for Backspace requirements"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.framework = EvaluationFramework(api_base_url)
        self.datasets = TieredDatasets()
        
    def get_backspace_test_cases(self) -> List[TestCase]:
        """Get test cases that match Backspace specification requirements"""
        
        return [
            # Tier 1: Basic functionality tests
            TestCase(
                id="backspace_basic_readme",
                name="Basic README Enhancement",
                description="Add basic documentation to a simple repository",
                repo_url="https://github.com/octocat/Hello-World",
                prompt="Add a proper project description and usage instructions to the README",
                expected_outcome={
                    "pr_created": True,
                    "files_modified": ["README", "README.md"],
                    "contains_description": True
                },
                tags=["basic", "documentation", "readme"],
                difficulty="easy"
            ),
            
            TestCase(
                id="backspace_api_validation",
                name="API Input Validation",
                description="Add input validation to API endpoints",
                repo_url="https://github.com/fastapi/fastapi",
                prompt="Add input validation to all POST endpoints and return proper error messages",
                expected_outcome={
                    "pr_created": True,
                    "validation_added": True,
                    "error_handling": True
                },
                tags=["api", "validation", "backend"],
                difficulty="medium",
                timeout_seconds=600
            ),
            
            TestCase(
                id="backspace_health_check",
                name="Health Check Endpoint",
                description="Add health check endpoint to API",
                repo_url="https://github.com/pallets/flask",
                prompt="Add a health check endpoint that returns API status and version",
                expected_outcome={
                    "pr_created": True,
                    "endpoint_added": True,
                    "returns_status": True
                },
                tags=["api", "health", "monitoring"],
                difficulty="easy"
            ),
            
            TestCase(
                id="backspace_error_handling",
                name="Comprehensive Error Handling",
                description="Add error handling and logging",
                repo_url="https://github.com/express/express",
                prompt="Add comprehensive error handling and request logging middleware",
                expected_outcome={
                    "pr_created": True,
                    "error_handling": True,
                    "logging_added": True
                },
                tags=["error-handling", "logging", "middleware"],
                difficulty="medium"
            ),
            
            TestCase(
                id="backspace_security_headers",
                name="Security Headers",
                description="Add security headers to web application",
                repo_url="https://github.com/django/django",
                prompt="Add security headers middleware for CORS, CSP, and HSTS",
                expected_outcome={
                    "pr_created": True,
                    "security_headers": True,
                    "cors_configured": True
                },
                tags=["security", "headers", "web"],
                difficulty="hard",
                timeout_seconds=900
            ),
            
            # Tier 2: Advanced functionality tests
            TestCase(
                id="backspace_database_connection",
                name="Database Connection Helper",
                description="Add database connection and helper functions",
                repo_url="https://github.com/sqlalchemy/sqlalchemy",
                prompt="Add database connection helper with retry logic and connection pooling",
                expected_outcome={
                    "pr_created": True,
                    "db_helper": True,
                    "retry_logic": True,
                    "connection_pooling": True
                },
                tags=["database", "connection", "backend"],
                difficulty="hard"
            ),
            
            TestCase(
                id="backspace_test_coverage",
                name="Test Coverage Improvement",
                description="Add unit tests to improve coverage",
                repo_url="https://github.com/pytest-dev/pytest",
                prompt="Add unit tests for core functionality to improve test coverage",
                expected_outcome={
                    "pr_created": True,
                    "tests_added": True,
                    "coverage_improved": True
                },
                tags=["testing", "coverage", "quality"],
                difficulty="medium"
            ),
            
            # Tier 3: Streaming API specification tests
            TestCase(
                id="backspace_streaming_format",
                name="Streaming API Format Compliance",
                description="Verify streaming API matches Backspace specification",
                repo_url="https://github.com/fastapi/fastapi",
                prompt="Add Server-Sent Events streaming endpoint with proper format",
                expected_outcome={
                    "pr_created": True,
                    "sse_format": True,
                    "proper_headers": True,
                    "event_streaming": True
                },
                tags=["streaming", "sse", "api", "spec-compliance"],
                difficulty="hard"
            )
        ]
        
    def get_performance_benchmarks(self) -> List[TestCase]:
        """Get performance benchmark test cases"""
        
        return [
            TestCase(
                id="perf_execution_time",
                name="Execution Time Benchmark",
                description="Measure execution time for simple changes",
                repo_url="https://github.com/octocat/Hello-World",
                prompt="Add a simple comment to the README file",
                expected_outcome={
                    "execution_time_under": 60,  # seconds
                    "pr_created": True
                },
                tags=["performance", "benchmark", "speed"],
                difficulty="easy",
                timeout_seconds=120
            ),
            
            TestCase(
                id="perf_concurrent_requests",
                name="Concurrent Request Handling",
                description="Test handling multiple concurrent requests",
                repo_url="https://github.com/octocat/Hello-World",
                prompt="Add project badges to README",
                expected_outcome={
                    "handles_concurrency": True,
                    "no_resource_conflicts": True
                },
                tags=["performance", "concurrency", "scalability"],
                difficulty="medium"
            )
        ]
        
    async def run_backspace_evaluation(self, include_performance: bool = True) -> Dict[str, Any]:
        """Run complete Backspace evaluation suite"""
        
        logger.info("Starting Backspace evaluation suite")
        
        # Get test cases
        basic_tests = self.get_backspace_test_cases()
        performance_tests = self.get_performance_benchmarks() if include_performance else []
        
        all_tests = basic_tests + performance_tests
        
        # Run tests
        results = await self.framework.run_parallel_tests(all_tests, max_concurrent=2)
        
        # Generate Backspace-specific report
        report = self._generate_backspace_report(results)
        
        return report
        
    def _generate_backspace_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate Backspace-specific evaluation report"""
        
        # Basic metrics
        basic_report = self.framework.generate_report()
        
        # Backspace-specific metrics
        streaming_compliance = self._check_streaming_compliance(results)
        pr_quality = self._evaluate_pr_quality(results)
        specification_adherence = self._check_specification_adherence(results)
        
        backspace_report = {
            **basic_report,
            "backspace_metrics": {
                "streaming_compliance": streaming_compliance,
                "pr_quality_score": pr_quality,
                "specification_adherence": specification_adherence,
                "ready_for_production": self._is_production_ready(results)
            },
            "recommendations": self._generate_recommendations(results),
            "tier_performance": self._analyze_tier_performance(results)
        }
        
        return backspace_report
        
    def _check_streaming_compliance(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Check compliance with Backspace streaming specification"""
        
        streaming_tests = [r for r in results if "streaming" in r.test_case.tags]
        
        if not streaming_tests:
            return {"score": 0, "reason": "No streaming tests executed"}
            
        compliant_count = 0
        total_events = 0
        
        for result in streaming_tests:
            events = result.artifacts.get("total_events", 0)
            total_events += events
            
            # Check for proper event types
            expected_events = ["Status", "Tool: Read", "Tool: Edit", "Tool: Bash"]
            has_expected_events = any(
                event_type in " ".join(result.logs) for event_type in expected_events
            )
            
            if has_expected_events and events > 5:
                compliant_count += 1
                
        compliance_score = compliant_count / len(streaming_tests) if streaming_tests else 0
        
        return {
            "score": compliance_score,
            "compliant_tests": compliant_count,
            "total_streaming_tests": len(streaming_tests),
            "average_events_per_test": total_events / len(streaming_tests) if streaming_tests else 0
        }
        
    def _evaluate_pr_quality(self, results: List[EvaluationResult]) -> float:
        """Evaluate overall PR quality score"""
        
        pr_results = [r for r in results if r.pr_url]
        
        if not pr_results:
            return 0.0
            
        quality_scores = []
        
        for result in pr_results:
            score = 0.0
            
            # Basic PR creation (40 points)
            if result.pr_url:
                score += 40
                
            # Files modified appropriately (20 points)
            if result.metrics.files_modified > 0:
                score += 20
                
            # No critical errors (20 points)
            if result.metrics.error_count == 0:
                score += 20
                
            # Reasonable execution time (10 points)
            if result.metrics.execution_time < 300:  # 5 minutes
                score += 10
                
            # Low warning count (10 points)
            if result.metrics.warnings_count <= 2:
                score += 10
                
            quality_scores.append(score)
            
        return sum(quality_scores) / len(quality_scores)
        
    def _check_specification_adherence(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Check adherence to Backspace specification"""
        
        total_tests = len(results)
        
        # Required capabilities
        has_repo_cloning = any("cloning" in " ".join(r.logs).lower() for r in results)
        has_file_modification = any(r.metrics.files_modified > 0 for r in results)
        has_git_operations = any("git" in " ".join(r.logs).lower() for r in results)
        has_pr_creation = any(r.pr_url for r in results)
        
        capabilities_score = sum([
            has_repo_cloning,
            has_file_modification,
            has_git_operations,
            has_pr_creation
        ]) / 4
        
        return {
            "capabilities_score": capabilities_score,
            "has_repo_cloning": has_repo_cloning,
            "has_file_modification": has_file_modification,
            "has_git_operations": has_git_operations,
            "has_pr_creation": has_pr_creation,
            "overall_adherence": capabilities_score * 100
        }
        
    def _is_production_ready(self, results: List[EvaluationResult]) -> bool:
        """Determine if the system is production ready"""
        
        if not results:
            return False
            
        success_rate = sum(1 for r in results if r.pr_url) / len(results)
        avg_execution_time = sum(r.metrics.execution_time for r in results) / len(results)
        error_rate = sum(r.metrics.error_count for r in results) / len(results)
        
        return (
            success_rate >= 0.8 and  # 80% success rate
            avg_execution_time < 600 and  # Under 10 minutes average
            error_rate < 2  # Less than 2 errors per test on average
        )
        
    def _generate_recommendations(self, results: List[EvaluationResult]) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        failed_tests = [r for r in results if r.pr_url is None]
        slow_tests = [r for r in results if r.metrics.execution_time > 300]
        error_prone_tests = [r for r in results if r.metrics.error_count > 1]
        
        if failed_tests:
            recommendations.append(f"Improve reliability: {len(failed_tests)} tests failed to create PRs")
            
        if slow_tests:
            recommendations.append(f"Optimize performance: {len(slow_tests)} tests took over 5 minutes")
            
        if error_prone_tests:
            recommendations.append(f"Enhance error handling: {len(error_prone_tests)} tests had multiple errors")
            
        # Success rate analysis
        success_rate = sum(1 for r in results if r.pr_url) / len(results) if results else 0
        if success_rate < 0.9:
            recommendations.append("Increase success rate to >90% for production readiness")
            
        if not recommendations:
            recommendations.append("System performing well - ready for production!")
            
        return recommendations
        
    def _analyze_tier_performance(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze performance by difficulty tier"""
        
        tiers = {"easy": [], "medium": [], "hard": []}
        
        for result in results:
            difficulty = result.test_case.difficulty
            if difficulty in tiers:
                tiers[difficulty].append(result)
                
        tier_analysis = {}
        
        for tier, tier_results in tiers.items():
            if tier_results:
                success_rate = sum(1 for r in tier_results if r.pr_url) / len(tier_results)
                avg_time = sum(r.metrics.execution_time for r in tier_results) / len(tier_results)
                
                tier_analysis[tier] = {
                    "count": len(tier_results),
                    "success_rate": success_rate,
                    "average_time": avg_time,
                    "ready": success_rate >= 0.8
                }
                
        return tier_analysis