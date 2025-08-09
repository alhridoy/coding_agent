"""
ABC-Compliant Evaluation Framework
Integrates all validators for rigorous agentic benchmark evaluation
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .eval_framework import EvaluationFramework, TestCase, EvaluationResult
from .pr_validator import PRValidator, PRValidationResult
from .task_validator import TaskSpecificationValidator, TaskValidationResult
from .environment_validator import EnvironmentValidator, EnvironmentValidationResult
from .shortcut_detector import ShortcutDetectionSystem, ShortcutDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class ABCValidationResult:
    """Comprehensive validation result following ABC principles"""
    test_case: TestCase
    original_result: EvaluationResult
    
    # Task validity
    task_validation: Optional[TaskValidationResult] = None
    
    # Outcome validity  
    pr_validation: Optional[PRValidationResult] = None
    
    # Environment isolation
    env_validation: Optional[EnvironmentValidationResult] = None
    
    # Shortcut detection
    shortcut_detection: Optional[ShortcutDetectionResult] = None
    
    # Overall assessment
    is_truly_successful: bool = False
    confidence_score: float = 0.0
    abc_compliance_score: float = 0.0
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Generate detailed validation report"""
        return {
            "test_id": self.test_case.id,
            "original_status": self.original_result.status,
            "original_pr_url": self.original_result.pr_url,
            "truly_successful": self.is_truly_successful,
            "confidence_score": self.confidence_score,
            "abc_compliance_score": self.abc_compliance_score,
            "validations": {
                "task_valid": self.task_validation.is_valid if self.task_validation else None,
                "task_solvable": self.task_validation.is_solvable if self.task_validation else None,
                "pr_valid": self.pr_validation.is_valid if self.pr_validation else None,
                "env_isolated": self.env_validation.is_isolated if self.env_validation else None,
                "no_shortcuts": self.shortcut_detection.is_clean if self.shortcut_detection else None
            },
            "issues": self._collect_all_issues(),
            "recommendations": self._collect_all_recommendations()
        }
    
    def _collect_all_issues(self) -> List[str]:
        """Collect all issues from all validators"""
        issues = []
        
        if self.task_validation:
            issues.extend(self.task_validation.ambiguity_issues)
            issues.extend(self.task_validation.missing_specifications)
        
        if self.pr_validation:
            issues.extend(self.pr_validation.issues_found)
        
        if self.env_validation:
            issues.extend([v.description for v in self.env_validation.violations])
        
        if self.shortcut_detection:
            issues.extend([v.description for v in self.shortcut_detection.violations])
        
        return issues
    
    def _collect_all_recommendations(self) -> List[str]:
        """Collect all recommendations from all validators"""
        recommendations = []
        
        if self.task_validation:
            if not self.task_validation.is_valid:
                recommendations.append("Fix task specification issues")
        
        if self.pr_validation:
            recommendations.extend(self.pr_validation.suggestions)
        
        if self.env_validation:
            recommendations.extend([v.recommendation for v in self.env_validation.violations])
        
        if self.shortcut_detection:
            recommendations.extend([v.recommendation for v in self.shortcut_detection.violations])
        
        return list(set(recommendations))  # Remove duplicates


class ABCCompliantEvaluationFramework(EvaluationFramework):
    """
    Enhanced evaluation framework that implements ABC (Agentic Benchmark Checklist) principles
    for rigorous evaluation of autonomous coding agents
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000", github_token: Optional[str] = None):
        super().__init__(api_base_url)
        
        # Initialize validators
        self.pr_validator = PRValidator(github_token)
        self.task_validator = TaskSpecificationValidator()
        self.env_validator = EnvironmentValidator()
        self.shortcut_detector = ShortcutDetectionSystem()
        
        # Validation results storage
        self.abc_results: List[ABCValidationResult] = []
        
        # Configuration
        self.validation_config = {
            "validate_tasks": True,
            "validate_prs": True,
            "validate_environment": True,
            "detect_shortcuts": True,
            "require_task_validity": True,
            "minimum_confidence_score": 0.7,
            "minimum_abc_compliance": 0.8
        }
    
    async def run_abc_test_case(self, test_case: TestCase) -> ABCValidationResult:
        """Run a test case with full ABC validation"""
        logger.info(f"Starting ABC-compliant evaluation for test case: {test_case.id}")
        
        # Step 1: Validate task specification before execution
        task_validation = None
        if self.validation_config["validate_tasks"]:
            task_validation = await self.task_validator.validate_task(test_case)
            
            if self.validation_config["require_task_validity"] and not task_validation.is_valid:
                logger.warning(f"Task {test_case.id} failed task validity check")
                return ABCValidationResult(
                    test_case=test_case,
                    original_result=self._create_failed_result(test_case, "Task specification invalid"),
                    task_validation=task_validation,
                    is_truly_successful=False,
                    confidence_score=0.0,
                    abc_compliance_score=0.0
                )
        
        # Step 2: Capture environment state before execution
        env_state_before = None
        if self.validation_config["validate_environment"]:
            session_id = f"test_{test_case.id}_{int(time.time())}"
            env_state_before = await self.env_validator.capture_environment_state(session_id)
        
        # Step 3: Execute the original test
        original_result = await super().run_test_case(test_case)
        
        # Step 4: Capture environment state after execution
        env_state_after = None
        env_validation = None
        if self.validation_config["validate_environment"] and env_state_before:
            env_state_after = await self.env_validator.capture_environment_state(session_id)
            env_validation = await self.env_validator.validate_isolation(
                session_id, test_case, env_state_before, env_state_after
            )
        
        # Step 5: Validate PR if created
        pr_validation = None
        if self.validation_config["validate_prs"] and original_result.pr_url:
            pr_validation = await self.pr_validator.validate_pr(
                original_result.pr_url,
                test_case,
                original_result.logs
            )
        
        # Step 6: Detect shortcuts
        shortcut_detection = None
        if self.validation_config["detect_shortcuts"]:
            shortcut_detection = await self.shortcut_detector.analyze_for_shortcuts(
                test_case,
                original_result,
                original_result.logs
            )
        
        # Step 7: Calculate overall assessment
        abc_result = self._calculate_abc_assessment(
            test_case=test_case,
            original_result=original_result,
            task_validation=task_validation,
            pr_validation=pr_validation,
            env_validation=env_validation,
            shortcut_detection=shortcut_detection
        )
        
        self.abc_results.append(abc_result)
        logger.info(f"ABC evaluation completed for {test_case.id}: truly_successful={abc_result.is_truly_successful}")
        
        return abc_result
    
    def _calculate_abc_assessment(
        self,
        test_case: TestCase,
        original_result: EvaluationResult,
        task_validation: Optional[TaskValidationResult],
        pr_validation: Optional[PRValidationResult],
        env_validation: Optional[EnvironmentValidationResult],
        shortcut_detection: Optional[ShortcutDetectionResult]
    ) -> ABCValidationResult:
        """Calculate comprehensive ABC assessment"""
        
        # Start with original result
        base_success = original_result.status.value == 'passed'
        
        # Apply ABC validation criteria
        validation_checks = {
            "task_valid": task_validation.is_valid if task_validation else True,
            "task_solvable": task_validation.is_solvable if task_validation else True,
            "pr_valid": pr_validation.is_valid if pr_validation else base_success,
            "env_isolated": env_validation.is_isolated if env_validation else True,
            "env_clean": env_validation.is_clean if env_validation else True,
            "no_shortcuts": shortcut_detection.is_clean if shortcut_detection else True
        }
        
        # Calculate confidence score
        confidence_components = []
        
        if task_validation:
            confidence_components.append(task_validation.specificity_score)
        
        if pr_validation:
            confidence_components.append(pr_validation.confidence_score)
        
        if env_validation:
            confidence_components.append(env_validation.isolation_score)
        
        if shortcut_detection:
            confidence_components.append(1.0 - shortcut_detection.suspicion_score)
        
        confidence_score = sum(confidence_components) / len(confidence_components) if confidence_components else 0.5
        
        # Calculate ABC compliance score
        compliance_score = sum(validation_checks.values()) / len(validation_checks)
        
        # Determine if truly successful
        is_truly_successful = (
            base_success and
            all(validation_checks.values()) and
            confidence_score >= self.validation_config["minimum_confidence_score"] and
            compliance_score >= self.validation_config["minimum_abc_compliance"]
        )
        
        # Create validation summary
        validation_summary = {
            "base_success": base_success,
            "validation_checks": validation_checks,
            "confidence_score": confidence_score,
            "compliance_score": compliance_score,
            "meets_minimum_confidence": confidence_score >= self.validation_config["minimum_confidence_score"],
            "meets_minimum_compliance": compliance_score >= self.validation_config["minimum_abc_compliance"]
        }
        
        return ABCValidationResult(
            test_case=test_case,
            original_result=original_result,
            task_validation=task_validation,
            pr_validation=pr_validation,
            env_validation=env_validation,
            shortcut_detection=shortcut_detection,
            is_truly_successful=is_truly_successful,
            confidence_score=confidence_score,
            abc_compliance_score=compliance_score,
            validation_summary=validation_summary
        )
    
    def _create_failed_result(self, test_case: TestCase, reason: str) -> EvaluationResult:
        """Create a failed evaluation result"""
        from .eval_framework import TestStatus, EvaluationMetrics
        
        return EvaluationResult(
            test_case=test_case,
            status=TestStatus.FAILED,
            metrics=EvaluationMetrics(),
            start_time=time.time(),
            end_time=time.time(),
            error_message=reason
        )
    
    async def run_abc_test_suite(self, test_cases: List[TestCase]) -> List[ABCValidationResult]:
        """Run a complete test suite with ABC validation"""
        logger.info(f"Running ABC-compliant test suite with {len(test_cases)} test cases")
        
        results = []
        
        for test_case in test_cases:
            try:
                result = await self.run_abc_test_case(test_case)
                results.append(result)
                
                # Log progress with ABC metrics
                logger.info(
                    f"Completed {test_case.id}: "
                    f"original={result.original_result.status.value}, "
                    f"truly_successful={result.is_truly_successful}, "
                    f"confidence={result.confidence_score:.2f}, "
                    f"compliance={result.abc_compliance_score:.2f}"
                )
                
            except Exception as e:
                logger.error(f"Error running ABC test {test_case.id}: {e}")
                # Create failed result
                failed_result = ABCValidationResult(
                    test_case=test_case,
                    original_result=self._create_failed_result(test_case, str(e)),
                    is_truly_successful=False,
                    confidence_score=0.0,
                    abc_compliance_score=0.0
                )
                results.append(failed_result)
        
        return results
    
    def generate_abc_report(self) -> Dict[str, Any]:
        """Generate comprehensive ABC compliance report"""
        if not self.abc_results:
            return {"error": "No ABC results to report"}
        
        total_tests = len(self.abc_results)
        
        # Basic metrics
        original_passed = sum(1 for r in self.abc_results if r.original_result.status.value == 'passed')
        truly_successful = sum(1 for r in self.abc_results if r.is_truly_successful)
        
        # ABC compliance metrics
        avg_confidence = sum(r.confidence_score for r in self.abc_results) / total_tests
        avg_compliance = sum(r.abc_compliance_score for r in self.abc_results) / total_tests
        
        # Validation breakdown
        validation_stats = {
            "task_validity": {
                "total_validated": sum(1 for r in self.abc_results if r.task_validation),
                "valid": sum(1 for r in self.abc_results if r.task_validation and r.task_validation.is_valid),
                "solvable": sum(1 for r in self.abc_results if r.task_validation and r.task_validation.is_solvable)
            },
            "pr_validity": {
                "total_validated": sum(1 for r in self.abc_results if r.pr_validation),
                "valid": sum(1 for r in self.abc_results if r.pr_validation and r.pr_validation.is_valid)
            },
            "environment": {
                "total_validated": sum(1 for r in self.abc_results if r.env_validation),
                "isolated": sum(1 for r in self.abc_results if r.env_validation and r.env_validation.is_isolated),
                "clean": sum(1 for r in self.abc_results if r.env_validation and r.env_validation.is_clean)
            },
            "shortcuts": {
                "total_checked": sum(1 for r in self.abc_results if r.shortcut_detection),
                "clean": sum(1 for r in self.abc_results if r.shortcut_detection and r.shortcut_detection.is_clean)
            }
        }
        
        # Impact analysis
        overestimation_rate = (original_passed - truly_successful) / original_passed if original_passed > 0 else 0
        
        # Collect all issues
        all_issues = []
        for result in self.abc_results:
            all_issues.extend(result._collect_all_issues())
        
        issue_frequency = {}
        for issue in all_issues:
            issue_frequency[issue] = issue_frequency.get(issue, 0) + 1
        
        # Most common issues
        most_common_issues = sorted(issue_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "original_passed": original_passed,
                "truly_successful": truly_successful,
                "original_success_rate": original_passed / total_tests,
                "true_success_rate": truly_successful / total_tests,
                "overestimation_rate": overestimation_rate,
                "average_confidence_score": avg_confidence,
                "average_compliance_score": avg_compliance
            },
            "abc_metrics": {
                "meets_confidence_threshold": sum(
                    1 for r in self.abc_results 
                    if r.confidence_score >= self.validation_config["minimum_confidence_score"]
                ) / total_tests,
                "meets_compliance_threshold": sum(
                    1 for r in self.abc_results 
                    if r.abc_compliance_score >= self.validation_config["minimum_abc_compliance"]
                ) / total_tests
            },
            "validation_breakdown": validation_stats,
            "quality_analysis": {
                "most_common_issues": most_common_issues,
                "total_unique_issues": len(issue_frequency),
                "avg_issues_per_test": len(all_issues) / total_tests
            },
            "recommendations": self._generate_abc_recommendations(validation_stats, overestimation_rate)
        }
        
        return report
    
    def _generate_abc_recommendations(
        self, 
        validation_stats: Dict[str, Any], 
        overestimation_rate: float
    ) -> List[str]:
        """Generate recommendations based on ABC analysis"""
        recommendations = []
        
        # Overestimation warning
        if overestimation_rate > 0.1:
            recommendations.append(
                f"CRITICAL: Performance overestimated by {overestimation_rate:.1%} - "
                "implement ABC validation immediately"
            )
        
        # Task validity
        task_stats = validation_stats["task_validity"]
        if task_stats["total_validated"] > 0:
            validity_rate = task_stats["valid"] / task_stats["total_validated"]
            if validity_rate < 0.8:
                recommendations.append(
                    "Improve task specifications - many tasks have validity issues"
                )
        
        # PR validity
        pr_stats = validation_stats["pr_validity"]
        if pr_stats["total_validated"] > 0:
            pr_validity_rate = pr_stats["valid"] / pr_stats["total_validated"]
            if pr_validity_rate < 0.7:
                recommendations.append(
                    "Implement stricter PR validation - many PRs don't solve tasks correctly"
                )
        
        # Environment isolation
        env_stats = validation_stats["environment"]
        if env_stats["total_validated"] > 0:
            isolation_rate = env_stats["isolated"] / env_stats["total_validated"]
            if isolation_rate < 0.9:
                recommendations.append(
                    "Fix environment isolation - contamination detected between tests"
                )
        
        # Shortcut detection
        shortcut_stats = validation_stats["shortcuts"]
        if shortcut_stats["total_checked"] > 0:
            clean_rate = shortcut_stats["clean"] / shortcut_stats["total_checked"]
            if clean_rate < 0.8:
                recommendations.append(
                    "Address shortcut attempts - agents may be cheating or taking shortcuts"
                )
        
        return recommendations
    
    def save_abc_report(self, filepath: str):
        """Save ABC compliance report to file"""
        report = self.generate_abc_report()
        
        # Add detailed test results
        report["detailed_results"] = [
            result.get_detailed_report() for result in self.abc_results
        ]
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"ABC compliance report saved to: {filepath}")
    
    def get_abc_summary(self) -> str:
        """Get a concise summary of ABC compliance"""
        if not self.abc_results:
            return "No ABC evaluation results available"
        
        report = self.generate_abc_report()
        summary = report["summary"]
        
        return (
            f"ABC Evaluation Summary:\n"
            f"├─ Tests Run: {summary['total_tests']}\n"
            f"├─ Original Success Rate: {summary['original_success_rate']:.1%}\n"
            f"├─ True Success Rate: {summary['true_success_rate']:.1%}\n"
            f"├─ Overestimation: {summary['overestimation_rate']:.1%}\n"
            f"├─ Avg Confidence: {summary['average_confidence_score']:.2f}\n"
            f"└─ ABC Compliance: {summary['average_compliance_score']:.2f}\n"
        )


# Factory function for easy instantiation
def create_abc_framework(
    api_base_url: str = "http://localhost:8000",
    github_token: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> ABCCompliantEvaluationFramework:
    """Create ABC-compliant evaluation framework with optional configuration"""
    
    framework = ABCCompliantEvaluationFramework(api_base_url, github_token)
    
    if config_overrides:
        framework.validation_config.update(config_overrides)
    
    return framework