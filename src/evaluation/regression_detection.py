"""
Regression detection system for autonomous coding agents
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from .eval_framework import EvaluationResult, TestStatus

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    test_id: str
    success_rate: float
    avg_execution_time: float
    avg_quality_score: float
    error_rate: float
    timestamp: float
    version: str = "unknown"


@dataclass
class RegressionResult:
    """Result of regression analysis"""
    test_id: str
    regression_detected: bool
    severity: str  # "critical", "major", "minor", "none"
    metrics_affected: List[str]
    performance_delta: Dict[str, float]
    recommendation: str


class RegressionDetector:
    """Detects performance regressions in autonomous coding agents"""
    
    def __init__(self, baseline_file: str = "performance_baselines.json"):
        self.baseline_file = baseline_file
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.thresholds = {
            "success_rate_drop": 0.10,  # 10% drop in success rate
            "execution_time_increase": 0.30,  # 30% increase in execution time
            "quality_score_drop": 0.15,  # 15% drop in quality score
            "error_rate_increase": 2.0   # 2x increase in error rate
        }
        self.load_baselines()
        
    def load_baselines(self):
        """Load performance baselines from file"""
        
        try:
            if Path(self.baseline_file).exists():
                with open(self.baseline_file, 'r') as f:
                    baseline_data = json.load(f)
                    
                self.baselines = {
                    test_id: PerformanceBaseline(**data)
                    for test_id, data in baseline_data.items()
                }
                    
                logger.info(f"Loaded {len(self.baselines)} performance baselines")
            else:
                logger.info("No baseline file found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading baselines: {e}")
            self.baselines = {}
            
    def save_baselines(self):
        """Save performance baselines to file"""
        
        try:
            baseline_data = {
                test_id: asdict(baseline)
                for test_id, baseline in self.baselines.items()
            }
            
            with open(self.baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
                
            logger.info(f"Saved {len(self.baselines)} performance baselines")
            
        except Exception as e:
            logger.error(f"Error saving baselines: {e}")
            
    def update_baseline(self, results: List[EvaluationResult], version: str = "unknown"):
        """Update performance baselines with new results"""
        
        logger.info(f"Updating baselines with {len(results)} results")
        
        # Group results by test_id
        test_groups = {}
        for result in results:
            test_id = result.test_case.id
            if test_id not in test_groups:
                test_groups[test_id] = []
            test_groups[test_id].append(result)
            
        # Calculate baselines for each test
        for test_id, test_results in test_groups.items():
            baseline = self._calculate_baseline(test_results, version)
            self.baselines[test_id] = baseline
            
        self.save_baselines()
        
    def _calculate_baseline(self, results: List[EvaluationResult], version: str) -> PerformanceBaseline:
        """Calculate baseline metrics from results"""
        
        if not results:
            return PerformanceBaseline(
                test_id="unknown",
                success_rate=0.0,
                avg_execution_time=0.0,
                avg_quality_score=0.0,
                error_rate=0.0,
                timestamp=time.time(),
                version=version
            )
            
        test_id = results[0].test_case.id
        
        # Calculate metrics
        successful_runs = [r for r in results if r.status == TestStatus.PASSED]
        success_rate = len(successful_runs) / len(results)
        
        avg_execution_time = sum(r.metrics.execution_time for r in results) / len(results)
        avg_quality_score = sum(r.metrics.code_quality_score for r in results) / len(results)
        
        total_errors = sum(r.metrics.error_count for r in results)
        error_rate = total_errors / len(results)
        
        return PerformanceBaseline(
            test_id=test_id,
            success_rate=success_rate,
            avg_execution_time=avg_execution_time,
            avg_quality_score=avg_quality_score,
            error_rate=error_rate,
            timestamp=time.time(),
            version=version
        )
        
    def detect_regressions(self, results: List[EvaluationResult]) -> List[RegressionResult]:
        """Detect regressions by comparing against baselines"""
        
        logger.info(f"Detecting regressions in {len(results)} results")
        
        regression_results = []
        
        # Group results by test_id
        test_groups = {}
        for result in results:
            test_id = result.test_case.id
            if test_id not in test_groups:
                test_groups[test_id] = []
            test_groups[test_id].append(result)
            
        # Analyze each test group
        for test_id, test_results in test_groups.items():
            if test_id in self.baselines:
                regression = self._analyze_test_regression(test_id, test_results)
                regression_results.append(regression)
            else:
                logger.warning(f"No baseline found for test {test_id}")
                
        return regression_results
        
    def _analyze_test_regression(self, test_id: str, results: List[EvaluationResult]) -> RegressionResult:
        """Analyze regression for a specific test"""
        
        baseline = self.baselines[test_id]
        current = self._calculate_baseline(results, "current")
        
        # Calculate performance deltas
        performance_delta = {
            "success_rate": current.success_rate - baseline.success_rate,
            "execution_time": (current.avg_execution_time - baseline.avg_execution_time) / baseline.avg_execution_time if baseline.avg_execution_time > 0 else 0,
            "quality_score": current.avg_quality_score - baseline.avg_quality_score,
            "error_rate": current.error_rate - baseline.error_rate
        }
        
        # Detect regressions
        regressions = []
        
        if performance_delta["success_rate"] < -self.thresholds["success_rate_drop"]:
            regressions.append("success_rate")
            
        if performance_delta["execution_time"] > self.thresholds["execution_time_increase"]:
            regressions.append("execution_time")
            
        if performance_delta["quality_score"] < -self.thresholds["quality_score_drop"]:
            regressions.append("quality_score")
            
        if performance_delta["error_rate"] > self.thresholds["error_rate_increase"]:
            regressions.append("error_rate")
            
        # Determine severity
        severity = self._determine_severity(regressions, performance_delta)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(regressions, performance_delta)
        
        return RegressionResult(
            test_id=test_id,
            regression_detected=len(regressions) > 0,
            severity=severity,
            metrics_affected=regressions,
            performance_delta=performance_delta,
            recommendation=recommendation
        )
        
    def _determine_severity(self, regressions: List[str], deltas: Dict[str, float]) -> str:
        """Determine regression severity"""
        
        if not regressions:
            return "none"
            
        # Critical: Success rate drops significantly or massive time increase
        if "success_rate" in regressions and deltas["success_rate"] < -0.25:
            return "critical"
        if "execution_time" in regressions and deltas["execution_time"] > 1.0:  # 100% increase
            return "critical"
            
        # Major: Multiple metrics affected or significant drops
        if len(regressions) >= 2:
            return "major"
        if "success_rate" in regressions and deltas["success_rate"] < -0.15:
            return "major"
            
        # Minor: Single metric with moderate impact
        return "minor"
        
    def _generate_recommendation(self, regressions: List[str], deltas: Dict[str, float]) -> str:
        """Generate actionable recommendation"""
        
        if not regressions:
            return "No regressions detected. Performance is stable."
            
        recommendations = []
        
        if "success_rate" in regressions:
            drop_pct = abs(deltas["success_rate"]) * 100
            recommendations.append(f"Success rate dropped by {drop_pct:.1f}%. Investigate failures and error handling.")
            
        if "execution_time" in regressions:
            increase_pct = deltas["execution_time"] * 100
            recommendations.append(f"Execution time increased by {increase_pct:.1f}%. Profile code and optimize performance bottlenecks.")
            
        if "quality_score" in regressions:
            drop_pct = abs(deltas["quality_score"])
            recommendations.append(f"Code quality score dropped by {drop_pct:.1f} points. Review code generation logic.")
            
        if "error_rate" in regressions:
            recommendations.append("Error rate increased significantly. Improve error handling and input validation.")
            
        return " ".join(recommendations)
        
    def generate_regression_report(self, regression_results: List[RegressionResult]) -> Dict[str, Any]:
        """Generate comprehensive regression report"""
        
        total_tests = len(regression_results)
        regressions_detected = [r for r in regression_results if r.regression_detected]
        
        severity_counts = {
            "critical": len([r for r in regressions_detected if r.severity == "critical"]),
            "major": len([r for r in regressions_detected if r.severity == "major"]),
            "minor": len([r for r in regressions_detected if r.severity == "minor"])
        }
        
        metrics_affected = {}
        for result in regressions_detected:
            for metric in result.metrics_affected:
                metrics_affected[metric] = metrics_affected.get(metric, 0) + 1
                
        report = {
            "summary": {
                "total_tests_analyzed": total_tests,
                "regressions_detected": len(regressions_detected),
                "regression_rate": len(regressions_detected) / total_tests if total_tests > 0 else 0,
                "severity_breakdown": severity_counts
            },
            "affected_metrics": metrics_affected,
            "critical_issues": [
                {
                    "test_id": r.test_id,
                    "metrics_affected": r.metrics_affected,
                    "recommendation": r.recommendation
                }
                for r in regressions_detected if r.severity == "critical"
            ],
            "detailed_results": [
                {
                    "test_id": r.test_id,
                    "regression_detected": r.regression_detected,
                    "severity": r.severity,
                    "metrics_affected": r.metrics_affected,
                    "performance_delta": r.performance_delta,
                    "recommendation": r.recommendation
                }
                for r in regression_results
            ],
            "overall_health": self._assess_overall_health(regression_results)
        }
        
        return report
        
    def _assess_overall_health(self, regression_results: List[RegressionResult]) -> Dict[str, Any]:
        """Assess overall system health"""
        
        if not regression_results:
            return {"status": "unknown", "reason": "No data available"}
            
        critical_count = len([r for r in regression_results if r.severity == "critical"])
        major_count = len([r for r in regression_results if r.severity == "major"])
        regression_rate = len([r for r in regression_results if r.regression_detected]) / len(regression_results)
        
        if critical_count > 0:
            status = "critical"
            reason = f"{critical_count} critical regressions detected"
        elif major_count > 2:
            status = "degraded"
            reason = f"{major_count} major regressions detected"
        elif regression_rate > 0.30:
            status = "degraded"
            reason = f"High regression rate: {regression_rate:.1%}"
        elif regression_rate > 0.10:
            status = "warning"
            reason = f"Moderate regression rate: {regression_rate:.1%}"
        else:
            status = "healthy"
            reason = "No significant regressions detected"
            
        return {
            "status": status,
            "reason": reason,
            "regression_rate": regression_rate,
            "critical_issues": critical_count,
            "major_issues": major_count
        }
        
    def get_performance_trends(self, test_id: str, window_size: int = 10) -> Dict[str, Any]:
        """Get performance trends for a specific test"""
        
        if test_id not in self.baselines:
            return {"error": f"No baseline data for test {test_id}"}
            
        baseline = self.baselines[test_id]
        
        # For now, return current baseline data
        # In a full implementation, this would track historical data
        return {
            "test_id": test_id,
            "current_metrics": {
                "success_rate": baseline.success_rate,
                "avg_execution_time": baseline.avg_execution_time,
                "avg_quality_score": baseline.avg_quality_score,
                "error_rate": baseline.error_rate
            },
            "last_updated": baseline.timestamp,
            "version": baseline.version,
            "trend_analysis": "Historical trend analysis requires multiple baseline snapshots"
        }
        
    def cleanup_old_baselines(self, max_age_days: int = 30):
        """Remove baselines older than specified age"""
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        old_baselines = [
            test_id for test_id, baseline in self.baselines.items()
            if current_time - baseline.timestamp > max_age_seconds
        ]
        
        for test_id in old_baselines:
            del self.baselines[test_id]
            
        if old_baselines:
            logger.info(f"Cleaned up {len(old_baselines)} old baselines")
            self.save_baselines()
            
    def export_performance_data(self, filepath: str):
        """Export performance data for analysis"""
        
        export_data = {
            "baselines": {
                test_id: asdict(baseline)
                for test_id, baseline in self.baselines.items()
            },
            "thresholds": self.thresholds,
            "export_timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported performance data to {filepath}")
        
    def import_performance_data(self, filepath: str):
        """Import performance data from file"""
        
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
                
            # Import baselines
            if "baselines" in import_data:
                for test_id, baseline_data in import_data["baselines"].items():
                    self.baselines[test_id] = PerformanceBaseline(**baseline_data)
                    
            # Import thresholds if available
            if "thresholds" in import_data:
                self.thresholds.update(import_data["thresholds"])
                
            logger.info(f"Imported performance data from {filepath}")
            self.save_baselines()
            
        except Exception as e:
            logger.error(f"Error importing performance data: {e}")
            
    def set_thresholds(self, new_thresholds: Dict[str, float]):
        """Update regression detection thresholds"""
        
        self.thresholds.update(new_thresholds)
        logger.info(f"Updated regression thresholds: {self.thresholds}")
        
    def get_baseline_summary(self) -> Dict[str, Any]:
        """Get summary of all baselines"""
        
        if not self.baselines:
            return {"total_baselines": 0, "message": "No baselines available"}
            
        summary = {
            "total_baselines": len(self.baselines),
            "avg_success_rate": sum(b.success_rate for b in self.baselines.values()) / len(self.baselines),
            "avg_execution_time": sum(b.avg_execution_time for b in self.baselines.values()) / len(self.baselines),
            "avg_quality_score": sum(b.avg_quality_score for b in self.baselines.values()) / len(self.baselines),
            "latest_update": max(b.timestamp for b in self.baselines.values()),
            "test_coverage": list(self.baselines.keys())
        }
        
        return summary