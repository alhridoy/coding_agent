"""
Core evaluation framework for autonomous coding agents
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from enum import Enum
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TestCase:
    """Individual test case definition"""
    id: str
    name: str
    description: str
    repo_url: str
    prompt: str
    expected_outcome: Dict[str, Any]
    timeout_seconds: int = 300
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # api, frontend, backend, etc.


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a test case"""
    success_rate: float = 0.0
    execution_time: float = 0.0
    code_quality_score: float = 0.0
    pr_quality_score: float = 0.0
    error_count: int = 0
    warnings_count: int = 0
    files_modified: int = 0
    tests_passed: int = 0
    tests_failed: int = 0


@dataclass
class EvaluationResult:
    """Result of a single test case evaluation"""
    test_case: TestCase
    status: TestStatus
    metrics: EvaluationMetrics
    start_time: float
    end_time: float
    error_message: Optional[str] = None
    pr_url: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)


class EvaluationFramework:
    """Core evaluation framework for testing autonomous coding agents"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.results: List[EvaluationResult] = []
        self.evaluators: List[Callable] = []
        
    def add_evaluator(self, evaluator: Callable):
        """Add a custom evaluator function"""
        self.evaluators.append(evaluator)
        
    async def run_test_case(self, test_case: TestCase) -> EvaluationResult:
        """Run a single test case"""
        logger.info(f"Starting test case: {test_case.id}")
        
        result = EvaluationResult(
            test_case=test_case,
            status=TestStatus.RUNNING,
            metrics=EvaluationMetrics(),
            start_time=time.time(),
            end_time=0.0
        )
        
        try:
            # Execute the test case
            await self._execute_test(test_case, result)
            
            # Run custom evaluators
            for evaluator in self.evaluators:
                await evaluator(result)
                
            # Determine final status
            if result.status == TestStatus.RUNNING:
                result.status = TestStatus.PASSED if result.pr_url else TestStatus.FAILED
                
        except asyncio.TimeoutError:
            result.status = TestStatus.TIMEOUT
            result.error_message = f"Test timed out after {test_case.timeout_seconds} seconds"
            logger.error(f"Test {test_case.id} timed out")
            
        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            result.logs.append(traceback.format_exc())
            logger.error(f"Test {test_case.id} failed: {e}")
            
        finally:
            result.end_time = time.time()
            result.metrics.execution_time = result.end_time - result.start_time
            
        self.results.append(result)
        return result
        
    async def _execute_test(self, test_case: TestCase, result: EvaluationResult):
        """Execute the actual test case"""
        import httpx
        
        # Prepare request payload
        payload = {
            "repo_url": test_case.repo_url,
            "prompt": test_case.prompt,
            "branch_name": f"eval/{test_case.id}",
            "pr_title": f"Evaluation: {test_case.name}"
        }
        
        # Execute with timeout
        async with httpx.AsyncClient(timeout=test_case.timeout_seconds) as client:
            async with client.stream(
                "POST",
                f"{self.api_base_url}/code",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise Exception(f"API request failed: {response.status_code} - {error_text.decode()}")
                
                # Process streaming events
                event_count = 0
                async for chunk in response.aiter_text():
                    for line in chunk.split('\n'):
                        if line.strip().startswith('data: '):
                            event_count += 1
                            try:
                                event_data = json.loads(line[6:])
                                await self._process_event(event_data, result)
                            except json.JSONDecodeError:
                                result.logs.append(f"Invalid JSON: {line}")
                
                result.artifacts["total_events"] = event_count
                
    async def _process_event(self, event_data: Dict[str, Any], result: EvaluationResult):
        """Process a streaming event"""
        event_type = event_data.get('type', 'Unknown')
        
        # Log all events
        result.logs.append(f"{event_type}: {event_data}")
        
        # Track specific events
        if event_type == "Error":
            result.metrics.error_count += 1
            
        elif event_type.startswith("Tool: Edit"):
            result.metrics.files_modified += 1
            
        elif 'pr_url' in event_data:
            result.pr_url = event_data.get('pr_url', '')
            
        elif event_type == "Status" and "warning" in event_data.get('message', '').lower():
            result.metrics.warnings_count += 1
            
    async def run_test_suite(self, test_cases: List[TestCase]) -> List[EvaluationResult]:
        """Run a complete test suite"""
        logger.info(f"Running test suite with {len(test_cases)} test cases")
        
        results = []
        for test_case in test_cases:
            result = await self.run_test_case(test_case)
            results.append(result)
            
            # Log progress
            logger.info(f"Completed {test_case.id}: {result.status.value}")
            
        return results
        
    async def run_parallel_tests(self, test_cases: List[TestCase], max_concurrent: int = 3) -> List[EvaluationResult]:
        """Run tests in parallel with concurrency limit"""
        logger.info(f"Running {len(test_cases)} tests with max concurrency: {max_concurrent}")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(test_case: TestCase):
            async with semaphore:
                return await self.run_test_case(test_case)
                
        tasks = [run_with_semaphore(tc) for tc in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = EvaluationResult(
                    test_case=test_cases[i],
                    status=TestStatus.FAILED,
                    metrics=EvaluationMetrics(),
                    start_time=time.time(),
                    end_time=time.time(),
                    error_message=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
                
        return processed_results
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate evaluation report"""
        if not self.results:
            return {"error": "No results to report"}
            
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        timeout = sum(1 for r in self.results if r.status == TestStatus.TIMEOUT)
        
        avg_time = sum(r.metrics.execution_time for r in self.results) / total_tests
        avg_quality = sum(r.metrics.code_quality_score for r in self.results) / total_tests
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "timeout": timeout,
                "success_rate": passed / total_tests * 100,
                "average_execution_time": avg_time,
                "average_quality_score": avg_quality
            },
            "results": [
                {
                    "test_id": r.test_case.id,
                    "status": r.status.value,
                    "execution_time": r.metrics.execution_time,
                    "pr_url": r.pr_url,
                    "error": r.error_message
                }
                for r in self.results
            ],
            "metrics": {
                "total_files_modified": sum(r.metrics.files_modified for r in self.results),
                "total_errors": sum(r.metrics.error_count for r in self.results),
                "total_warnings": sum(r.metrics.warnings_count for r in self.results)
            }
        }
        
        return report
        
    def save_report(self, filepath: str):
        """Save evaluation report to file"""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Evaluation report saved to: {filepath}")
        
    def get_failed_tests(self) -> List[EvaluationResult]:
        """Get all failed test results"""
        return [r for r in self.results if r.status == TestStatus.FAILED]
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics across all tests"""
        if not self.results:
            return {}
            
        return {
            "avg_execution_time": sum(r.metrics.execution_time for r in self.results) / len(self.results),
            "max_execution_time": max(r.metrics.execution_time for r in self.results),
            "min_execution_time": min(r.metrics.execution_time for r in self.results),
            "success_rate": sum(1 for r in self.results if r.status == TestStatus.PASSED) / len(self.results),
            "avg_files_modified": sum(r.metrics.files_modified for r in self.results) / len(self.results)
        }