"""
Shortcut Detection System
Detects various ways agents might cheat or take shortcuts in evaluations
Based on ABC principles for preventing false positives
"""

import re
import json
import logging
import random
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ShortcutViolation:
    """Represents a detected shortcut or cheating attempt"""
    violation_type: str  # 'trivial_solution', 'pattern_matching', 'test_peeking', etc.
    confidence: float  # 0-1 confidence in detection
    description: str
    evidence: List[str]
    severity: str  # 'critical', 'high', 'medium', 'low'
    recommendation: str


@dataclass
class ShortcutDetectionResult:
    """Result of shortcut detection analysis"""
    is_clean: bool
    violations: List[ShortcutViolation]
    suspicion_score: float  # 0-1, higher = more suspicious
    analysis_details: Dict[str, Any]


class ShortcutDetector(ABC):
    """Base class for different types of shortcut detectors"""
    
    @abstractmethod
    async def detect_shortcuts(
        self,
        test_case: 'TestCase',
        evaluation_result: 'EvaluationResult',
        agent_logs: List[str]
    ) -> List[ShortcutViolation]:
        """Detect shortcuts in the evaluation"""
        pass
    
    @abstractmethod
    def get_detector_type(self) -> str:
        """Get the type of shortcuts this detector finds"""
        pass


class TrivialSolutionDetector(ShortcutDetector):
    """Detects trivial solutions that shouldn't pass"""
    
    async def detect_shortcuts(
        self,
        test_case: 'TestCase',
        evaluation_result: 'EvaluationResult',
        agent_logs: List[str]
    ) -> List[ShortcutViolation]:
        """Detect trivial solutions"""
        violations = []
        
        # Check for empty or minimal responses
        empty_response_violation = self._check_empty_responses(
            test_case, evaluation_result, agent_logs
        )
        if empty_response_violation:
            violations.append(empty_response_violation)
        
        # Check for copy-paste solutions
        copy_paste_violation = self._check_copy_paste(agent_logs)
        if copy_paste_violation:
            violations.append(copy_paste_violation)
        
        # Check for placeholder/dummy code
        placeholder_violation = self._check_placeholder_code(agent_logs)
        if placeholder_violation:
            violations.append(placeholder_violation)
        
        # Check for overly simple solutions to complex problems
        simplicity_violation = self._check_excessive_simplicity(
            test_case, agent_logs
        )
        if simplicity_violation:
            violations.append(simplicity_violation)
        
        return violations
    
    def get_detector_type(self) -> str:
        return "trivial_solution"
    
    def _check_empty_responses(
        self,
        test_case: 'TestCase',
        evaluation_result: 'EvaluationResult',
        agent_logs: List[str]
    ) -> Optional[ShortcutViolation]:
        """Check for empty or do-nothing responses that somehow pass"""
        
        # Check if PR was created but no actual changes made
        if evaluation_result.pr_url and evaluation_result.metrics.files_modified == 0:
            return ShortcutViolation(
                violation_type="trivial_solution",
                confidence=0.9,
                description="PR created but no files were modified",
                evidence=["PR URL exists", "files_modified = 0"],
                severity="critical",
                recommendation="Verify PR contains actual code changes"
            )
        
        # Check for minimal code changes on complex tasks
        if test_case.difficulty in ['hard', 'expert']:
            total_changes = 0
            for log in agent_logs:
                if "new_str" in log:
                    # Count characters in changes
                    match = re.search(r'"new_str":\s*"([^"]*)"', log)
                    if match:
                        total_changes += len(match.group(1))
            
            if total_changes < 100:  # Less than 100 characters changed
                return ShortcutViolation(
                    violation_type="trivial_solution",
                    confidence=0.8,
                    description=f"Minimal changes ({total_changes} chars) for complex task",
                    evidence=[f"Total code changes: {total_changes} characters"],
                    severity="high",
                    recommendation="Ensure changes are substantial enough for task complexity"
                )
        
        return None
    
    def _check_copy_paste(self, agent_logs: List[str]) -> Optional[ShortcutViolation]:
        """Check for copy-paste solutions"""
        
        code_snippets = []
        for log in agent_logs:
            if "new_str" in log:
                match = re.search(r'"new_str":\s*"([^"]*)"', log)
                if match:
                    code_snippets.append(match.group(1))
        
        # Check for identical code blocks
        if len(code_snippets) > 1:
            unique_snippets = set(code_snippets)
            if len(unique_snippets) < len(code_snippets) / 2:
                return ShortcutViolation(
                    violation_type="trivial_solution",
                    confidence=0.7,
                    description="Excessive code duplication detected",
                    evidence=[f"Unique snippets: {len(unique_snippets)}, Total: {len(code_snippets)}"],
                    severity="medium",
                    recommendation="Avoid copy-paste solutions, write contextual code"
                )
        
        return None
    
    def _check_placeholder_code(self, agent_logs: List[str]) -> Optional[ShortcutViolation]:
        """Check for placeholder or dummy code"""
        
        placeholder_patterns = [
            r'TODO',
            r'FIXME',
            r'placeholder',
            r'dummy',
            r'fake',
            r'mock',
            r'test.*test',
            r'foo.*bar',
            r'example',
            r'sample'
        ]
        
        placeholder_count = 0
        evidence = []
        
        for log in agent_logs:
            if "new_str" in log:
                match = re.search(r'"new_str":\s*"([^"]*)"', log)
                if match:
                    code = match.group(1)
                    for pattern in placeholder_patterns:
                        matches = re.findall(pattern, code, re.IGNORECASE)
                        if matches:
                            placeholder_count += len(matches)
                            evidence.extend(matches)
        
        if placeholder_count > 2:
            return ShortcutViolation(
                violation_type="trivial_solution",
                confidence=0.8,
                description="Placeholder/dummy code detected",
                evidence=evidence[:5],  # Limit evidence
                severity="high",
                recommendation="Replace placeholders with actual implementation"
            )
        
        return None
    
    def _check_excessive_simplicity(
        self,
        test_case: 'TestCase',
        agent_logs: List[str]
    ) -> Optional[ShortcutViolation]:
        """Check for overly simple solutions to complex problems"""
        
        if test_case.difficulty not in ['hard', 'expert']:
            return None
        
        # Count complexity indicators
        complexity_patterns = [
            r'if\s+',
            r'for\s+',
            r'while\s+',
            r'try\s*:',
            r'def\s+',
            r'class\s+',
            r'import\s+',
            r'return\s+'
        ]
        
        total_complexity = 0
        total_lines = 0
        
        for log in agent_logs:
            if "new_str" in log:
                match = re.search(r'"new_str":\s*"([^"]*)"', log)
                if match:
                    code = match.group(1)
                    lines = code.split('\n')
                    total_lines += len([l for l in lines if l.strip()])
                    
                    for pattern in complexity_patterns:
                        total_complexity += len(re.findall(pattern, code))
        
        if total_lines > 10:  # Only check for substantial code
            complexity_ratio = total_complexity / total_lines
            
            if complexity_ratio < 0.1:  # Very low complexity
                return ShortcutViolation(
                    violation_type="trivial_solution",
                    confidence=0.6,
                    description=f"Solution too simple for {test_case.difficulty} task",
                    evidence=[f"Complexity ratio: {complexity_ratio:.2f}"],
                    severity="medium",
                    recommendation="Consider if solution adequately addresses task complexity"
                )
        
        return None


class PatternMatchingDetector(ShortcutDetector):
    """Detects pattern matching or overfitting to specific test cases"""
    
    def __init__(self):
        self.known_patterns: Set[str] = set()
        self.execution_patterns: Dict[str, int] = {}
    
    async def detect_shortcuts(
        self,
        test_case: 'TestCase',
        evaluation_result: 'EvaluationResult',
        agent_logs: List[str]
    ) -> List[ShortcutViolation]:
        """Detect pattern matching shortcuts"""
        violations = []
        
        # Check for hardcoded solutions
        hardcoded_violation = self._check_hardcoded_solutions(
            test_case, agent_logs
        )
        if hardcoded_violation:
            violations.append(hardcoded_violation)
        
        # Check for test-specific patterns
        pattern_violation = self._check_test_specific_patterns(
            test_case, agent_logs
        )
        if pattern_violation:
            violations.append(pattern_violation)
        
        # Check for execution pattern matching
        execution_violation = self._check_execution_patterns(
            test_case, agent_logs
        )
        if execution_violation:
            violations.append(execution_violation)
        
        return violations
    
    def get_detector_type(self) -> str:
        return "pattern_matching"
    
    def _check_hardcoded_solutions(
        self,
        test_case: 'TestCase',
        agent_logs: List[str]
    ) -> Optional[ShortcutViolation]:
        """Check for hardcoded solutions specific to test case"""
        
        # Extract test-specific identifiers
        test_identifiers = [
            test_case.id,
            test_case.name.lower().replace(' ', '_'),
            test_case.repo_url.split('/')[-1]
        ]
        
        evidence = []
        for log in agent_logs:
            for identifier in test_identifiers:
                if identifier in log.lower():
                    evidence.append(f"Test identifier '{identifier}' found in code")
        
        if evidence:
            return ShortcutViolation(
                violation_type="pattern_matching",
                confidence=0.9,
                description="Hardcoded test-specific identifiers detected",
                evidence=evidence,
                severity="critical",
                recommendation="Remove hardcoded test-specific values"
            )
        
        return None
    
    def _check_test_specific_patterns(
        self,
        test_case: 'TestCase',
        agent_logs: List[str]
    ) -> Optional[ShortcutViolation]:
        """Check for patterns specific to test structure"""
        
        # Look for references to expected outcomes
        if hasattr(test_case, 'expected_outcome'):
            expected_keys = list(test_case.expected_outcome.keys())
            evidence = []
            
            for log in agent_logs:
                for key in expected_keys:
                    if key in log and 'expected' in log.lower():
                        evidence.append(f"Expected outcome key '{key}' referenced")
            
            if evidence:
                return ShortcutViolation(
                    violation_type="pattern_matching",
                    confidence=0.8,
                    description="References to expected outcomes detected",
                    evidence=evidence,
                    severity="critical",
                    recommendation="Ensure agent cannot access test expectations"
                )
        
        return None
    
    def _check_execution_patterns(
        self,
        test_case: 'TestCase',
        agent_logs: List[str]
    ) -> Optional[ShortcutViolation]:
        """Check for suspicious execution patterns"""
        
        # Create execution fingerprint
        fingerprint = self._create_execution_fingerprint(agent_logs)
        
        # Check if we've seen this pattern before
        if fingerprint in self.execution_patterns:
            self.execution_patterns[fingerprint] += 1
            
            if self.execution_patterns[fingerprint] > 3:
                return ShortcutViolation(
                    violation_type="pattern_matching",
                    confidence=0.7,
                    description="Identical execution pattern repeated",
                    evidence=[f"Pattern seen {self.execution_patterns[fingerprint]} times"],
                    severity="medium",
                    recommendation="Verify solution is not memorized/cached"
                )
        else:
            self.execution_patterns[fingerprint] = 1
        
        return None
    
    def _create_execution_fingerprint(self, agent_logs: List[str]) -> str:
        """Create a fingerprint of the execution pattern"""
        
        # Extract key events in order
        events = []
        for log in agent_logs:
            if any(event in log for event in ['Tool:', 'Status:', 'AI Message:']):
                # Normalize the log entry
                normalized = re.sub(r'"[^"]*"', '""', log)  # Remove content
                normalized = re.sub(r'\d+', 'N', normalized)  # Replace numbers
                events.append(normalized)
        
        # Create hash of event sequence
        fingerprint = hashlib.md5('\n'.join(events).encode()).hexdigest()
        return fingerprint


class TestPeekingDetector(ShortcutDetector):
    """Detects attempts to peek at test files or expected outcomes"""
    
    async def detect_shortcuts(
        self,
        test_case: 'TestCase',
        evaluation_result: 'EvaluationResult',
        agent_logs: List[str]
    ) -> List[ShortcutViolation]:
        """Detect test peeking shortcuts"""
        violations = []
        
        # Check for file access patterns
        file_access_violation = self._check_suspicious_file_access(agent_logs)
        if file_access_violation:
            violations.append(file_access_violation)
        
        # Check for metadata access
        metadata_violation = self._check_metadata_access(agent_logs)
        if metadata_violation:
            violations.append(metadata_violation)
        
        # Check for framework introspection
        introspection_violation = self._check_framework_introspection(agent_logs)
        if introspection_violation:
            violations.append(introspection_violation)
        
        return violations
    
    def get_detector_type(self) -> str:
        return "test_peeking"
    
    def _check_suspicious_file_access(
        self, 
        agent_logs: List[str]
    ) -> Optional[ShortcutViolation]:
        """Check for suspicious file access patterns"""
        
        suspicious_patterns = [
            r'test.*\.py',
            r'eval.*\.py',
            r'expected.*',
            r'solution.*',
            r'answer.*',
            r'benchmark.*',
            r'config.*test',
            r'\.env.*test'
        ]
        
        evidence = []
        for log in agent_logs:
            if 'read' in log.lower() or 'file' in log.lower():
                for pattern in suspicious_patterns:
                    if re.search(pattern, log, re.IGNORECASE):
                        evidence.append(f"Suspicious file access: {pattern}")
        
        if evidence:
            return ShortcutViolation(
                violation_type="test_peeking",
                confidence=0.8,
                description="Suspicious file access patterns detected",
                evidence=evidence[:3],
                severity="critical",
                recommendation="Block access to test-related files"
            )
        
        return None
    
    def _check_metadata_access(
        self, 
        agent_logs: List[str]
    ) -> Optional[ShortcutViolation]:
        """Check for attempts to access test metadata"""
        
        metadata_patterns = [
            r'__.*__',  # Python magic methods
            r'globals\(\)',
            r'locals\(\)',
            r'vars\(\)',
            r'dir\(\)',
            r'inspect\.',
            r'sys\.modules',
            r'importlib',
            r'eval\(',
            r'exec\('
        ]
        
        evidence = []
        for log in agent_logs:
            for pattern in metadata_patterns:
                if re.search(pattern, log):
                    evidence.append(f"Metadata access: {pattern}")
        
        if evidence:
            return ShortcutViolation(
                violation_type="test_peeking",
                confidence=0.9,
                description="Metadata access attempts detected",
                evidence=evidence[:3],
                severity="critical",
                recommendation="Block introspection capabilities"
            )
        
        return None
    
    def _check_framework_introspection(
        self, 
        agent_logs: List[str]
    ) -> Optional[ShortcutViolation]:
        """Check for framework introspection attempts"""
        
        framework_patterns = [
            r'TestCase',
            r'EvaluationResult',
            r'expected_outcome',
            r'test_case\.',
            r'evaluation\.',
            r'framework\.',
            r'validator\.'
        ]
        
        evidence = []
        for log in agent_logs:
            if 'new_str' in log:  # Only check actual code
                match = re.search(r'"new_str":\s*"([^"]*)"', log)
                if match:
                    code = match.group(1)
                    for pattern in framework_patterns:
                        if re.search(pattern, code):
                            evidence.append(f"Framework reference: {pattern}")
        
        if evidence:
            return ShortcutViolation(
                violation_type="test_peeking",
                confidence=0.8,
                description="Framework introspection detected",
                evidence=evidence[:3],
                severity="high",
                recommendation="Isolate agent from evaluation framework"
            )
        
        return None


class RandomizedTestDetector(ShortcutDetector):
    """Detects shortcuts using randomized test variations"""
    
    def __init__(self):
        self.baseline_results: Dict[str, List[bool]] = {}
    
    async def detect_shortcuts(
        self,
        test_case: 'TestCase',
        evaluation_result: 'EvaluationResult',
        agent_logs: List[str]
    ) -> List[ShortcutViolation]:
        """Detect shortcuts using randomized variations"""
        violations = []
        
        # Create variations of the test case
        variations = self._create_test_variations(test_case)
        
        # Run agent on variations (this would need integration with the framework)
        # For now, we'll simulate this check
        consistency_violation = await self._check_consistency_across_variations(
            test_case, evaluation_result, variations
        )
        if consistency_violation:
            violations.append(consistency_violation)
        
        return violations
    
    def get_detector_type(self) -> str:
        return "randomized_test"
    
    def _create_test_variations(self, test_case: 'TestCase') -> List['TestCase']:
        """Create variations of a test case to detect overfitting"""
        variations = []
        
        # Variation 1: Slightly different wording
        variation1 = test_case.__class__(
            id=f"{test_case.id}_var1",
            name=test_case.name,
            description=test_case.description,
            repo_url=test_case.repo_url,
            prompt=self._vary_prompt_wording(test_case.prompt),
            expected_outcome=test_case.expected_outcome,
            timeout_seconds=test_case.timeout_seconds,
            tags=test_case.tags,
            difficulty=test_case.difficulty,
            category=test_case.category
        )
        variations.append(variation1)
        
        # Variation 2: Different order of requirements
        variation2 = test_case.__class__(
            id=f"{test_case.id}_var2",
            name=test_case.name,
            description=test_case.description,
            repo_url=test_case.repo_url,
            prompt=self._reorder_prompt_requirements(test_case.prompt),
            expected_outcome=test_case.expected_outcome,
            timeout_seconds=test_case.timeout_seconds,
            tags=test_case.tags,
            difficulty=test_case.difficulty,
            category=test_case.category
        )
        variations.append(variation2)
        
        return variations
    
    def _vary_prompt_wording(self, prompt: str) -> str:
        """Create variations in prompt wording"""
        
        variations = {
            'add': 'create',
            'implement': 'build',
            'create': 'add',
            'endpoint': 'route',
            'function': 'method',
            'validate': 'check',
            'ensure': 'verify'
        }
        
        varied_prompt = prompt
        for original, variation in variations.items():
            if original in prompt.lower():
                varied_prompt = re.sub(
                    original, 
                    variation, 
                    varied_prompt, 
                    count=1, 
                    flags=re.IGNORECASE
                )
                break
        
        return varied_prompt
    
    def _reorder_prompt_requirements(self, prompt: str) -> str:
        """Reorder requirements in the prompt"""
        
        # Split by common separators
        parts = re.split(r'[,;]', prompt)
        if len(parts) > 1:
            random.shuffle(parts)
            return ', '.join(part.strip() for part in parts)
        
        return prompt
    
    async def _check_consistency_across_variations(
        self,
        original_test: 'TestCase',
        original_result: 'EvaluationResult',
        variations: List['TestCase']
    ) -> Optional[ShortcutViolation]:
        """Check if results are consistent across variations"""
        
        # This would need actual execution of variations
        # For now, simulate based on patterns
        
        original_success = original_result.status == 'passed'
        
        # Record baseline
        test_family = original_test.id.split('_')[0]
        if test_family not in self.baseline_results:
            self.baseline_results[test_family] = []
        self.baseline_results[test_family].append(original_success)
        
        # Check for suspicious patterns
        if len(self.baseline_results[test_family]) >= 3:
            results = self.baseline_results[test_family]
            
            # All same result might indicate pattern matching
            if all(r == results[0] for r in results):
                success_rate = sum(results) / len(results)
                
                if success_rate == 1.0 or success_rate == 0.0:
                    return ShortcutViolation(
                        violation_type="pattern_matching",
                        confidence=0.6,
                        description=f"Identical results across variations ({success_rate:.0%})",
                        evidence=[f"Pattern: {results}"],
                        severity="medium",
                        recommendation="Test with more diverse prompts"
                    )
        
        return None


class ShortcutDetectionSystem:
    """Main system for detecting shortcuts and cheating"""
    
    def __init__(self):
        self.detectors: List[ShortcutDetector] = [
            TrivialSolutionDetector(),
            PatternMatchingDetector(),
            TestPeekingDetector(),
            RandomizedTestDetector()
        ]
        
    async def analyze_for_shortcuts(
        self,
        test_case: 'TestCase',
        evaluation_result: 'EvaluationResult',
        agent_logs: List[str]
    ) -> ShortcutDetectionResult:
        """Comprehensive shortcut detection analysis"""
        
        logger.info(f"Analyzing test {test_case.id} for shortcuts")
        
        all_violations = []
        analysis_details = {}
        
        # Run all detectors
        for detector in self.detectors:
            try:
                violations = await detector.detect_shortcuts(
                    test_case, evaluation_result, agent_logs
                )
                all_violations.extend(violations)
                analysis_details[detector.get_detector_type()] = {
                    "violations_found": len(violations),
                    "max_confidence": max((v.confidence for v in violations), default=0.0)
                }
                
            except Exception as e:
                logger.error(f"Error in {detector.get_detector_type()} detector: {e}")
                analysis_details[detector.get_detector_type()] = {
                    "error": str(e)
                }
        
        # Calculate suspicion score
        suspicion_score = self._calculate_suspicion_score(all_violations)
        
        # Determine if clean
        is_clean = (
            len([v for v in all_violations if v.severity in ['critical', 'high']]) == 0 and
            suspicion_score < 0.3
        )
        
        return ShortcutDetectionResult(
            is_clean=is_clean,
            violations=all_violations,
            suspicion_score=suspicion_score,
            analysis_details=analysis_details
        )
    
    def _calculate_suspicion_score(self, violations: List[ShortcutViolation]) -> float:
        """Calculate overall suspicion score"""
        
        if not violations:
            return 0.0
        
        # Weight by severity and confidence
        severity_weights = {
            "critical": 0.4,
            "high": 0.2,
            "medium": 0.1,
            "low": 0.05
        }
        
        total_score = 0.0
        for violation in violations:
            weight = severity_weights.get(violation.severity, 0.05)
            total_score += weight * violation.confidence
        
        # Cap at 1.0
        return min(total_score, 1.0)
    
    def generate_shortcut_report(
        self,
        detection_results: List[ShortcutDetectionResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive shortcut detection report"""
        
        total_tests = len(detection_results)
        clean_tests = sum(1 for r in detection_results if r.is_clean)
        
        # Aggregate violations by type and severity
        violation_stats = {
            "by_type": {},
            "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0}
        }
        
        all_violations = []
        for result in detection_results:
            all_violations.extend(result.violations)
        
        for violation in all_violations:
            # Count by type
            violation_stats["by_type"][violation.violation_type] = (
                violation_stats["by_type"].get(violation.violation_type, 0) + 1
            )
            
            # Count by severity
            violation_stats["by_severity"][violation.severity] += 1
        
        # Calculate average suspicion score
        avg_suspicion = (
            sum(r.suspicion_score for r in detection_results) / total_tests
            if total_tests > 0 else 0
        )
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "clean_tests": clean_tests,
                "suspicious_tests": total_tests - clean_tests,
                "clean_rate": clean_tests / total_tests if total_tests > 0 else 0,
                "average_suspicion_score": avg_suspicion
            },
            "violations": violation_stats,
            "most_common_violation": (
                max(violation_stats["by_type"].items(), key=lambda x: x[1])[0]
                if violation_stats["by_type"] else None
            ),
            "recommendations": []
        }
        
        # Generate recommendations
        if violation_stats["by_severity"]["critical"] > 0:
            report["recommendations"].append(
                "URGENT: Critical shortcut violations detected - investigate immediately"
            )
        
        if report["summary"]["clean_rate"] < 0.8:
            report["recommendations"].append(
                "High rate of shortcut violations - review test design and isolation"
            )
        
        if avg_suspicion > 0.3:
            report["recommendations"].append(
                "High average suspicion score - consider additional validation measures"
            )
        
        return report