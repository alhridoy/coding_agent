"""
Enhanced evaluators for comprehensive code quality assessment
"""

import re
import ast
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .eval_framework import EvaluationResult, Severity

logger = logging.getLogger(__name__)


@dataclass
class CodeQualityMetrics:
    """Code quality metrics"""
    complexity_score: float = 0.0
    maintainability_score: float = 0.0
    readability_score: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    overall_score: float = 0.0


@dataclass
class PRQualityMetrics:
    """Pull request quality metrics"""
    title_quality: float = 0.0
    description_quality: float = 0.0
    changes_scope: float = 0.0
    test_coverage: float = 0.0
    documentation_updated: bool = False
    follows_conventions: bool = False


class CodeQualityEvaluator:
    """Evaluates code quality of generated changes"""
    
    def __init__(self):
        self.metrics = CodeQualityMetrics()
        
    async def evaluate(self, result: EvaluationResult) -> CodeQualityMetrics:
        """Evaluate code quality from the result"""
        
        # Extract code changes from logs
        code_changes = self._extract_code_changes(result.logs)
        
        # Analyze different aspects
        complexity = self._analyze_complexity(code_changes)
        maintainability = self._analyze_maintainability(code_changes)
        readability = self._analyze_readability(code_changes)
        security = self._analyze_security(code_changes)
        performance = self._analyze_performance(code_changes)
        
        # Calculate overall score
        overall = (complexity + maintainability + readability + security + performance) / 5
        
        metrics = CodeQualityMetrics(
            complexity_score=complexity,
            maintainability_score=maintainability,
            readability_score=readability,
            security_score=security,
            performance_score=performance,
            overall_score=overall
        )
        
        # Update result metrics
        result.metrics.code_quality_score = overall
        
        return metrics
        
    def _extract_code_changes(self, logs: List[str]) -> List[str]:
        """Extract actual code changes from logs"""
        
        code_changes = []
        
        for log in logs:
            # Look for Tool: Edit events
            if "Tool: Edit" in log or "new_str" in log:
                try:
                    # Try to extract the new code
                    if "new_str" in log:
                        # Extract content between quotes or markers
                        matches = re.findall(r'"new_str":\s*"([^"]*)"', log)
                        if matches:
                            code_changes.extend(matches)
                except Exception:
                    continue
                    
        return code_changes
        
    def _analyze_complexity(self, code_changes: List[str]) -> float:
        """Analyze code complexity"""
        
        if not code_changes:
            return 75.0  # Neutral score for no changes
            
        total_score = 0.0
        valid_changes = 0
        
        for change in code_changes:
            try:
                # Count control structures, nesting levels
                complexity_indicators = [
                    len(re.findall(r'\bif\b', change)),
                    len(re.findall(r'\bfor\b', change)),
                    len(re.findall(r'\bwhile\b', change)),
                    len(re.findall(r'\btry\b', change)),
                    len(re.findall(r'\bexcept\b', change))
                ]
                
                total_complexity = sum(complexity_indicators)
                lines = len(change.split('\n'))
                
                # Score based on complexity density
                if lines > 0:
                    complexity_density = total_complexity / lines
                    # Lower complexity is better
                    score = max(0, 100 - (complexity_density * 50))
                    total_score += score
                    valid_changes += 1
                    
            except Exception:
                continue
                
        return total_score / valid_changes if valid_changes > 0 else 75.0
        
    def _analyze_maintainability(self, code_changes: List[str]) -> float:
        """Analyze code maintainability"""
        
        maintainability_score = 75.0  # Base score
        
        for change in code_changes:
            # Check for good practices
            if any(indicator in change.lower() for indicator in [
                'def ', 'class ', 'import ', 'from ', 'return',
                'docstring', '"""', "'''"
            ]):
                maintainability_score += 5
                
            # Check for bad practices
            if any(indicator in change.lower() for indicator in [
                'todo', 'fixme', 'hack', 'hardcode', 'magic number'
            ]):
                maintainability_score -= 10
                
            # Check for proper structure
            if re.search(r'def\s+\w+\([^)]*\):', change):
                maintainability_score += 3  # Well-defined functions
                
        return min(100, max(0, maintainability_score))
        
    def _analyze_readability(self, code_changes: List[str]) -> float:
        """Analyze code readability"""
        
        readability_score = 75.0
        
        for change in code_changes:
            # Check for comments and documentation
            comment_lines = len(re.findall(r'#.*', change))
            docstring_blocks = len(re.findall(r'""".*?"""', change, re.DOTALL))
            total_lines = len(change.split('\n'))
            
            if total_lines > 0:
                comment_ratio = (comment_lines + docstring_blocks * 3) / total_lines
                readability_score += min(20, comment_ratio * 50)
                
            # Check for meaningful variable names
            variable_names = re.findall(r'\b[a-z_][a-z0-9_]*\s*=', change.lower())
            meaningful_names = [name for name in variable_names 
                              if len(name.strip(' =')) > 2 and name.strip(' =') not in ['x', 'y', 'i', 'j']]
            
            if variable_names:
                name_quality = len(meaningful_names) / len(variable_names)
                readability_score += name_quality * 10
                
        return min(100, max(0, readability_score))
        
    def _analyze_security(self, code_changes: List[str]) -> float:
        """Analyze security aspects of code changes"""
        
        security_score = 90.0  # Start with high security score
        
        security_issues = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess\.call',
            r'os\.system',
            r'sql.*\+.*\+',  # SQL injection patterns
            r'password\s*=\s*["\'][^"\']*["\']',  # Hardcoded passwords
            r'api[_-]?key\s*=\s*["\'][^"\']*["\']',  # Hardcoded API keys
        ]
        
        for change in code_changes:
            for pattern in security_issues:
                if re.search(pattern, change, re.IGNORECASE):
                    security_score -= 20
                    logger.warning(f"Security issue detected: {pattern}")
                    
            # Check for good security practices
            if any(practice in change.lower() for practice in [
                'hash', 'encrypt', 'validate', 'sanitize', 'escape'
            ]):
                security_score += 5
                
        return min(100, max(0, security_score))
        
    def _analyze_performance(self, code_changes: List[str]) -> float:
        """Analyze performance implications"""
        
        performance_score = 80.0
        
        for change in code_changes:
            # Check for performance anti-patterns
            performance_issues = [
                r'for.*in.*range\(len\(',  # Inefficient iteration
                r'list\(.*\)\[0\]',  # Inefficient list operations
                r'\.append\(.*\)\s*\n.*\.append',  # Multiple appends
            ]
            
            for pattern in performance_issues:
                if re.search(pattern, change):
                    performance_score -= 10
                    
            # Check for good performance practices
            if any(practice in change.lower() for practice in [
                'cache', 'optimize', 'efficient', 'async', 'await'
            ]):
                performance_score += 5
                
        return min(100, max(0, performance_score))


class PRQualityEvaluator:
    """Evaluates pull request quality"""
    
    async def evaluate(self, result: EvaluationResult) -> PRQualityMetrics:
        """Evaluate PR quality"""
        
        if not result.pr_url:
            return PRQualityMetrics()
            
        # Extract PR information from logs
        pr_info = self._extract_pr_info(result.logs)
        
        title_quality = self._evaluate_title_quality(pr_info.get('title', ''))
        description_quality = self._evaluate_description_quality(pr_info.get('body', ''))
        changes_scope = self._evaluate_changes_scope(result)
        
        metrics = PRQualityMetrics(
            title_quality=title_quality,
            description_quality=description_quality,
            changes_scope=changes_scope,
            test_coverage=self._estimate_test_coverage(result),
            documentation_updated=self._check_documentation_updated(result),
            follows_conventions=self._check_conventions(pr_info)
        )
        
        # Calculate overall PR quality score
        overall_pr_score = (
            title_quality * 0.2 +
            description_quality * 0.3 +
            changes_scope * 0.2 +
            metrics.test_coverage * 0.1 +
            (100 if metrics.documentation_updated else 0) * 0.1 +
            (100 if metrics.follows_conventions else 0) * 0.1
        )
        
        result.metrics.pr_quality_score = overall_pr_score
        
        return metrics
        
    def _extract_pr_info(self, logs: List[str]) -> Dict[str, str]:
        """Extract PR information from logs"""
        
        pr_info = {"title": "", "body": ""}
        
        for log in logs:
            if "pr_title" in log.lower():
                # Extract title
                title_match = re.search(r'"title":\s*"([^"]*)"', log)
                if title_match:
                    pr_info["title"] = title_match.group(1)
                    
            elif "pr_body" in log.lower() or "body" in log.lower():
                # Extract body
                body_match = re.search(r'"body":\s*"([^"]*)"', log)
                if body_match:
                    pr_info["body"] = body_match.group(1)
                    
        return pr_info
        
    def _evaluate_title_quality(self, title: str) -> float:
        """Evaluate PR title quality"""
        
        if not title:
            return 0.0
            
        score = 50.0  # Base score
        
        # Check length
        if 10 <= len(title) <= 72:
            score += 20
        elif len(title) < 10:
            score -= 20
            
        # Check for conventional commit format
        if re.match(r'^(feat|fix|docs|style|refactor|test|chore):', title.lower()):
            score += 15
            
        # Check for descriptive content
        if any(word in title.lower() for word in [
            'add', 'implement', 'fix', 'update', 'improve', 'enhance'
        ]):
            score += 10
            
        # Avoid vague titles
        if any(vague in title.lower() for vague in [
            'update', 'change', 'fix stuff', 'misc'
        ]):
            score -= 10
            
        return min(100, max(0, score))
        
    def _evaluate_description_quality(self, description: str) -> float:
        """Evaluate PR description quality"""
        
        if not description:
            return 30.0  # Some points for having a description
            
        score = 40.0
        
        # Check length
        if len(description) > 50:
            score += 20
            
        # Check for structure
        if any(section in description.lower() for section in [
            'summary', 'changes', 'testing', 'motivation'
        ]):
            score += 20
            
        # Check for technical details
        if any(detail in description.lower() for detail in [
            'implementation', 'algorithm', 'performance', 'security'
        ]):
            score += 10
            
        # Check for testing information
        if any(test_info in description.lower() for test_info in [
            'test', 'tested', 'coverage', 'validation'
        ]):
            score += 10
            
        return min(100, max(0, score))
        
    def _evaluate_changes_scope(self, result: EvaluationResult) -> float:
        """Evaluate the scope and appropriateness of changes"""
        
        files_modified = result.metrics.files_modified
        
        if files_modified == 0:
            return 0.0
            
        # Ideal range is 1-5 files for most changes
        if 1 <= files_modified <= 5:
            return 100.0
        elif files_modified <= 10:
            return 80.0
        elif files_modified <= 20:
            return 60.0
        else:
            return 30.0  # Too many files might indicate overly broad changes
            
    def _estimate_test_coverage(self, result: EvaluationResult) -> float:
        """Estimate test coverage from changes"""
        
        # Look for test-related activity in logs
        test_indicators = 0
        
        for log in result.logs:
            if any(test_word in log.lower() for test_word in [
                'test', 'spec', 'pytest', 'unittest', 'jest'
            ]):
                test_indicators += 1
                
        # Simple heuristic: more test activity = better coverage
        return min(100, test_indicators * 10)
        
    def _check_documentation_updated(self, result: EvaluationResult) -> bool:
        """Check if documentation was updated"""
        
        for log in result.logs:
            if any(doc_file in log.lower() for doc_file in [
                'readme', 'docs', 'documentation', '.md'
            ]):
                return True
                
        return False
        
    def _check_conventions(self, pr_info: Dict[str, str]) -> bool:
        """Check if PR follows conventions"""
        
        title = pr_info.get('title', '')
        body = pr_info.get('body', '')
        
        # Check title conventions
        has_good_title = (
            len(title) >= 10 and
            not title.endswith('.') and
            title[0].isupper()
        )
        
        # Check body conventions
        has_good_body = len(body) >= 20
        
        return has_good_title and has_good_body


class WorkflowEvaluator:
    """Evaluates the overall workflow execution"""
    
    async def evaluate(self, result: EvaluationResult) -> Dict[str, Any]:
        """Evaluate workflow execution"""
        
        workflow_metrics = {
            "steps_completed": self._count_completed_steps(result),
            "error_recovery": self._evaluate_error_recovery(result),
            "efficiency": self._evaluate_efficiency(result),
            "completeness": self._evaluate_completeness(result)
        }
        
        return workflow_metrics
        
    def _count_completed_steps(self, result: EvaluationResult) -> int:
        """Count workflow steps completed"""
        
        expected_steps = [
            "cloning", "analyzing", "modifying", "committing", "pushing", "pr"
        ]
        
        completed_steps = 0
        log_text = " ".join(result.logs).lower()
        
        for step in expected_steps:
            if step in log_text:
                completed_steps += 1
                
        return completed_steps
        
    def _evaluate_error_recovery(self, result: EvaluationResult) -> float:
        """Evaluate error recovery capability"""
        
        if result.metrics.error_count == 0:
            return 100.0
            
        # Check if errors were handled gracefully
        recovery_indicators = 0
        
        for log in result.logs:
            if any(indicator in log.lower() for indicator in [
                'retry', 'fallback', 'alternative', 'recovering'
            ]):
                recovery_indicators += 1
                
        # Score based on recovery attempts vs errors
        recovery_score = min(100, (recovery_indicators / result.metrics.error_count) * 100)
        
        return recovery_score
        
    def _evaluate_efficiency(self, result: EvaluationResult) -> float:
        """Evaluate workflow efficiency"""
        
        execution_time = result.metrics.execution_time
        
        # Efficiency based on time taken
        if execution_time < 60:  # Under 1 minute
            return 100.0
        elif execution_time < 300:  # Under 5 minutes
            return 80.0
        elif execution_time < 600:  # Under 10 minutes
            return 60.0
        else:
            return 30.0
            
    def _evaluate_completeness(self, result: EvaluationResult) -> float:
        """Evaluate workflow completeness"""
        
        completeness_score = 0.0
        
        # Check for expected outcomes
        if result.pr_url:
            completeness_score += 50  # Main goal achieved
            
        if result.metrics.files_modified > 0:
            completeness_score += 25  # Changes made
            
        if result.metrics.error_count == 0:
            completeness_score += 15  # No errors
            
        if result.metrics.execution_time < 600:  # Under 10 minutes
            completeness_score += 10  # Reasonable time
            
        return completeness_score


class SecurityEvaluator:
    """Evaluates security aspects of the generated code"""
    
    def __init__(self):
        self.security_patterns = {
            "high_risk": [
                r"eval\s*\(",
                r"exec\s*\(",
                r"__import__\s*\(",
                r"os\.system\s*\(",
                r"subprocess\.call.*shell\s*=\s*True"
            ],
            "medium_risk": [
                r"password\s*=\s*['\"][^'\"]*['\"]",
                r"api[_-]?key\s*=\s*['\"][^'\"]*['\"]",
                r"secret\s*=\s*['\"][^'\"]*['\"]",
                r"sql.*\+.*\+",  # SQL injection potential
            ],
            "low_risk": [
                r"print\s*\(.*password",
                r"logging.*password",
                r"console\.log.*password"
            ]
        }
        
    async def evaluate(self, result: EvaluationResult) -> Dict[str, Any]:
        """Evaluate security aspects"""
        
        code_changes = self._extract_code_from_logs(result.logs)
        
        security_issues = []
        
        for code in code_changes:
            issues = self._scan_for_security_issues(code)
            security_issues.extend(issues)
            
        security_score = self._calculate_security_score(security_issues)
        
        return {
            "security_score": security_score,
            "issues_found": security_issues,
            "risk_level": self._determine_risk_level(security_issues)
        }
        
    def _extract_code_from_logs(self, logs: List[str]) -> List[str]:
        """Extract code snippets from logs"""
        
        code_snippets = []
        
        for log in logs:
            # Look for code in edit operations
            if "new_str" in log:
                try:
                    # Extract code between quotes
                    matches = re.findall(r'"new_str":\s*"([^"]*)"', log)
                    code_snippets.extend(matches)
                except Exception:
                    continue
                    
        return code_snippets
        
    def _scan_for_security_issues(self, code: str) -> List[Dict[str, Any]]:
        """Scan code for security issues"""
        
        issues = []
        
        for risk_level, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                
                for match in matches:
                    issues.append({
                        "type": "security_vulnerability",
                        "risk_level": risk_level,
                        "pattern": pattern,
                        "location": match.span(),
                        "code_snippet": match.group(0)
                    })
                    
        return issues
        
    def _calculate_security_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate overall security score"""
        
        if not issues:
            return 100.0
            
        score = 100.0
        
        for issue in issues:
            risk_level = issue["risk_level"]
            
            if risk_level == "high_risk":
                score -= 30
            elif risk_level == "medium_risk":
                score -= 15
            elif risk_level == "low_risk":
                score -= 5
                
        return max(0, score)
        
    def _determine_risk_level(self, issues: List[Dict[str, Any]]) -> str:
        """Determine overall risk level"""
        
        if not issues:
            return "low"
            
        risk_levels = [issue["risk_level"] for issue in issues]
        
        if "high_risk" in risk_levels:
            return "high"
        elif "medium_risk" in risk_levels:
            return "medium"
        else:
            return "low"