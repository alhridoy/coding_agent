"""
PR Validation System for rigorous outcome validation
Based on ABC (Agentic Benchmark Checklist) best practices
"""

import re
import ast
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import difflib
from github import Github
import httpx

logger = logging.getLogger(__name__)


@dataclass
class PRValidationResult:
    """Detailed PR validation results"""
    is_valid: bool
    confidence_score: float  # 0-1 score indicating confidence in validation
    validation_details: Dict[str, Any]
    issues_found: List[str]
    suggestions: List[str]


@dataclass
class CodeChange:
    """Represents a code change in the PR"""
    file_path: str
    old_content: str
    new_content: str
    diff_lines: List[str]
    change_type: str  # 'added', 'modified', 'deleted'


class PRValidator:
    """
    Comprehensive PR validation beyond simple URL checking.
    Implements ABC outcome validity principles.
    """
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
        self.github_client = Github(github_token) if github_token else None
        
    async def validate_pr(
        self, 
        pr_url: str,
        test_case: 'TestCase',
        agent_logs: List[str]
    ) -> PRValidationResult:
        """
        Comprehensively validate a PR against test case requirements.
        
        This goes beyond checking if a PR exists - it validates:
        1. PR actually contains relevant changes
        2. Changes match the task requirements
        3. No harmful or irrelevant changes
        4. Code quality meets standards
        """
        logger.info(f"Validating PR: {pr_url} for test case: {test_case.id}")
        
        validation_results = {
            "pr_exists": False,
            "changes_relevant": False,
            "requirements_met": False,
            "no_harmful_changes": True,
            "code_quality_acceptable": False,
            "matches_prompt_intent": False
        }
        
        issues = []
        suggestions = []
        
        try:
            # Step 1: Verify PR exists and is accessible
            pr_data = await self._fetch_pr_data(pr_url)
            if not pr_data:
                issues.append("PR URL is invalid or inaccessible")
                return PRValidationResult(
                    is_valid=False,
                    confidence_score=0.0,
                    validation_details=validation_results,
                    issues_found=issues,
                    suggestions=["Ensure PR was created successfully"]
                )
            
            validation_results["pr_exists"] = True
            
            # Step 2: Extract and analyze code changes
            code_changes = await self._extract_code_changes(pr_data, agent_logs)
            
            # Step 3: Validate changes are relevant to the task
            relevance_score = self._validate_change_relevance(
                code_changes, 
                test_case.prompt,
                test_case.expected_outcome
            )
            validation_results["changes_relevant"] = relevance_score > 0.7
            
            if relevance_score < 0.7:
                issues.append(f"Changes may not be relevant to task (relevance: {relevance_score:.2f})")
                suggestions.append("Ensure changes directly address the task requirements")
            
            # Step 4: Check if requirements are met
            requirements_met = self._validate_requirements_met(
                code_changes,
                test_case.expected_outcome
            )
            validation_results["requirements_met"] = requirements_met["all_met"]
            
            for req, met in requirements_met["details"].items():
                if not met:
                    issues.append(f"Requirement not met: {req}")
            
            # Step 5: Check for harmful changes
            harmful_changes = self._detect_harmful_changes(code_changes)
            validation_results["no_harmful_changes"] = len(harmful_changes) == 0
            
            for harmful in harmful_changes:
                issues.append(f"Harmful change detected: {harmful}")
                validation_results["no_harmful_changes"] = False
            
            # Step 6: Validate code quality
            quality_score = self._validate_code_quality(code_changes)
            validation_results["code_quality_acceptable"] = quality_score > 0.6
            
            if quality_score < 0.6:
                issues.append(f"Code quality below threshold (score: {quality_score:.2f})")
                suggestions.append("Improve code structure, naming, and documentation")
            
            # Step 7: Validate prompt intent matching
            intent_score = self._validate_prompt_intent(
                code_changes,
                test_case.prompt,
                pr_data.get("title", ""),
                pr_data.get("body", "")
            )
            validation_results["matches_prompt_intent"] = intent_score > 0.8
            
            # Calculate overall confidence score
            confidence_score = self._calculate_confidence_score(validation_results)
            
            # Determine if PR is valid
            is_valid = (
                validation_results["pr_exists"] and
                validation_results["changes_relevant"] and
                validation_results["requirements_met"] and
                validation_results["no_harmful_changes"] and
                confidence_score > 0.7
            )
            
            return PRValidationResult(
                is_valid=is_valid,
                confidence_score=confidence_score,
                validation_details=validation_results,
                issues_found=issues,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error validating PR: {e}")
            issues.append(f"Validation error: {str(e)}")
            return PRValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_details=validation_results,
                issues_found=issues,
                suggestions=["Fix validation errors and retry"]
            )
    
    async def _fetch_pr_data(self, pr_url: str) -> Optional[Dict[str, Any]]:
        """Fetch PR data from GitHub"""
        try:
            # Parse PR URL
            match = re.match(r'https://github.com/([^/]+)/([^/]+)/pull/(\d+)', pr_url)
            if not match:
                return None
            
            owner, repo, pr_number = match.groups()
            
            if self.github_client:
                # Use GitHub API
                repo_obj = self.github_client.get_repo(f"{owner}/{repo}")
                pr = repo_obj.get_pull(int(pr_number))
                
                # Get files changed
                files = []
                for file in pr.get_files():
                    files.append({
                        "filename": file.filename,
                        "status": file.status,
                        "additions": file.additions,
                        "deletions": file.deletions,
                        "changes": file.changes,
                        "patch": file.patch
                    })
                
                return {
                    "number": pr.number,
                    "title": pr.title,
                    "body": pr.body,
                    "state": pr.state,
                    "files": files,
                    "commits": pr.commits,
                    "additions": pr.additions,
                    "deletions": pr.deletions,
                    "changed_files": pr.changed_files
                }
            else:
                # Fallback to basic HTTP request
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
                    )
                    if response.status_code == 200:
                        return response.json()
                    
            return None
            
        except Exception as e:
            logger.error(f"Error fetching PR data: {e}")
            return None
    
    async def _extract_code_changes(
        self, 
        pr_data: Dict[str, Any],
        agent_logs: List[str]
    ) -> List[CodeChange]:
        """Extract code changes from PR data and agent logs"""
        changes = []
        
        # Extract from PR data if available
        if "files" in pr_data:
            for file in pr_data["files"]:
                if file.get("patch"):
                    change = CodeChange(
                        file_path=file["filename"],
                        old_content="",  # Would need to fetch separately
                        new_content="",  # Would need to fetch separately
                        diff_lines=file["patch"].split('\n'),
                        change_type=file["status"]
                    )
                    changes.append(change)
        
        # Also extract from agent logs as backup
        for log in agent_logs:
            if "Tool: Edit" in log or "new_str" in log:
                try:
                    # Extract file path and content
                    if "filepath" in log:
                        filepath_match = re.search(r'"filepath":\s*"([^"]+)"', log)
                        old_str_match = re.search(r'"old_str":\s*"([^"]*)"', log)
                        new_str_match = re.search(r'"new_str":\s*"([^"]*)"', log)
                        
                        if filepath_match and new_str_match:
                            change = CodeChange(
                                file_path=filepath_match.group(1),
                                old_content=old_str_match.group(1) if old_str_match else "",
                                new_content=new_str_match.group(1),
                                diff_lines=[],
                                change_type="modified"
                            )
                            changes.append(change)
                except Exception as e:
                    logger.warning(f"Error extracting change from log: {e}")
        
        return changes
    
    def _validate_change_relevance(
        self,
        code_changes: List[CodeChange],
        prompt: str,
        expected_outcome: Dict[str, Any]
    ) -> float:
        """Validate that changes are relevant to the task"""
        if not code_changes:
            return 0.0
        
        relevance_score = 0.0
        total_checks = 0
        
        # Check 1: Files modified match expected files
        if "files_modified" in expected_outcome:
            expected_files = expected_outcome["files_modified"]
            actual_files = [change.file_path for change in code_changes]
            
            matching_files = sum(
                1 for f in actual_files 
                if any(exp in f for exp in expected_files)
            )
            
            if expected_files:
                relevance_score += matching_files / len(expected_files)
                total_checks += 1
        
        # Check 2: Keywords from prompt appear in changes
        prompt_keywords = self._extract_keywords(prompt)
        keyword_matches = 0
        
        for change in code_changes:
            change_text = change.new_content.lower()
            for keyword in prompt_keywords:
                if keyword.lower() in change_text:
                    keyword_matches += 1
                    break
        
        if prompt_keywords and code_changes:
            relevance_score += keyword_matches / len(code_changes)
            total_checks += 1
        
        # Check 3: Change patterns match task type
        task_patterns = self._identify_task_patterns(prompt)
        pattern_matches = 0
        
        for change in code_changes:
            for pattern in task_patterns:
                if re.search(pattern, change.new_content, re.IGNORECASE):
                    pattern_matches += 1
                    break
        
        if task_patterns and code_changes:
            relevance_score += pattern_matches / len(code_changes)
            total_checks += 1
        
        return relevance_score / total_checks if total_checks > 0 else 0.5
    
    def _validate_requirements_met(
        self,
        code_changes: List[CodeChange],
        expected_outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if specific requirements are met"""
        requirements_met = {"all_met": True, "details": {}}
        
        # Check each expected outcome
        for requirement, expected in expected_outcome.items():
            if requirement == "pr_created":
                # This is handled elsewhere
                requirements_met["details"][requirement] = True
                continue
                
            elif requirement == "files_modified":
                actual_files = [c.file_path for c in code_changes]
                met = any(
                    any(exp in actual for exp in expected)
                    for actual in actual_files
                )
                requirements_met["details"][requirement] = met
                
            elif requirement == "validation_added":
                # Check for validation patterns
                validation_patterns = [
                    r'validate',
                    r'check',
                    r'verify',
                    r'assert',
                    r'require',
                    r'schema',
                    r'ValidationError'
                ]
                met = any(
                    any(re.search(pattern, c.new_content, re.IGNORECASE) 
                        for pattern in validation_patterns)
                    for c in code_changes
                )
                requirements_met["details"][requirement] = met
                
            elif requirement == "endpoint_added":
                # Check for endpoint patterns
                endpoint_patterns = [
                    r'@app\.(get|post|put|delete|patch)',
                    r'router\.(get|post|put|delete|patch)',
                    r'route\s*\(',
                    r'endpoints?\s*=',
                    r'path\s*\('
                ]
                met = any(
                    any(re.search(pattern, c.new_content, re.IGNORECASE) 
                        for pattern in endpoint_patterns)
                    for c in code_changes
                )
                requirements_met["details"][requirement] = met
                
            else:
                # Generic check - look for the requirement keyword in changes
                met = any(
                    requirement.lower() in c.new_content.lower()
                    for c in code_changes
                )
                requirements_met["details"][requirement] = met
            
            if not requirements_met["details"][requirement]:
                requirements_met["all_met"] = False
        
        return requirements_met
    
    def _detect_harmful_changes(self, code_changes: List[CodeChange]) -> List[str]:
        """Detect potentially harmful or suspicious changes"""
        harmful_changes = []
        
        harmful_patterns = [
            (r'rm\s+-rf\s+/', "Dangerous file deletion command"),
            (r'eval\s*\(', "Use of eval() is a security risk"),
            (r'exec\s*\(', "Use of exec() is a security risk"),
            (r'__import__', "Dynamic imports can be dangerous"),
            (r'os\.system', "Direct system calls are risky"),
            (r'subprocess.*shell\s*=\s*True', "Shell injection risk"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
            (r'DROP\s+TABLE', "Dangerous SQL operation"),
            (r'DELETE\s+FROM.*WHERE\s+1\s*=\s*1', "Dangerous SQL delete all"),
        ]
        
        for change in code_changes:
            for pattern, description in harmful_patterns:
                if re.search(pattern, change.new_content, re.IGNORECASE):
                    harmful_changes.append(
                        f"{description} in {change.file_path}"
                    )
        
        # Check for files that shouldn't be modified
        protected_files = [
            '.github/workflows',
            'requirements.txt',
            'package.json',
            'Gemfile',
            '.gitignore',
            'LICENSE'
        ]
        
        for change in code_changes:
            for protected in protected_files:
                if protected in change.file_path and change.change_type != 'added':
                    harmful_changes.append(
                        f"Modification to protected file: {change.file_path}"
                    )
        
        return harmful_changes
    
    def _validate_code_quality(self, code_changes: List[CodeChange]) -> float:
        """Validate code quality of changes"""
        if not code_changes:
            return 0.0
        
        quality_score = 0.0
        total_checks = 0
        
        for change in code_changes:
            if not change.new_content.strip():
                continue
                
            # Check 1: Proper indentation (simplified)
            lines = change.new_content.split('\n')
            properly_indented = all(
                line == line.lstrip() or line.lstrip() == line[4:] or line.lstrip() == line[2:]
                for line in lines if line.strip()
            )
            quality_score += 1.0 if properly_indented else 0.5
            total_checks += 1
            
            # Check 2: No extremely long lines
            long_lines = sum(1 for line in lines if len(line) > 120)
            quality_score += 1.0 if long_lines == 0 else max(0, 1 - long_lines / len(lines))
            total_checks += 1
            
            # Check 3: Meaningful variable names (heuristic)
            var_pattern = r'\b([a-z_][a-z0-9_]*)\s*='
            variables = re.findall(var_pattern, change.new_content.lower())
            meaningful_vars = sum(
                1 for var in variables 
                if len(var) > 2 and var not in ['tmp', 'temp', 'val', 'var', 'foo', 'bar']
            )
            if variables:
                quality_score += meaningful_vars / len(variables)
                total_checks += 1
            
            # Check 4: Has comments or docstrings
            has_comments = (
                '#' in change.new_content or 
                '"""' in change.new_content or 
                "'''" in change.new_content or
                '//' in change.new_content or
                '/*' in change.new_content
            )
            quality_score += 1.0 if has_comments else 0.3
            total_checks += 1
        
        return quality_score / total_checks if total_checks > 0 else 0.5
    
    def _validate_prompt_intent(
        self,
        code_changes: List[CodeChange],
        prompt: str,
        pr_title: str,
        pr_body: str
    ) -> float:
        """Validate that changes match the intent of the prompt"""
        intent_score = 0.0
        total_checks = 0
        
        # Extract key actions from prompt
        action_words = ['add', 'implement', 'create', 'fix', 'update', 'remove', 'modify', 'enhance']
        prompt_actions = []
        
        for action in action_words:
            if action in prompt.lower():
                prompt_actions.append(action)
        
        # Check if PR title/body reflects the prompt
        pr_text = f"{pr_title} {pr_body}".lower()
        title_matches = sum(1 for action in prompt_actions if action in pr_text)
        
        if prompt_actions:
            intent_score += title_matches / len(prompt_actions)
            total_checks += 1
        
        # Check if code changes align with prompt action
        if 'add' in prompt_actions or 'create' in prompt_actions:
            # Should see new code additions
            has_additions = any(
                len(c.new_content) > len(c.old_content) 
                for c in code_changes
            )
            intent_score += 1.0 if has_additions else 0.0
            total_checks += 1
            
        if 'remove' in prompt_actions or 'delete' in prompt_actions:
            # Should see code removals
            has_deletions = any(
                len(c.new_content) < len(c.old_content)
                for c in code_changes
            )
            intent_score += 1.0 if has_deletions else 0.0
            total_checks += 1
        
        # Check specific intent keywords
        intent_keywords = self._extract_intent_keywords(prompt)
        keyword_matches = sum(
            1 for keyword in intent_keywords
            if any(keyword in c.new_content.lower() for c in code_changes)
        )
        
        if intent_keywords:
            intent_score += keyword_matches / len(intent_keywords)
            total_checks += 1
        
        return intent_score / total_checks if total_checks > 0 else 0.5
    
    def _calculate_confidence_score(self, validation_results: Dict[str, bool]) -> float:
        """Calculate overall confidence score"""
        weights = {
            "pr_exists": 0.1,
            "changes_relevant": 0.25,
            "requirements_met": 0.3,
            "no_harmful_changes": 0.15,
            "code_quality_acceptable": 0.1,
            "matches_prompt_intent": 0.1
        }
        
        score = sum(
            weights[key] * (1.0 if value else 0.0)
            for key, value in validation_results.items()
            if key in weights
        )
        
        return score
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common words
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'as', 'that', 'this', 'it', 'be',
            'are', 'was', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Filter keywords
        keywords = [
            word for word in words 
            if len(word) > 3 and word not in stopwords
        ]
        
        # Return unique keywords
        return list(set(keywords))[:10]  # Limit to top 10
    
    def _extract_intent_keywords(self, prompt: str) -> List[str]:
        """Extract intent-specific keywords"""
        # Technical terms that indicate specific intent
        technical_terms = [
            'validation', 'authentication', 'authorization', 'api', 'endpoint',
            'database', 'cache', 'logging', 'error', 'security', 'test',
            'performance', 'optimization', 'configuration', 'middleware'
        ]
        
        keywords = []
        prompt_lower = prompt.lower()
        
        for term in technical_terms:
            if term in prompt_lower:
                keywords.append(term)
        
        return keywords
    
    def _identify_task_patterns(self, prompt: str) -> List[str]:
        """Identify regex patterns based on task type"""
        patterns = []
        
        prompt_lower = prompt.lower()
        
        if 'validation' in prompt_lower:
            patterns.extend([
                r'validate\w*',
                r'check\w*',
                r'verify\w*',
                r'is_valid',
                r'ValidationError'
            ])
            
        if 'endpoint' in prompt_lower or 'api' in prompt_lower:
            patterns.extend([
                r'@(app|router)\.',
                r'def\s+\w+.*request',
                r'return\s+.*response',
                r'status_code'
            ])
            
        if 'test' in prompt_lower:
            patterns.extend([
                r'def\s+test_',
                r'assert\s+',
                r'@pytest',
                r'describe\(',
                r'it\('
            ])
            
        if 'security' in prompt_lower:
            patterns.extend([
                r'authenticate',
                r'authorize',
                r'permission',
                r'csrf',
                r'sanitize'
            ])
        
        return patterns


# Integration with existing evaluation framework
class EnhancedEvaluationResult:
    """Enhanced evaluation result with PR validation"""
    
    def __init__(
        self,
        original_result: 'EvaluationResult',
        pr_validation: Optional[PRValidationResult] = None
    ):
        self.original_result = original_result
        self.pr_validation = pr_validation
        
    @property
    def is_truly_successful(self) -> bool:
        """Determine if the test truly succeeded based on comprehensive validation"""
        return (
            self.original_result.status == 'passed' and
            self.pr_validation is not None and
            self.pr_validation.is_valid
        )
    
    @property
    def confidence_score(self) -> float:
        """Get confidence in the result"""
        if self.pr_validation:
            return self.pr_validation.confidence_score
        return 0.0 if self.original_result.status == 'failed' else 0.5
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """Get detailed validation report"""
        report = {
            "test_id": self.original_result.test_case.id,
            "original_status": self.original_result.status,
            "pr_created": self.original_result.pr_url is not None,
            "pr_url": self.original_result.pr_url,
            "truly_successful": self.is_truly_successful,
            "confidence_score": self.confidence_score
        }
        
        if self.pr_validation:
            report["validation_details"] = self.pr_validation.validation_details
            report["validation_issues"] = self.pr_validation.issues_found
            report["suggestions"] = self.pr_validation.suggestions
        
        return report