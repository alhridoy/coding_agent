"""
Task Specification Validator with Oracle Solvers
Implements ABC task validity principles to ensure tasks are well-defined and solvable
"""

import re
import ast
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TaskValidationResult:
    """Result of task specification validation"""
    is_valid: bool
    is_solvable: bool
    specificity_score: float  # 0-1, how specific the task is
    ambiguity_issues: List[str]
    missing_specifications: List[str]
    oracle_solution: Optional[Dict[str, Any]]
    estimated_difficulty: str  # 'easy', 'medium', 'hard', 'expert'


@dataclass 
class OracleSolution:
    """Represents an oracle (ideal) solution for a task"""
    approach: str
    expected_files: List[str]
    expected_patterns: List[str]
    key_implementations: List[str]
    validation_criteria: Dict[str, Any]


class OracleSolver(ABC):
    """Base class for oracle solvers that verify task solvability"""
    
    @abstractmethod
    async def can_solve(self, test_case: 'TestCase') -> Tuple[bool, Optional[OracleSolution]]:
        """Check if the oracle can solve this task"""
        pass
    
    @abstractmethod
    def get_solver_type(self) -> str:
        """Get the type of problems this solver handles"""
        pass


class APIEndpointOracleSolver(OracleSolver):
    """Oracle solver for API endpoint tasks"""
    
    async def can_solve(self, test_case: 'TestCase') -> Tuple[bool, Optional[OracleSolution]]:
        """Verify API endpoint tasks are solvable"""
        prompt_lower = test_case.prompt.lower()
        
        # Check if this is an API task
        api_keywords = ['endpoint', 'api', 'route', 'http', 'rest']
        if not any(keyword in prompt_lower for keyword in api_keywords):
            return False, None
        
        # Determine what type of endpoint
        endpoint_type = self._determine_endpoint_type(prompt_lower)
        if not endpoint_type:
            return False, None
        
        # Build oracle solution
        solution = OracleSolution(
            approach=f"Create {endpoint_type['method']} endpoint at {endpoint_type['path']}",
            expected_files=self._get_expected_files(test_case.repo_url),
            expected_patterns=self._get_endpoint_patterns(endpoint_type),
            key_implementations=[
                f"Define {endpoint_type['method']} route",
                "Add request handling logic",
                "Return appropriate response"
            ],
            validation_criteria={
                "has_route_definition": True,
                "has_handler_function": True,
                "returns_response": True,
                "method_matches": endpoint_type['method']
            }
        )
        
        return True, solution
    
    def get_solver_type(self) -> str:
        return "api_endpoint"
    
    def _determine_endpoint_type(self, prompt: str) -> Optional[Dict[str, str]]:
        """Determine endpoint details from prompt"""
        # Method detection
        methods = {
            'get': ['get', 'fetch', 'retrieve', 'list', 'read'],
            'post': ['post', 'create', 'add', 'submit'],
            'put': ['put', 'update', 'modify'],
            'delete': ['delete', 'remove'],
            'patch': ['patch', 'partial']
        }
        
        detected_method = 'get'  # default
        for method, keywords in methods.items():
            if any(kw in prompt for kw in keywords):
                detected_method = method
                break
        
        # Path detection
        path_match = re.search(r'/[\w/]+', prompt)
        path = path_match.group(0) if path_match else '/api/endpoint'
        
        # Specific endpoint patterns
        if 'health' in prompt:
            return {'method': 'get', 'path': '/health'}
        elif 'status' in prompt:
            return {'method': 'get', 'path': '/status'}
        elif 'user' in prompt:
            return {'method': detected_method, 'path': '/users'}
        
        return {'method': detected_method, 'path': path}
    
    def _get_expected_files(self, repo_url: str) -> List[str]:
        """Get expected files based on repo type"""
        if 'flask' in repo_url.lower():
            return ['app.py', 'routes.py', 'main.py']
        elif 'express' in repo_url.lower():
            return ['index.js', 'app.js', 'routes/index.js']
        elif 'fastapi' in repo_url.lower():
            return ['main.py', 'app.py', 'routers/']
        elif 'django' in repo_url.lower():
            return ['views.py', 'urls.py']
        else:
            return ['main.*', 'app.*', 'server.*']
    
    def _get_endpoint_patterns(self, endpoint_type: Dict[str, str]) -> List[str]:
        """Get regex patterns for endpoint validation"""
        method = endpoint_type['method']
        path = endpoint_type['path']
        
        patterns = [
            # Flask patterns
            f"@app\\.{method}\\(['\"]?{re.escape(path)}",
            f"@app\\.route\\(['\"]?{re.escape(path)}.*methods=\\[['\"]?{method.upper()}",
            
            # Express patterns  
            f"app\\.{method}\\(['\"]?{re.escape(path)}",
            f"router\\.{method}\\(['\"]?{re.escape(path)}",
            
            # FastAPI patterns
            f"@app\\.{method}\\(['\"]?{re.escape(path)}",
            f"@router\\.{method}\\(['\"]?{re.escape(path)}",
            
            # Generic patterns
            f"def.*{method}.*{path.replace('/', '_')}",
            f"function.*{method}.*{path.replace('/', '_')}"
        ]
        
        return patterns


class ValidationOracleSolver(OracleSolver):
    """Oracle solver for validation tasks"""
    
    async def can_solve(self, test_case: 'TestCase') -> Tuple[bool, Optional[OracleSolution]]:
        """Verify validation tasks are solvable"""
        prompt_lower = test_case.prompt.lower()
        
        # Check if this is a validation task
        validation_keywords = ['validat', 'check', 'verify', 'ensure', 'confirm']
        if not any(keyword in prompt_lower for keyword in validation_keywords):
            return False, None
        
        # Determine validation type
        validation_type = self._determine_validation_type(prompt_lower)
        
        solution = OracleSolution(
            approach=f"Implement {validation_type} validation",
            expected_files=["validators.py", "validation.py", "models.py", "schemas.py"],
            expected_patterns=self._get_validation_patterns(validation_type),
            key_implementations=[
                "Define validation rules",
                "Implement validation logic", 
                "Handle validation errors",
                "Return validation results"
            ],
            validation_criteria={
                "has_validation_logic": True,
                "handles_invalid_input": True,
                "returns_errors": True,
                "validates_correct_fields": True
            }
        )
        
        return True, solution
    
    def get_solver_type(self) -> str:
        return "validation"
    
    def _determine_validation_type(self, prompt: str) -> str:
        """Determine what type of validation is needed"""
        if 'input' in prompt:
            return "input"
        elif 'email' in prompt:
            return "email"
        elif 'password' in prompt:
            return "password"
        elif 'form' in prompt:
            return "form"
        elif 'request' in prompt:
            return "request"
        elif 'data' in prompt:
            return "data"
        else:
            return "general"
    
    def _get_validation_patterns(self, validation_type: str) -> List[str]:
        """Get validation patterns"""
        base_patterns = [
            r"validate\w*",
            r"check\w*",
            r"verify\w*",
            r"is_valid",
            r"ValidationError",
            r"raise.*Error"
        ]
        
        type_patterns = {
            "email": [r"email.*@", r"re\.\w*\(.*@.*\)"],
            "password": [r"len\(.*password.*\)", r"password.*\d+"],
            "input": [r"if\s+not\s+", r"required.*True"],
            "form": [r"form\.validate", r"forms\."],
            "request": [r"request\.(json|data|form)", r"400.*Bad Request"],
            "data": [r"isinstance\(", r"type\(.*\)"]
        }
        
        patterns = base_patterns.copy()
        if validation_type in type_patterns:
            patterns.extend(type_patterns[validation_type])
            
        return patterns


class TaskSpecificationValidator:
    """
    Validates task specifications to ensure they are well-defined and solvable.
    Implements ABC checklist items T.7-T.10
    """
    
    def __init__(self):
        self.oracle_solvers: List[OracleSolver] = [
            APIEndpointOracleSolver(),
            ValidationOracleSolver(),
            # Add more oracle solvers as needed
        ]
        
    async def validate_task(self, test_case: 'TestCase') -> TaskValidationResult:
        """Comprehensively validate a task specification"""
        logger.info(f"Validating task specification: {test_case.id}")
        
        ambiguity_issues = []
        missing_specifications = []
        
        # Check 1: Task description clarity
        clarity_score = self._check_description_clarity(test_case, ambiguity_issues)
        
        # Check 2: Specific requirements
        specificity_score = self._check_requirement_specificity(
            test_case, 
            missing_specifications
        )
        
        # Check 3: Expected outcomes are measurable
        measurability_score = self._check_outcome_measurability(
            test_case,
            ambiguity_issues
        )
        
        # Check 4: Oracle solver verification
        is_solvable, oracle_solution = await self._verify_with_oracle_solvers(test_case)
        
        # Check 5: Difficulty estimation
        estimated_difficulty = self._estimate_difficulty(test_case, oracle_solution)
        
        # Calculate overall validity
        is_valid = (
            clarity_score > 0.7 and
            specificity_score > 0.7 and
            measurability_score > 0.7 and
            is_solvable
        )
        
        return TaskValidationResult(
            is_valid=is_valid,
            is_solvable=is_solvable,
            specificity_score=(clarity_score + specificity_score + measurability_score) / 3,
            ambiguity_issues=ambiguity_issues,
            missing_specifications=missing_specifications,
            oracle_solution=oracle_solution.__dict__ if oracle_solution else None,
            estimated_difficulty=estimated_difficulty
        )
    
    def _check_description_clarity(
        self, 
        test_case: 'TestCase',
        ambiguity_issues: List[str]
    ) -> float:
        """Check if task description is clear and unambiguous"""
        clarity_score = 1.0
        
        # Check for vague terms
        vague_terms = [
            'some', 'various', 'appropriate', 'proper', 'good', 
            'better', 'optimize', 'improve', 'enhance'
        ]
        
        prompt_lower = test_case.prompt.lower()
        for term in vague_terms:
            if term in prompt_lower:
                ambiguity_issues.append(f"Vague term '{term}' makes requirements unclear")
                clarity_score -= 0.1
        
        # Check for specific actions
        action_verbs = [
            'add', 'create', 'implement', 'fix', 'remove', 
            'update', 'validate', 'test', 'document'
        ]
        
        has_clear_action = any(verb in prompt_lower for verb in action_verbs)
        if not has_clear_action:
            ambiguity_issues.append("No clear action verb in task description")
            clarity_score -= 0.3
        
        # Check prompt length (too short or too long can be problematic)
        word_count = len(test_case.prompt.split())
        if word_count < 5:
            ambiguity_issues.append("Task description too brief")
            clarity_score -= 0.2
        elif word_count > 100:
            ambiguity_issues.append("Task description too verbose, may contain multiple tasks")
            clarity_score -= 0.1
        
        return max(0, clarity_score)
    
    def _check_requirement_specificity(
        self,
        test_case: 'TestCase',
        missing_specifications: List[str]
    ) -> float:
        """Check if requirements are specific enough"""
        specificity_score = 1.0
        
        prompt_lower = test_case.prompt.lower()
        
        # Check for quantifiable requirements
        if 'all' in prompt_lower:
            # "All" without specifics is problematic
            if not re.search(r'all\s+\w+\s+(endpoints?|files?|functions?|methods?)', prompt_lower):
                missing_specifications.append("'All' used without specifying what items")
                specificity_score -= 0.2
        
        # Check for file/location specifications
        has_file_spec = (
            '.' in test_case.prompt or  # file extension
            '/' in test_case.prompt or  # path
            any(term in prompt_lower for term in ['file', 'module', 'class', 'function'])
        )
        
        if not has_file_spec and 'endpoint' not in prompt_lower:
            missing_specifications.append("No specific files or locations mentioned")
            specificity_score -= 0.2
        
        # Check for acceptance criteria
        if not test_case.expected_outcome or len(test_case.expected_outcome) < 2:
            missing_specifications.append("Insufficient acceptance criteria")
            specificity_score -= 0.3
        
        # Check for technical specifications
        technical_terms = [
            'json', 'xml', 'api', 'database', 'validation', 
            'authentication', 'error', 'response', 'request'
        ]
        
        has_technical_spec = any(term in prompt_lower for term in technical_terms)
        if not has_technical_spec:
            missing_specifications.append("No technical specifications provided")
            specificity_score -= 0.1
        
        return max(0, specificity_score)
    
    def _check_outcome_measurability(
        self,
        test_case: 'TestCase',
        ambiguity_issues: List[str]
    ) -> float:
        """Check if expected outcomes are measurable"""
        measurability_score = 1.0
        
        if not test_case.expected_outcome:
            ambiguity_issues.append("No expected outcomes defined")
            return 0.0
        
        # Check each expected outcome
        for key, value in test_case.expected_outcome.items():
            # Boolean outcomes are good
            if isinstance(value, bool):
                continue
                
            # Lists should have specific items
            elif isinstance(value, list):
                if not value:
                    ambiguity_issues.append(f"Empty list for outcome '{key}'")
                    measurability_score -= 0.2
                elif any(isinstance(item, str) and '*' in item for item in value):
                    # Wildcards reduce measurability
                    measurability_score -= 0.1
                    
            # Strings should be specific
            elif isinstance(value, str):
                if value in ['any', 'some', 'various']:
                    ambiguity_issues.append(f"Vague value '{value}' for outcome '{key}'")
                    measurability_score -= 0.2
                    
            # Numbers are good
            elif isinstance(value, (int, float)):
                continue
                
            else:
                ambiguity_issues.append(f"Unclear outcome type for '{key}'")
                measurability_score -= 0.1
        
        return max(0, measurability_score)
    
    async def _verify_with_oracle_solvers(
        self, 
        test_case: 'TestCase'
    ) -> Tuple[bool, Optional[OracleSolution]]:
        """Verify task is solvable using oracle solvers"""
        for solver in self.oracle_solvers:
            try:
                can_solve, solution = await solver.can_solve(test_case)
                if can_solve:
                    logger.info(f"Task {test_case.id} verified solvable by {solver.get_solver_type()}")
                    return True, solution
            except Exception as e:
                logger.warning(f"Oracle solver {solver.get_solver_type()} failed: {e}")
                continue
        
        logger.warning(f"No oracle solver could verify task {test_case.id}")
        return False, None
    
    def _estimate_difficulty(
        self, 
        test_case: 'TestCase',
        oracle_solution: Optional[OracleSolution]
    ) -> str:
        """Estimate task difficulty"""
        # Start with declared difficulty
        base_difficulty = test_case.difficulty
        
        # Adjust based on analysis
        complexity_score = 0
        
        # Word count factor
        word_count = len(test_case.prompt.split())
        if word_count > 50:
            complexity_score += 1
        if word_count > 100:
            complexity_score += 1
            
        # Technical complexity
        technical_terms = [
            'distributed', 'async', 'concurrent', 'cache', 'optimize',
            'security', 'authentication', 'authorization', 'encryption',
            'database', 'migration', 'scaling', 'performance'
        ]
        
        prompt_lower = test_case.prompt.lower()
        technical_count = sum(1 for term in technical_terms if term in prompt_lower)
        complexity_score += min(technical_count, 3)
        
        # Multiple requirements
        if test_case.expected_outcome:
            complexity_score += len(test_case.expected_outcome) // 3
        
        # Oracle solution complexity
        if oracle_solution:
            complexity_score += len(oracle_solution.key_implementations) // 4
        
        # Map to difficulty
        if complexity_score <= 1:
            return "easy"
        elif complexity_score <= 3:
            return "medium"
        elif complexity_score <= 5:
            return "hard"
        else:
            return "expert"


class TaskSpecificationReporter:
    """Generate reports on task specification quality"""
    
    @staticmethod
    def generate_report(
        test_cases: List['TestCase'],
        validation_results: List[TaskValidationResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive task specification report"""
        
        total_tasks = len(test_cases)
        valid_tasks = sum(1 for r in validation_results if r.is_valid)
        solvable_tasks = sum(1 for r in validation_results if r.is_solvable)
        
        # Aggregate issues
        all_ambiguity_issues = []
        all_missing_specs = []
        
        for result in validation_results:
            all_ambiguity_issues.extend(result.ambiguity_issues)
            all_missing_specs.extend(result.missing_specifications)
        
        # Count unique issues
        unique_ambiguity = len(set(all_ambiguity_issues))
        unique_missing = len(set(all_missing_specs))
        
        # Difficulty distribution
        difficulty_dist = {
            "easy": 0,
            "medium": 0,
            "hard": 0,
            "expert": 0
        }
        
        for result in validation_results:
            difficulty_dist[result.estimated_difficulty] += 1
        
        report = {
            "summary": {
                "total_tasks": total_tasks,
                "valid_tasks": valid_tasks,
                "solvable_tasks": solvable_tasks,
                "validity_rate": valid_tasks / total_tasks if total_tasks > 0 else 0,
                "solvability_rate": solvable_tasks / total_tasks if total_tasks > 0 else 0,
                "average_specificity": sum(r.specificity_score for r in validation_results) / total_tasks if total_tasks > 0 else 0
            },
            "issues": {
                "unique_ambiguity_issues": unique_ambiguity,
                "unique_missing_specifications": unique_missing,
                "most_common_ambiguity": max(set(all_ambiguity_issues), key=all_ambiguity_issues.count) if all_ambiguity_issues else None,
                "most_common_missing": max(set(all_missing_specs), key=all_missing_specs.count) if all_missing_specs else None
            },
            "difficulty_distribution": difficulty_dist,
            "recommendations": []
        }
        
        # Generate recommendations
        if report["summary"]["validity_rate"] < 0.8:
            report["recommendations"].append("Improve task specifications - many tasks have ambiguous requirements")
            
        if report["summary"]["solvability_rate"] < 0.9:
            report["recommendations"].append("Review unsolvable tasks - ensure all tasks can be completed")
            
        if report["summary"]["average_specificity"] < 0.7:
            report["recommendations"].append("Increase requirement specificity - avoid vague terms and provide concrete criteria")
        
        return report