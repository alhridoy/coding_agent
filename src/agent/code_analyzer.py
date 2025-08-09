"""
Advanced code analysis engine using AST parsing and static analysis
"""

import ast
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CodeFile:
    """Represents a code file with metadata"""
    path: str
    language: str
    lines: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity: int
    dependencies: List[str]
    test_coverage: float = 0.0


@dataclass
class CodePattern:
    """Represents a code pattern or issue"""
    type: str
    severity: str
    description: str
    file_path: str
    line_number: int
    suggestion: str
    example_fix: str = ""


@dataclass
class RepositoryAnalysis:
    """Complete repository analysis"""
    total_files: int
    languages: Dict[str, int]
    architecture_patterns: List[str]
    code_quality_score: float
    test_coverage: float
    security_issues: List[CodePattern]
    performance_issues: List[CodePattern]
    code_smells: List[CodePattern]
    suggested_improvements: List[str]
    dependency_graph: Dict[str, List[str]]


class CodeAnalyzer:
    """Advanced code analysis engine"""
    
    def __init__(self):
        self.language_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
        }
        
        self.ignore_patterns = [
            r'__pycache__',
            r'\.git',
            r'node_modules',
            r'\.env',
            r'\.venv',
            r'build',
            r'dist',
            r'\.pyc$',
            r'\.log$',
        ]
    
    async def analyze_repository(self, session_id: str, sandbox_manager) -> RepositoryAnalysis:
        """Perform comprehensive repository analysis"""
        logger.info(f"Starting repository analysis for session {session_id}")
        
        # Get file structure
        file_structure = await self._get_file_structure(session_id, sandbox_manager)
        
        # Analyze each file
        file_analyses = []
        for file_path in file_structure:
            if self._should_analyze_file(file_path):
                analysis = await self._analyze_file(session_id, file_path, sandbox_manager)
                if analysis:
                    file_analyses.append(analysis)
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(file_analyses)
        
        # Detect architecture patterns
        architecture_patterns = self._detect_architecture_patterns(file_analyses, dependency_graph)
        
        # Calculate metrics
        languages = self._calculate_language_distribution(file_analyses)
        code_quality_score = self._calculate_code_quality_score(file_analyses)
        
        # Find issues and improvements
        security_issues = self._find_security_issues(file_analyses)
        performance_issues = self._find_performance_issues(file_analyses)
        code_smells = self._find_code_smells(file_analyses)
        
        # Generate improvement suggestions
        suggested_improvements = self._generate_improvement_suggestions(
            file_analyses, security_issues, performance_issues, code_smells
        )
        
        return RepositoryAnalysis(
            total_files=len(file_analyses),
            languages=languages,
            architecture_patterns=architecture_patterns,
            code_quality_score=code_quality_score,
            test_coverage=self._calculate_test_coverage(file_analyses),
            security_issues=security_issues,
            performance_issues=performance_issues,
            code_smells=code_smells,
            suggested_improvements=suggested_improvements,
            dependency_graph=dependency_graph
        )
    
    async def _get_file_structure(self, session_id: str, sandbox_manager) -> List[str]:
        """Get all files in the repository"""
        result = await sandbox_manager.execute_command(
            session_id, 
            "find . -type f -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.jsx' -o -name '*.tsx' -o -name '*.java' -o -name '*.go' -o -name '*.rs' -o -name '*.cpp' -o -name '*.c' -o -name '*.cs' -o -name '*.rb' -o -name '*.php' | grep -v __pycache__ | grep -v node_modules | head -50"
        )
        
        if result["success"]:
            return [f.strip() for f in result["stdout"].split('\n') if f.strip()]
        return []
    
    def _should_analyze_file(self, file_path: str) -> bool:
        """Check if file should be analyzed"""
        for pattern in self.ignore_patterns:
            if re.search(pattern, file_path):
                return False
        
        extension = Path(file_path).suffix.lower()
        return extension in self.language_extensions
    
    async def _analyze_file(self, session_id: str, file_path: str, sandbox_manager) -> Optional[CodeFile]:
        """Analyze a single file"""
        try:
            # Read file content
            result = await sandbox_manager.read_file(session_id, file_path)
            if not result["success"]:
                return None
            
            content = result["content"]
            extension = Path(file_path).suffix.lower()
            language = self.language_extensions.get(extension, 'unknown')
            
            # Basic metrics
            lines = len(content.split('\n'))
            
            # Language-specific analysis
            if language == 'python':
                return self._analyze_python_file(file_path, content, lines)
            elif language in ['javascript', 'typescript']:
                return self._analyze_javascript_file(file_path, content, lines, language)
            else:
                return self._analyze_generic_file(file_path, content, lines, language)
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return None
    
    def _analyze_python_file(self, file_path: str, content: str, lines: int) -> CodeFile:
        """Analyze Python file using AST"""
        try:
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Calculate cyclomatic complexity (simplified)
            complexity = self._calculate_python_complexity(tree)
            
            # Extract dependencies
            dependencies = self._extract_python_dependencies(imports)
            
            return CodeFile(
                path=file_path,
                language='python',
                lines=lines,
                functions=functions,
                classes=classes,
                imports=imports,
                complexity=complexity,
                dependencies=dependencies
            )
            
        except SyntaxError:
            logger.warning(f"Syntax error in Python file: {file_path}")
            return self._analyze_generic_file(file_path, content, lines, 'python')
    
    def _analyze_javascript_file(self, file_path: str, content: str, lines: int, language: str) -> CodeFile:
        """Analyze JavaScript/TypeScript file"""
        # Simple regex-based analysis (could be enhanced with proper JS parser)
        
        # Find functions
        function_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|\([^)]*\)\s*{)|(\w+)\s*:\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))'
        functions = []
        for match in re.finditer(function_pattern, content):
            name = match.group(1) or match.group(2) or match.group(3)
            if name:
                functions.append(name)
        
        # Find classes
        class_pattern = r'class\s+(\w+)'
        classes = [match.group(1) for match in re.finditer(class_pattern, content)]
        
        # Find imports
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\)'
        imports = []
        for match in re.finditer(import_pattern, content):
            imp = match.group(1) or match.group(2)
            if imp:
                imports.append(imp)
        
        # Calculate complexity (simplified)
        complexity = self._calculate_js_complexity(content)
        
        # Extract dependencies
        dependencies = self._extract_js_dependencies(imports)
        
        return CodeFile(
            path=file_path,
            language=language,
            lines=lines,
            functions=functions,
            classes=classes,
            imports=imports,
            complexity=complexity,
            dependencies=dependencies
        )
    
    def _analyze_generic_file(self, file_path: str, content: str, lines: int, language: str) -> CodeFile:
        """Generic file analysis"""
        return CodeFile(
            path=file_path,
            language=language,
            lines=lines,
            functions=[],
            classes=[],
            imports=[],
            complexity=1,
            dependencies=[]
        )
    
    def _calculate_python_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity for Python code"""
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_js_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity for JavaScript code"""
        complexity = 1
        
        # Count decision points
        patterns = [
            r'\bif\s*\(',
            r'\bwhile\s*\(',
            r'\bfor\s*\(',
            r'\bswitch\s*\(',
            r'\bcatch\s*\(',
            r'\?\s*.*?\s*:',  # ternary operator
            r'\&\&',
            r'\|\|'
        ]
        
        for pattern in patterns:
            complexity += len(re.findall(pattern, content))
        
        return complexity
    
    def _extract_python_dependencies(self, imports: List[str]) -> List[str]:
        """Extract meaningful dependencies from Python imports"""
        dependencies = []
        
        for imp in imports:
            # Skip standard library imports
            if imp not in ['os', 'sys', 'json', 'time', 'datetime', 're', 'math', 'random']:
                # Extract package name
                package = imp.split('.')[0]
                if package not in dependencies:
                    dependencies.append(package)
        
        return dependencies
    
    def _extract_js_dependencies(self, imports: List[str]) -> List[str]:
        """Extract meaningful dependencies from JavaScript imports"""
        dependencies = []
        
        for imp in imports:
            # Skip relative imports
            if not imp.startswith('.'):
                # Extract package name
                package = imp.split('/')[0]
                if package.startswith('@'):
                    # Scoped package
                    parts = imp.split('/')
                    if len(parts) > 1:
                        package = f"{parts[0]}/{parts[1]}"
                
                if package not in dependencies:
                    dependencies.append(package)
        
        return dependencies
    
    def _build_dependency_graph(self, files: List[CodeFile]) -> Dict[str, List[str]]:
        """Build dependency graph between files"""
        graph = defaultdict(list)
        
        for file in files:
            for dep in file.dependencies:
                if dep not in graph[file.path]:
                    graph[file.path].append(dep)
        
        return dict(graph)
    
    def _detect_architecture_patterns(self, files: List[CodeFile], dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Detect architecture patterns in the codebase"""
        patterns = []
        
        # Check for common patterns
        file_paths = [f.path for f in files]
        
        # MVC pattern
        if any('controller' in path.lower() for path in file_paths) and \
           any('model' in path.lower() for path in file_paths) and \
           any('view' in path.lower() for path in file_paths):
            patterns.append('MVC')
        
        # Microservices pattern
        if any('service' in path.lower() for path in file_paths) and \
           any('api' in path.lower() for path in file_paths):
            patterns.append('Microservices')
        
        # Repository pattern
        if any('repository' in path.lower() for path in file_paths):
            patterns.append('Repository')
        
        # Factory pattern
        if any('factory' in path.lower() for path in file_paths):
            patterns.append('Factory')
        
        # REST API pattern
        if any('router' in path.lower() or 'route' in path.lower() for path in file_paths):
            patterns.append('REST API')
        
        return patterns
    
    def _calculate_language_distribution(self, files: List[CodeFile]) -> Dict[str, int]:
        """Calculate language distribution"""
        distribution = defaultdict(int)
        
        for file in files:
            distribution[file.language] += 1
        
        return dict(distribution)
    
    def _calculate_code_quality_score(self, files: List[CodeFile]) -> float:
        """Calculate overall code quality score"""
        if not files:
            return 0.0
        
        total_score = 0.0
        
        for file in files:
            score = 100.0
            
            # Penalize high complexity
            if file.complexity > 10:
                score -= min(20, file.complexity - 10)
            
            # Penalize very long files
            if file.lines > 500:
                score -= min(15, (file.lines - 500) / 100)
            
            # Reward documentation (functions/classes ratio)
            if file.functions or file.classes:
                score += 5
            
            total_score += max(0, score)
        
        return total_score / len(files)
    
    def _calculate_test_coverage(self, files: List[CodeFile]) -> float:
        """Calculate test coverage based on test files"""
        total_files = len(files)
        test_files = len([f for f in files if 'test' in f.path.lower() or 'spec' in f.path.lower()])
        
        if total_files == 0:
            return 0.0
        
        return (test_files / total_files) * 100
    
    def _find_security_issues(self, files: List[CodeFile]) -> List[CodePattern]:
        """Find potential security issues"""
        issues = []
        
        for file in files:
            if file.language == 'python':
                # Check for common Python security issues
                if 'eval' in str(file.functions):
                    issues.append(CodePattern(
                        type='security',
                        severity='high',
                        description='Use of eval() function detected',
                        file_path=file.path,
                        line_number=0,
                        suggestion='Avoid using eval() as it can execute arbitrary code'
                    ))
                
                if 'pickle' in file.imports:
                    issues.append(CodePattern(
                        type='security',
                        severity='medium',
                        description='Use of pickle module detected',
                        file_path=file.path,
                        line_number=0,
                        suggestion='Be cautious with pickle, consider using json for data serialization'
                    ))
        
        return issues
    
    def _find_performance_issues(self, files: List[CodeFile]) -> List[CodePattern]:
        """Find potential performance issues"""
        issues = []
        
        for file in files:
            if file.complexity > 20:
                issues.append(CodePattern(
                    type='performance',
                    severity='medium',
                    description=f'High cyclomatic complexity ({file.complexity})',
                    file_path=file.path,
                    line_number=0,
                    suggestion='Consider breaking down complex functions into smaller ones'
                ))
            
            if file.lines > 1000:
                issues.append(CodePattern(
                    type='performance',
                    severity='low',
                    description=f'Large file ({file.lines} lines)',
                    file_path=file.path,
                    line_number=0,
                    suggestion='Consider splitting large files into smaller modules'
                ))
        
        return issues
    
    def _find_code_smells(self, files: List[CodeFile]) -> List[CodePattern]:
        """Find code smells"""
        smells = []
        
        for file in files:
            # Long parameter lists (simplified check)
            if len(file.functions) > 20:
                smells.append(CodePattern(
                    type='code_smell',
                    severity='low',
                    description='Many functions in single file',
                    file_path=file.path,
                    line_number=0,
                    suggestion='Consider organizing functions into classes or modules'
                ))
            
            # Empty or minimal files
            if file.lines < 10 and file.language != 'unknown':
                smells.append(CodePattern(
                    type='code_smell',
                    severity='low',
                    description='Very small file',
                    file_path=file.path,
                    line_number=0,
                    suggestion='Consider if this file is necessary or should be merged'
                ))
        
        return smells
    
    def _generate_improvement_suggestions(self, files: List[CodeFile], security_issues: List[CodePattern], 
                                        performance_issues: List[CodePattern], code_smells: List[CodePattern]) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        # Test coverage
        test_coverage = self._calculate_test_coverage(files)
        if test_coverage < 50:
            suggestions.append(f"Improve test coverage (currently {test_coverage:.1f}%)")
        
        # Documentation
        python_files = [f for f in files if f.language == 'python']
        if python_files:
            documented = len([f for f in python_files if f.functions or f.classes])
            if documented / len(python_files) < 0.7:
                suggestions.append("Add more documentation and docstrings")
        
        # Security
        if security_issues:
            suggestions.append(f"Address {len(security_issues)} security issues")
        
        # Performance
        if performance_issues:
            suggestions.append(f"Address {len(performance_issues)} performance issues")
        
        # Code quality
        avg_complexity = sum(f.complexity for f in files) / len(files) if files else 0
        if avg_complexity > 10:
            suggestions.append("Reduce code complexity by refactoring complex functions")
        
        # Dependencies
        all_deps = set()
        for file in files:
            all_deps.update(file.dependencies)
        
        if len(all_deps) > 50:
            suggestions.append("Consider reducing number of dependencies")
        
        return suggestions
