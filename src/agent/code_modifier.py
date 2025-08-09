"""
Advanced code modification engine that can make intelligent changes to code
"""

import ast
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class CodeChange:
    """Represents a single code change"""
    file_path: str
    change_type: str  # 'add', 'modify', 'delete', 'create'
    old_content: str
    new_content: str
    line_start: int
    line_end: int
    description: str
    reasoning: str


@dataclass
class ModificationPlan:
    """Complete plan for modifying code"""
    changes: List[CodeChange]
    test_commands: List[str]
    validation_steps: List[str]
    rollback_plan: List[str]
    estimated_time: int  # minutes
    risk_level: str  # 'low', 'medium', 'high'


class CodeModificationStrategy(ABC):
    """Abstract base class for code modification strategies"""
    
    @abstractmethod
    async def can_handle(self, prompt: str, repo_analysis: Any) -> bool:
        """Check if this strategy can handle the given prompt"""
        pass
    
    @abstractmethod
    async def create_plan(self, prompt: str, repo_analysis: Any, session_id: str, sandbox_manager) -> ModificationPlan:
        """Create a modification plan"""
        pass


class AddEndpointStrategy(CodeModificationStrategy):
    """Strategy for adding new API endpoints"""
    
    async def can_handle(self, prompt: str, repo_analysis: Any) -> bool:
        """Check if this is an endpoint addition request"""
        endpoint_keywords = ['endpoint', 'route', 'api', 'handler', 'controller']
        action_keywords = ['add', 'create', 'implement', 'new']
        
        prompt_lower = prompt.lower()
        
        has_endpoint = any(keyword in prompt_lower for keyword in endpoint_keywords)
        has_action = any(keyword in prompt_lower for keyword in action_keywords)
        
        return has_endpoint and has_action
    
    async def create_plan(self, prompt: str, repo_analysis: Any, session_id: str, sandbox_manager) -> ModificationPlan:
        """Create plan for adding an endpoint"""
        changes = []
        
        # Find main application file
        app_file = await self._find_app_file(repo_analysis, session_id, sandbox_manager)
        
        if app_file:
            # Analyze existing endpoints
            existing_endpoints = await self._analyze_endpoints(app_file, session_id, sandbox_manager)
            
            # Generate new endpoint code
            endpoint_info = self._extract_endpoint_info(prompt)
            new_endpoint_code = self._generate_endpoint_code(endpoint_info, existing_endpoints)
            
            # Create change for adding endpoint
            changes.append(CodeChange(
                file_path=app_file,
                change_type='add',
                old_content='',
                new_content=new_endpoint_code,
                line_start=-1,  # Will be determined during insertion
                line_end=-1,
                description=f"Add {endpoint_info['method']} {endpoint_info['path']} endpoint",
                reasoning=f"Adding new endpoint as requested: {prompt}"
            ))
            
            # Add model if needed
            if endpoint_info.get('needs_model'):
                model_code = self._generate_model_code(endpoint_info)
                changes.append(CodeChange(
                    file_path='models.py',
                    change_type='create',
                    old_content='',
                    new_content=model_code,
                    line_start=1,
                    line_end=1,
                    description=f"Add data model for {endpoint_info['name']}",
                    reasoning="Model needed for request/response validation"
                ))
        
        return ModificationPlan(
            changes=changes,
            test_commands=['python -m pytest tests/'],
            validation_steps=['Check endpoint responds correctly', 'Validate request/response format'],
            rollback_plan=['git checkout HEAD~1'],
            estimated_time=15,
            risk_level='low'
        )
    
    async def _find_app_file(self, repo_analysis: Any, session_id: str, sandbox_manager) -> Optional[str]:
        """Find the main application file"""
        common_names = ['app.py', 'main.py', 'server.py', 'api.py', 'application.py']
        
        for name in common_names:
            result = await sandbox_manager.read_file(session_id, name)
            if result["success"]:
                content = result["content"]
                # Check if it's likely an app file (has Flask/FastAPI imports)
                if any(framework in content for framework in ['FastAPI', 'Flask', 'app =', 'application =']):
                    return name
        
        return None
    
    async def _analyze_endpoints(self, app_file: str, session_id: str, sandbox_manager) -> List[Dict[str, Any]]:
        """Analyze existing endpoints in the app file"""
        result = await sandbox_manager.read_file(session_id, app_file)
        if not result["success"]:
            return []
        
        content = result["content"]
        endpoints = []
        
        # FastAPI endpoints
        fastapi_pattern = r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']'
        for match in re.finditer(fastapi_pattern, content, re.IGNORECASE):
            endpoints.append({
                'method': match.group(1).upper(),
                'path': match.group(2),
                'framework': 'fastapi'
            })
        
        # Flask endpoints
        flask_pattern = r'@app\.route\(["\']([^"\']+)["\'].*?methods=\[["\']([^"\']+)["\']'
        for match in re.finditer(flask_pattern, content, re.IGNORECASE):
            endpoints.append({
                'method': match.group(2).upper(),
                'path': match.group(1),
                'framework': 'flask'
            })
        
        return endpoints
    
    def _extract_endpoint_info(self, prompt: str) -> Dict[str, Any]:
        """Extract endpoint information from the prompt"""
        info = {
            'name': 'new_endpoint',
            'method': 'GET',
            'path': '/new',
            'description': prompt,
            'needs_model': False,
            'returns_data': True
        }
        
        # Extract HTTP method
        methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        for method in methods:
            if method.lower() in prompt.lower():
                info['method'] = method
                break
        
        # Extract endpoint name/purpose
        if 'health' in prompt.lower():
            info['name'] = 'health_check'
            info['path'] = '/health'
            info['description'] = 'Health check endpoint'
        elif 'status' in prompt.lower():
            info['name'] = 'status'
            info['path'] = '/status'
            info['description'] = 'Status endpoint'
        elif 'user' in prompt.lower():
            info['name'] = 'user'
            info['path'] = '/users'
            info['needs_model'] = True
        elif 'auth' in prompt.lower():
            info['name'] = 'auth'
            info['path'] = '/auth'
            info['needs_model'] = True
        
        # Determine if it needs input validation
        if info['method'] in ['POST', 'PUT', 'PATCH']:
            info['needs_model'] = True
        
        return info
    
    def _generate_endpoint_code(self, endpoint_info: Dict[str, Any], existing_endpoints: List[Dict[str, Any]]) -> str:
        """Generate code for the new endpoint"""
        # Determine framework based on existing endpoints
        framework = 'fastapi'  # default
        if existing_endpoints:
            framework = existing_endpoints[0].get('framework', 'fastapi')
        
        if framework == 'fastapi':
            return self._generate_fastapi_endpoint(endpoint_info)
        else:
            return self._generate_flask_endpoint(endpoint_info)
    
    def _generate_fastapi_endpoint(self, info: Dict[str, Any]) -> str:
        """Generate FastAPI endpoint code"""
        method = info['method'].lower()
        path = info['path']
        name = info['name']
        description = info['description']
        
        if info['needs_model']:
            model_name = f"{name.title()}Request"
            code = f'''
@app.{method}("{path}")
async def {name}(request: {model_name}):
    """
    {description}
    """
    try:
        # TODO: Implement {name} logic here
        return {{"message": "Success", "data": request.dict()}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
        else:
            code = f'''
@app.{method}("{path}")
async def {name}():
    """
    {description}
    """
    try:
        # TODO: Implement {name} logic here
        return {{"status": "healthy", "timestamp": datetime.now().isoformat()}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
        
        return code.strip()
    
    def _generate_flask_endpoint(self, info: Dict[str, Any]) -> str:
        """Generate Flask endpoint code"""
        method = info['method']
        path = info['path']
        name = info['name']
        description = info['description']
        
        code = f'''
@app.route("{path}", methods=["{method}"])
def {name}():
    """
    {description}
    """
    try:
        # TODO: Implement {name} logic here
        return jsonify({{"status": "healthy", "timestamp": datetime.now().isoformat()}})
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500
'''
        
        return code.strip()
    
    def _generate_model_code(self, info: Dict[str, Any]) -> str:
        """Generate Pydantic model code"""
        name = info['name']
        model_name = f"{name.title()}Request"
        
        code = f'''
from pydantic import BaseModel
from typing import Optional

class {model_name}(BaseModel):
    """Request model for {name} endpoint"""
    # TODO: Add appropriate fields for your use case
    name: str
    description: Optional[str] = None
    
    class Config:
        json_encoders = {{
            # Add custom encoders if needed
        }}
'''
        
        return code.strip()


class AddValidationStrategy(CodeModificationStrategy):
    """Strategy for adding input validation"""
    
    async def can_handle(self, prompt: str, repo_analysis: Any) -> bool:
        """Check if this is a validation request"""
        validation_keywords = ['validation', 'validate', 'sanitize', 'check input', 'input validation']
        prompt_lower = prompt.lower()
        
        return any(keyword in prompt_lower for keyword in validation_keywords)
    
    async def create_plan(self, prompt: str, repo_analysis: Any, session_id: str, sandbox_manager) -> ModificationPlan:
        """Create plan for adding validation"""
        changes = []
        
        # Find endpoints that need validation
        app_file = await self._find_app_file(session_id, sandbox_manager)
        
        if app_file:
            endpoints = await self._find_endpoints_needing_validation(app_file, session_id, sandbox_manager)
            
            for endpoint in endpoints:
                validation_code = self._generate_validation_code(endpoint)
                changes.append(CodeChange(
                    file_path=app_file,
                    change_type='modify',
                    old_content=endpoint['original_code'],
                    new_content=validation_code,
                    line_start=endpoint['line_start'],
                    line_end=endpoint['line_end'],
                    description=f"Add validation to {endpoint['method']} {endpoint['path']}",
                    reasoning="Adding input validation for security and data integrity"
                ))
        
        return ModificationPlan(
            changes=changes,
            test_commands=['python -m pytest tests/test_validation.py'],
            validation_steps=['Test with invalid input', 'Verify error messages'],
            rollback_plan=['git checkout HEAD~1'],
            estimated_time=20,
            risk_level='medium'
        )
    
    async def _find_app_file(self, session_id: str, sandbox_manager) -> Optional[str]:
        """Find the main app file"""
        common_names = ['app.py', 'main.py', 'server.py']
        
        for name in common_names:
            result = await sandbox_manager.read_file(session_id, name)
            if result["success"]:
                return name
        
        return None
    
    async def _find_endpoints_needing_validation(self, app_file: str, session_id: str, sandbox_manager) -> List[Dict[str, Any]]:
        """Find endpoints that need validation"""
        result = await sandbox_manager.read_file(session_id, app_file)
        if not result["success"]:
            return []
        
        content = result["content"]
        lines = content.split('\n')
        endpoints = []
        
        # Look for POST/PUT endpoints without proper validation
        for i, line in enumerate(lines):
            if '@app.post(' in line or '@app.put(' in line:
                # Extract endpoint info
                endpoint_match = re.search(r'@app\.(post|put)\(["\']([^"\']+)["\']', line)
                if endpoint_match:
                    method = endpoint_match.group(1).upper()
                    path = endpoint_match.group(2)
                    
                    # Find the function
                    func_start = i + 1
                    while func_start < len(lines) and not lines[func_start].strip().startswith('def '):
                        func_start += 1
                    
                    if func_start < len(lines):
                        # Find function end
                        func_end = func_start + 1
                        indent_level = len(lines[func_start]) - len(lines[func_start].lstrip())
                        
                        while func_end < len(lines) and (
                            not lines[func_end].strip() or 
                            len(lines[func_end]) - len(lines[func_end].lstrip()) > indent_level
                        ):
                            func_end += 1
                        
                        func_code = '\n'.join(lines[i:func_end])
                        
                        # Check if validation is already present
                        if 'validate' not in func_code.lower() and 'BaseModel' not in func_code:
                            endpoints.append({
                                'method': method,
                                'path': path,
                                'line_start': i,
                                'line_end': func_end,
                                'original_code': func_code
                            })
        
        return endpoints
    
    def _generate_validation_code(self, endpoint: Dict[str, Any]) -> str:
        """Generate validation code for an endpoint"""
        original = endpoint['original_code']
        method = endpoint['method']
        path = endpoint['path']
        
        # Create a simple validation model
        model_name = f"{path.replace('/', '').title()}Request"
        
        validation_code = f'''
from pydantic import BaseModel, validator

class {model_name}(BaseModel):
    # Add validation fields based on your requirements
    data: str
    
    @validator('data')
    def validate_data(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Data cannot be empty')
        if len(v) > 1000:
            raise ValueError('Data too long')
        return v.strip()

{original.replace('def ', f'def ').replace(':', f'(request: {model_name}):')}'''
        
        return validation_code


class CodeModifier:
    """Main code modification engine"""
    
    def __init__(self):
        # Import here to avoid circular imports
        from .claude_strategy import ClaudeModificationStrategy
        
        self.strategies = [
            ClaudeModificationStrategy(),  # Use Claude for ALL code generation - no hardcoded strategies
        ]
        # self.analyzer = CodeAnalyzer()  # Import dynamically to avoid circular imports
    
    async def create_modification_plan(self, prompt: str, repo_analysis: Any, 
                                     session_id: str, sandbox_manager) -> ModificationPlan:
        """Create a comprehensive modification plan"""
        
        # Find the best strategy for this prompt
        for strategy in self.strategies:
            if await strategy.can_handle(prompt, repo_analysis):
                logger.info(f"Using strategy: {strategy.__class__.__name__}")
                return await strategy.create_plan(prompt, repo_analysis, session_id, sandbox_manager)
        
        # Fallback: generic modification plan
        return await self._create_generic_plan(prompt, repo_analysis, session_id, sandbox_manager)
    
    async def _create_generic_plan(self, prompt: str, repo_analysis: Any, 
                                 session_id: str, sandbox_manager) -> ModificationPlan:
        """Create a generic modification plan"""
        changes = []
        
        # Add a comment to README explaining what was attempted
        readme_change = CodeChange(
            file_path='README.md',
            change_type='modify',
            old_content='',
            new_content=f'\n\n## Recent Changes\n\nAttempted implementation: {prompt}\n',
            line_start=-1,
            line_end=-1,
            description='Document the attempted change',
            reasoning='Fallback modification to document the request'
        )
        changes.append(readme_change)
        
        return ModificationPlan(
            changes=changes,
            test_commands=[],
            validation_steps=['Verify documentation was added'],
            rollback_plan=['git checkout HEAD~1'],
            estimated_time=5,
            risk_level='low'
        )
    
    async def apply_modification_plan(self, plan: ModificationPlan, session_id: str, 
                                    sandbox_manager) -> List[Dict[str, Any]]:
        """Apply the modification plan"""
        results = []
        
        for change in plan.changes:
            try:
                result = await self._apply_single_change(change, session_id, sandbox_manager)
                results.append({
                    'change': change,
                    'success': result['success'],
                    'details': result.get('details', '')
                })
            except Exception as e:
                logger.error(f"Error applying change to {change.file_path}: {str(e)}")
                results.append({
                    'change': change,
                    'success': False,
                    'details': str(e)
                })
        
        return results
    
    async def _apply_single_change(self, change: CodeChange, session_id: str, 
                                 sandbox_manager) -> Dict[str, Any]:
        """Apply a single code change"""
        
        if change.change_type == 'create':
            # Create new file
            result = await sandbox_manager.write_file(session_id, change.file_path, change.new_content)
            return {
                'success': result['success'],
                'details': f"Created file {change.file_path}",
                'tool_event': {
                    'tool_name': 'write_file',
                    'tool_input': {
                        'file': change.file_path,
                        'old_str': '',
                        'new_str': change.new_content[:200] + '...' if len(change.new_content) > 200 else change.new_content
                    }
                }
            }
        
        elif change.change_type == 'modify':
            # Modify existing file
            read_result = await sandbox_manager.read_file(session_id, change.file_path)
            if not read_result['success']:
                return {
                    'success': False,
                    'details': f"Could not read file {change.file_path}"
                }
            
            current_content = read_result['content']
            
            if change.old_content and change.old_content in current_content:
                # Replace specific content
                new_content = current_content.replace(change.old_content, change.new_content)
            else:
                # Append to file
                new_content = current_content + '\n' + change.new_content
            
            write_result = await sandbox_manager.write_file(session_id, change.file_path, new_content)
            return {
                'success': write_result['success'],
                'details': f"Modified file {change.file_path}"
            }
        
        elif change.change_type == 'add':
            # Add content to existing file
            read_result = await sandbox_manager.read_file(session_id, change.file_path)
            if read_result['success']:
                current_content = read_result['content']
                new_content = current_content + '\n\n' + change.new_content
            else:
                new_content = change.new_content
            
            write_result = await sandbox_manager.write_file(session_id, change.file_path, new_content)
            return {
                'success': write_result['success'],
                'details': f"Added content to {change.file_path}"
            }
        
        else:
            return {
                'success': False,
                'details': f"Unknown change type: {change.change_type}"
            }
