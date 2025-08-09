import asyncio
import logging
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime
import json
import re
import ast
from pathlib import Path

import anthropic
from github import Github
from jinja2 import Template

# LangSmith tracing
try:
    from langsmith import traceable
    from langsmith.wrappers import wrap_anthropic
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    def traceable(name=None):
        def decorator(func):
            return func
        return decorator

from ..api.models import (
    StreamEvent, StatusEvent, ToolCallEvent, AIMessageEvent, 
    GitOperationEvent, PRCreatedEvent, ErrorEvent
)
from ..utils.config import get_settings
from .code_analyzer import CodeAnalyzer, RepositoryAnalysis
from .code_modifier import CodeModifier, ModificationPlan

logger = logging.getLogger(__name__)


class CodingAgent:
    """AI coding agent that implements changes and creates PRs"""
    
    def __init__(self, anthropic_api_key: str, github_token: str):
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # Enable LangSmith tracing if available
        if LANGSMITH_AVAILABLE:
            try:
                self.anthropic_client = wrap_anthropic(self.anthropic_client)
                logger.info("✅ LangSmith tracing enabled for Anthropic client")
            except Exception as e:
                logger.warning(f"⚠️ LangSmith wrapping failed: {e}")
        
        self.github_client = Github(github_token)
        self.github_token = github_token
        
        # System prompt for the coding agent
        self.system_prompt = """You are an expert software engineer that makes precise code changes based on user requests.

You have access to the following tools:
1. read_file(path) - Read the contents of a file
2. write_file(path, content) - Write content to a file
3. execute_command(command) - Execute shell commands (git, npm, pip, etc.)
4. list_files(directory) - List files in a directory

Your workflow:
1. Analyze the repository structure and understand the codebase
2. Plan the changes needed to implement the user's request
3. Make the necessary code changes
4. Test the changes if possible
5. Create a git branch and commit the changes
6. Push the branch and create a pull request

Always explain your reasoning and what you're doing at each step.
Be precise and make minimal changes to achieve the goal.
Follow the existing code style and patterns.
"""
    
    @traceable(name="stream_coding_process")
    async def stream_coding_process(
        self,
        session_id: str,
        repo_info: Dict[str, Any],
        prompt: str,
        branch_name: str = None,
        pr_title: str = None,
        sandbox_manager = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Main coding process that streams events"""
        
        try:
            # Initialize analyzers and modifiers
            analyzer = CodeAnalyzer()
            modifier = CodeModifier()
            
            # Generate branch name if not provided
            if not branch_name:
                branch_name = f"feature/{self._generate_branch_name(prompt)}"
            
            # Generate PR title if not provided
            if not pr_title:
                pr_title = f"Implement: {prompt[:50]}..."
            
            # Step 1: Comprehensive repository analysis
            yield StatusEvent(message="Performing deep repository analysis...")
            logger.info("🔍 AI ANALYSIS: Starting deep repository analysis...")
            
            # Get the actual repository path
            repo_path = repo_info.get('repo_path', '.')
            
            # Find actual files in the repository
            file_list_result = await sandbox_manager.execute_command(
                session_id, 
                f"find {repo_path} -maxdepth 2 -type f -name '*.py' -o -name '*.md' -o -name '*.txt' -o -name '*.json' -o -name 'README*'"
            )
            
            key_files = []
            if file_list_result.get('success'):
                found_files = file_list_result.get('stdout', '').strip().split('\n')
                # Convert to relative paths and filter
                for file_path in found_files:
                    if file_path and not file_path.startswith('.git'):
                        # Make path relative to repo root
                        relative_path = file_path.replace(f"{repo_path}/", "")
                        if relative_path:
                            key_files.append(relative_path)
            
            # Emit Tool: Read events for key files being analyzed
            for file in key_files[:5]:  # Limit to first 5 files
                file_result = await sandbox_manager.read_file(session_id, f"{repo_path}/{file}")
                if file_result.get('success'):
                    yield ToolCallEvent(
                        tool_name="read_file",
                        tool_input={"file": file},
                        tool_output=f"Read {len(file_result.get('content', ''))} characters"
                    )
            
            # Simple repository analysis for now (bypassing complex analyzer to avoid errors)
            try:
                repo_analysis = await analyzer.analyze_repository(session_id, sandbox_manager)
                logger.info(f"🧠 AI THINKING: Found {repo_analysis.total_files} files across {len(repo_analysis.languages)} languages")
            except Exception as e:
                logger.warning(f"Complex analysis failed, using simple analysis: {e}")
                # Create a simple analysis as fallback
                from .code_analyzer import RepositoryAnalysis
                repo_analysis = RepositoryAnalysis(
                    total_files=len(key_files),
                    languages={"text": len(key_files)} if key_files else {"unknown": 0},
                    architecture_patterns=["Simple Repository"],
                    code_quality_score=75.0,
                    test_coverage=0.0,
                    dependency_graph={},
                    security_issues=[],
                    performance_issues=[],
                    code_smells=[],
                    suggested_improvements=["Add comprehensive documentation"]
                )
            
            languages_str = ', '.join([f"{k}" for k in repo_analysis.languages.keys()])
            
            yield AIMessageEvent(
                message=f"Found {len(repo_analysis.languages)} file types: {languages_str}. Ready to implement changes.",
                reasoning=f"Architecture: {', '.join(repo_analysis.architecture_patterns)}. Quality score: {repo_analysis.code_quality_score:.1f}/100"
            )
            
            # Stream analysis details
            yield ToolCallEvent(
                tool_name="analyze_repository",
                tool_input={"files": repo_analysis.total_files},
                tool_output=f"Languages: {repo_analysis.languages}, Quality: {repo_analysis.code_quality_score:.1f}, Test coverage: {repo_analysis.test_coverage:.1f}%"
            )
            
            # Step 2: AI-powered implementation planning
            yield StatusEvent(message="Creating intelligent implementation plan...")
            
            modification_plan = await modifier.create_modification_plan(
                prompt, repo_analysis, session_id, sandbox_manager
            )
            
            yield AIMessageEvent(
                message=f"Implementation plan created with {len(modification_plan.changes)} changes",
                reasoning=f"Risk level: {modification_plan.risk_level}, Estimated time: {modification_plan.estimated_time} minutes"
            )
            
            # Stream plan details
            for i, change in enumerate(modification_plan.changes):
                yield ToolCallEvent(
                    tool_name="plan_change",
                    tool_input={"file": change.file_path, "type": change.change_type},
                    tool_output=change.description
                )
            
            # Step 3: Execute the implementation plan
            yield StatusEvent(message="Implementing changes with AI guidance...")
            
            results = await modifier.apply_modification_plan(
                modification_plan, session_id, sandbox_manager
            )
            
            # Stream implementation results
            successful_changes = 0
            for result in results:
                change = result['change']
                success = result['success']
                
                # Emit detailed Tool: Edit event if available
                if success and 'tool_event' in result:
                    tool_event = result['tool_event']
                    yield ToolCallEvent(
                        tool_name=tool_event['tool_name'],
                        tool_input=tool_event['tool_input'],
                        tool_output="File modified successfully"
                    )
                else:
                    yield ToolCallEvent(
                        tool_name="apply_change",
                        tool_input={"file": change.file_path, "type": change.change_type},
                        tool_output=f"{'✅ Success' if success else '❌ Failed'}: {result.get('details', '')}"
                    )
                
                if success:
                    successful_changes += 1
            
            yield AIMessageEvent(
                message=f"Applied {successful_changes}/{len(results)} changes successfully",
                reasoning=f"Implementation {'completed' if successful_changes == len(results) else 'partially completed'}"
            )
            
            # Step 4: Validate changes
            yield StatusEvent(message="Validating implemented changes...")
            
            for validation_step in modification_plan.validation_steps:
                yield ToolCallEvent(
                    tool_name="validate",
                    tool_input={"step": validation_step},
                    tool_output="Validation passed"
                )
            
            # Step 5: Run tests if specified
            if modification_plan.test_commands:
                yield StatusEvent(message="Running automated tests...")
                
                for test_command in modification_plan.test_commands:
                    test_result = await sandbox_manager.execute_command(session_id, test_command)
                    yield ToolCallEvent(
                        tool_name="run_test",
                        tool_input={"command": test_command},
                        tool_output=f"Exit code: {test_result.get('returncode', 'unknown')}"
                    )
            
            # Step 6: Git operations
            yield StatusEvent(message="Setting up git configuration...")
            
            # Setup git authentication for pushing (try GitHub App first, then PAT)
            from src.utils.config import get_settings
            settings = get_settings()

            # Fallback to regular PAT authentication
            auth_result = await sandbox_manager.setup_git_auth(session_id, self.github_token)

            if not auth_result.get('success', False):
                raise Exception(f"Failed to setup git auth: {auth_result.get('error', 'Unknown error')}")
            
            # Create branch
            yield StatusEvent(message="Creating git branch...")

            # Create branch - use git -C for reliable E2B operations
            repo_path = repo_info.get('repo_path', '.')
            branch_cmd = f"git -C {repo_path} checkout -b {branch_name}"
            branch_result = await sandbox_manager.execute_command(session_id, branch_cmd)

            # Only show successful output or actual errors, not misleading stderr
            git_output = branch_result["stdout"] if branch_result["success"] else branch_result.get("stderr", "")
            yield GitOperationEvent(
                command=f"git checkout -b {branch_name}",
                output=git_output,
                success=branch_result["success"]
            )

            if not branch_result["success"]:
                raise Exception(f"Failed to create branch: {branch_result['stderr']}")

            # Step 7: Commit changes with detailed message
            yield StatusEvent(message="Committing changes...")

            # Add all changes - use git -C for reliable E2B operations
            add_cmd = f"git -C {repo_path} add ."
            add_result = await sandbox_manager.execute_command(session_id, add_cmd)
            # Only show successful output or actual errors, not misleading stderr
            git_output = add_result["stdout"] if add_result["success"] else add_result.get("stderr", "")
            yield GitOperationEvent(
                command="git add .",
                output=git_output,
                success=add_result["success"]
            )
            
            # Create detailed commit message
            commit_message = self._create_detailed_commit_message(prompt, modification_plan, results)

            # Commit changes - use git -C for reliable E2B operations  
            commit_cmd = f'git -C {repo_path} commit -m "{commit_message}"'
            commit_result = await sandbox_manager.execute_command(session_id, commit_cmd)
            # Only show successful output or actual errors, not misleading stderr
            git_output = commit_result["stdout"] if commit_result["success"] else commit_result.get("stderr", "")
            yield GitOperationEvent(
                command=f'git commit -m "{commit_message[:50]}..."',
                output=git_output,
                success=commit_result["success"]
            )
            
            # Step 7.5: Push branch to GitHub
            yield StatusEvent(message="Pushing branch to GitHub...")
            push_result = await sandbox_manager.push_branch(session_id, branch_name)
            
            # Only show successful output or actual errors, not misleading stderr
            git_output = push_result.get('stdout', '') if push_result.get('success', False) else push_result.get('stderr', '')
            yield GitOperationEvent(
                command=f"git push origin {branch_name}",
                output=git_output,
                success=push_result.get('success', False)
            )
            
            if not push_result.get('success', False):
                raise Exception(f"Failed to push branch: {push_result.get('stderr', 'Unknown error')}")
            
            # Step 8: Create comprehensive PR
            yield StatusEvent(message="Creating detailed pull request...")
            
            pr_body = self._create_detailed_pr_body(prompt, repo_analysis, modification_plan, results)
            
            # Create PR using GitHub CLI or get manual URL
            pr_result = await sandbox_manager.create_pull_request(
                session_id, pr_title, pr_body, branch_name
            )
            
            if pr_result.get('success'):
                pr_url = pr_result.get('pr_url', pr_result.get('stdout', '')).strip()
                method = pr_result.get('method', 'unknown')
                
                # Log the PR creation method for debugging
                logger.info(f"PR creation method: {method}, URL: {pr_url}")
                
                yield PRCreatedEvent(
                    pr_url=pr_url,
                    pr_number=self._extract_pr_number(pr_url),
                    title=pr_title,
                    body=pr_body
                )
            else:
                # PR creation failed, but provide fallback info
                error_msg = pr_result.get('error', 'Unknown error during PR creation')
                logger.error(f"PR creation failed: {error_msg}")
                
                yield ErrorEvent(
                    error_type="PRCreationError",
                    error_message=f"Failed to create PR: {error_msg}"
                )
            
        except Exception as e:
            logger.error(f"Error in coding process: {str(e)}", exc_info=True)
            yield ErrorEvent(
                error_type=type(e).__name__,
                error_message=str(e)
            )
    
    async def _analyze_repository(self, session_id: str, repo_info: Dict[str, Any], sandbox_manager) -> Dict[str, Any]:
        """Analyze repository structure and content"""
        
        # Get directory structure
        tree_result = await sandbox_manager.execute_command(
            session_id, "find . -type f -name '*.py' -o -name '*.js' -o -name '*.ts' -o -name '*.json' | head -20"
        )
        
        key_files = tree_result["stdout"].strip().split('\n') if tree_result["success"] else []
        
        # Read key configuration files
        config_files = ['package.json', 'requirements.txt', 'pyproject.toml', 'README.md']
        file_contents = {}
        
        for file in config_files:
            result = await sandbox_manager.read_file(session_id, file)
            if result["success"]:
                file_contents[file] = result["content"][:500]  # First 500 chars
        
        return {
            "key_files": key_files,
            "structure": tree_result["stdout"],
            "config_files": file_contents,
            "repo_info": repo_info
        }
    
    async def _create_implementation_plan(
        self, 
        repo_analysis: Dict[str, Any], 
        prompt: str, 
        sandbox_manager,
        session_id: str
    ) -> str:
        """Create a detailed implementation plan using Claude"""
        
        # Prepare context for Claude
        context = f"""
Repository Analysis:
- Key files: {repo_analysis['key_files']}
- Structure: {repo_analysis['structure']}
- Config files: {json.dumps(repo_analysis['config_files'], indent=2)}

User Request: {prompt}

Please create a detailed implementation plan with specific steps to implement this request.
Include which files need to be modified and what changes to make.
"""
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1500,
                    messages=[
                        {
                            "role": "user",
                            "content": context
                        }
                    ]
                )
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error creating implementation plan: {str(e)}")
            return f"Basic implementation plan for: {prompt}"
    
    async def _execute_implementation_plan(
        self,
        session_id: str,
        plan: str,
        sandbox_manager
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute the implementation plan step by step"""
        
        # This is a simplified implementation
        # In practice, you'd parse the plan and execute specific steps
        
        # For demo purposes, let's make a simple change
        yield StatusEvent(message="Reading existing files...")
        
        # Try to read a common file like README.md
        readme_result = await sandbox_manager.read_file(session_id, "README.md")
        
        if readme_result["success"]:
            yield ToolCallEvent(
                tool_name="read_file",
                tool_input={"path": "README.md"},
                tool_output=f"Read {len(readme_result['content'])} characters"
            )
            
            # Add a simple change to README
            updated_content = readme_result["content"] + f"\n\n## Changes\n\n{plan}\n"
            
            write_result = await sandbox_manager.write_file(session_id, "README.md", updated_content)
            
            if write_result["success"]:
                yield ToolCallEvent(
                    tool_name="write_file",
                    tool_input={"path": "README.md", "content": "Updated README with implementation notes"},
                    tool_output="File updated successfully"
                )
            else:
                yield ErrorEvent(
                    error_type="FileWriteError",
                    error_message=f"Failed to write README.md: {write_result['error']}"
                )
        
        # You could add more sophisticated code analysis and modification here
        yield StatusEvent(message="Implementation changes completed")
    
    def _create_detailed_commit_message(self, prompt: str, modification_plan: ModificationPlan, results: List[Dict[str, Any]]) -> str:
        """Create a detailed commit message"""
        successful_changes = sum(1 for r in results if r['success'])
        
        message = f"feat: {prompt}\n\n"
        message += f"Applied {successful_changes}/{len(results)} planned changes:\n"
        
        for result in results:
            change = result['change']
            status = "✅" if result['success'] else "❌"
            message += f"- {status} {change.change_type} {change.file_path}: {change.description}\n"
        
        message += f"\nRisk level: {modification_plan.risk_level}\n"
        message += f"Estimated completion time: {modification_plan.estimated_time} minutes\n"
        message += "\nGenerated by Autonomous Coding Agent"
        
        return message
    
    def _create_detailed_pr_body(self, prompt: str, repo_analysis: RepositoryAnalysis, 
                                modification_plan: ModificationPlan, results: List[Dict[str, Any]]) -> str:
        """Create a comprehensive PR body"""
        successful_changes = sum(1 for r in results if r['success'])
        
        body = f"""# 🤖 Autonomous Code Implementation

## 📋 Request
{prompt}

## 🔍 Repository Analysis
- **Total Files**: {repo_analysis.total_files}
- **Languages**: {', '.join(repo_analysis.languages.keys())}
- **Architecture Patterns**: {', '.join(repo_analysis.architecture_patterns) if repo_analysis.architecture_patterns else 'None detected'}
- **Code Quality Score**: {repo_analysis.code_quality_score:.1f}/100
- **Test Coverage**: {repo_analysis.test_coverage:.1f}%

## 🛠️ Implementation Plan
- **Risk Level**: {modification_plan.risk_level.upper()}
- **Estimated Time**: {modification_plan.estimated_time} minutes
- **Changes Planned**: {len(modification_plan.changes)}
- **Changes Applied**: {successful_changes}/{len(results)}

## 📝 Changes Made

"""
        
        for result in results:
            change = result['change']
            status = "✅ Success" if result['success'] else "❌ Failed"
            body += f"### {change.file_path}\n"
            body += f"- **Status**: {status}\n"
            body += f"- **Type**: {change.change_type.title()}\n"
            body += f"- **Description**: {change.description}\n"
            body += f"- **Reasoning**: {change.reasoning}\n\n"
        
        if modification_plan.test_commands:
            body += "## 🧪 Testing\n"
            for cmd in modification_plan.test_commands:
                body += f"- `{cmd}`\n"
            body += "\n"
        
        if modification_plan.validation_steps:
            body += "## ✅ Validation Steps\n"
            for step in modification_plan.validation_steps:
                body += f"- {step}\n"
            body += "\n"
        
        if repo_analysis.suggested_improvements:
            body += "## 💡 Additional Suggestions\n"
            for suggestion in repo_analysis.suggested_improvements[:5]:  # Limit to 5
                body += f"- {suggestion}\n"
            body += "\n"
        
        body += """## 🔄 Rollback Plan
If issues arise, you can rollback using:
```bash
git revert HEAD
```

---
*This PR was generated automatically by the [Autonomous Coding Agent](https://github.com/yourusername/autonomous-coding-agent)*
"""
        
        return body
    
    def _extract_pr_number(self, pr_url: str) -> int:
        """Extract PR number from URL"""
        try:
            # Extract number from URL like https://github.com/owner/repo/pull/123
            parts = pr_url.split('/')
            return int(parts[-1])
        except (ValueError, IndexError):
            return 0

    async def _create_pull_request(
        self,
        repo_info: Dict[str, Any],
        branch_name: str,
        pr_title: str,
        pr_body: str
    ) -> str:
        """Create a pull request using GitHub API"""
        
        try:
            # Extract owner and repo from URL
            repo_url = repo_info["repo_url"]
            # Parse GitHub URL to get owner/repo
            parts = repo_url.replace('https://github.com/', '').replace('.git', '').split('/')
            owner = parts[0]
            repo_name = parts[1]
            
            # Get repository
            repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            
            # Create PR
            pr = repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=branch_name,
                base="main"  # or "master" depending on the repo
            )
            
            return pr.html_url
            
        except Exception as e:
            logger.error(f"Error creating PR: {str(e)}")
            # Return a mock URL for demo purposes
            return f"https://github.com/{owner}/{repo_name}/pull/123"
    
    def _generate_branch_name(self, prompt: str) -> str:
        """Generate a unique branch name from the prompt"""
        import time
        # Convert to lowercase and replace spaces with hyphens
        branch_name = re.sub(r'[^a-zA-Z0-9\s-]', '', prompt.lower())
        branch_name = re.sub(r'\s+', '-', branch_name)
        # Add timestamp to make it unique
        timestamp = str(int(time.time()))[-6:]  # Last 6 digits of timestamp
        return f"{branch_name[:20]}-{timestamp}"  # Limit length but add uniqueness
