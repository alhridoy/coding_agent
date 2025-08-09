"""
E2B Sandbox Integration for Autonomous Coding Agent
Provides secure, isolated environment for code execution
"""

import logging
import uuid
from typing import Dict, Any
from e2b_code_interpreter import Sandbox
from ..utils.config import get_settings

logger = logging.getLogger(__name__)


class E2BSandbox:
    """E2B-based sandbox manager for secure code execution"""
    
    def __init__(self):
        self.settings = get_settings()
        self.active_sessions: Dict[str, Sandbox] = {}
        self.session_repo_paths: Dict[str, str] = {}
        self.session_tokens: Dict[str, str] = {}
        
    async def create_session(self, session_id: str = None) -> str:
        """Create a new E2B sandbox session"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        try:
            timeout_seconds = min(self.settings.sandbox_timeout_minutes * 60, 3600)
            logger.info(f"Creating E2B sandbox with timeout: {timeout_seconds} seconds")
            
            sandbox = Sandbox(
                api_key=self.settings.e2b_api_key,
                timeout=timeout_seconds
            )
            
            self.active_sessions[session_id] = sandbox
            await self._initialize_sandbox(sandbox)
            
            logger.info(f"Created E2B sandbox session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create E2B sandbox session: {e}")
            try:
                logger.info("Retrying with default timeout (600 seconds)")
                sandbox = Sandbox(api_key=self.settings.e2b_api_key, timeout=600)
                self.active_sessions[session_id] = sandbox
                await self._initialize_sandbox(sandbox)
                logger.info(f"Created E2B sandbox session with default timeout: {session_id}")
                return session_id
            except Exception as e2:
                logger.error(f"Failed to create E2B sandbox with default timeout: {e2}")
                raise
    
    async def _initialize_sandbox(self, sandbox: Sandbox):
        """Initialize sandbox with basic tools"""
        try:
            # Install GitHub CLI
            gh_install = sandbox.run_code("!curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && echo \"deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main\" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null && sudo apt update && sudo apt install gh -y")
            if gh_install.error:
                logger.warning(f"Failed to install GitHub CLI: {gh_install.error}")

            # Install git
            git_install = sandbox.run_code("!apt-get update && apt-get install -y git")
            if git_install.error:
                logger.warning(f"Failed to install git: {git_install.error}")

            # Set up git configuration
            sandbox.run_code("!git config --global user.email 'agent@backspace.com'")
            sandbox.run_code("!git config --global user.name 'Backspace Agent'")
            sandbox.run_code("!git config --global init.defaultBranch main")

        except Exception as e:
            logger.warning(f"Failed to initialize sandbox: {e}")
    
    async def execute_command(self, session_id: str, command: str, 
                            working_dir: str = None, timeout: int = 300) -> Dict[str, Any]:
        """Execute a command in the sandbox"""
        logger.info(f"E2B: execute_command called with command='{command}', working_dir='{working_dir}'")
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}
        
        sandbox = self.active_sessions[session_id]
        
        try:
            # Handle working directory for commands
            if working_dir:
                full_command = f"cd {working_dir} && {command}"
            else:
                full_command = command
            
            # Ensure shell commands use ! prefix
            if not full_command.startswith("!"):
                full_command = "!" + full_command
            
            result = sandbox.run_code(full_command)
            
            # Handle E2B response format
            if hasattr(result, 'stdout'):
                return {
                    "success": not result.error,
                    "stdout": result.stdout or "",
                    "stderr": result.stderr or "",
                    "exit_code": 0 if not result.error else 1
                }
            else:
                # Handle logs format
                stdout = ""
                stderr = ""
                if hasattr(result, 'logs'):
                    if hasattr(result.logs, 'stdout') and result.logs.stdout:
                        stdout = '\n'.join([str(line) for line in result.logs.stdout])
                    if hasattr(result.logs, 'stderr') and result.logs.stderr:
                        stderr = '\n'.join([str(line) for line in result.logs.stderr])
                
                return {
                    "success": not result.error,
                    "stdout": stdout,
                    "stderr": stderr or str(result.error) if result.error else "",
                    "exit_code": 0 if not result.error else 1
                }
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def read_file(self, session_id: str, file_path: str) -> Dict[str, Any]:
        """Read a file from the sandbox"""
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}
        
        sandbox = self.active_sessions[session_id]
        
        try:
            # Use Python to read file safely
            python_code = f'''
try:
    with open("{file_path}", "r") as f:
        content = f.read()
    print(content, end="")
except FileNotFoundError:
    print("FILE_NOT_FOUND", end="")
except Exception as e:
    print(f"ERROR: {{e}}", end="")
'''
            result = sandbox.run_code(python_code)
            
            if not result.error:
                # Extract content from result
                if hasattr(result, 'stdout'):
                    content = result.stdout
                elif hasattr(result, 'logs') and hasattr(result.logs, 'stdout'):
                    content = '\n'.join([str(line) for line in result.logs.stdout])
                else:
                    content = str(result)
                
                if content == "FILE_NOT_FOUND":
                    return {"success": False, "error": "File not found"}
                elif content.startswith("ERROR:"):
                    return {"success": False, "error": content[6:]}
                else:
                    return {"success": True, "content": content}
            else:
                return {"success": False, "error": str(result.error)}
                
        except Exception as e:
            logger.error(f"File read failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def write_file(self, session_id: str, file_path: str, content: str) -> Dict[str, Any]:
        """Write content to a file in the sandbox"""
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}
        
        sandbox = self.active_sessions[session_id]
        
        try:
            # Create directory if needed
            dir_path = "/".join(file_path.split("/")[:-1])
            if dir_path:
                sandbox.run_code(f"!mkdir -p {dir_path}")
            
            # Use Python to write file safely (handles newlines and special characters)
            python_code = f'''
with open("{file_path}", "w") as f:
    f.write({repr(content)})
print("File written successfully")
'''
            result = sandbox.run_code(python_code)
            
            if not result.error:
                logger.info(f"Successfully wrote to file: {file_path}")
                return {"success": True}
            else:
                return {"success": False, "error": str(result.error)}
                
        except Exception as e:
            logger.error(f"File write failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def clone_repository(self, session_id: str, repo_url: str,
                             branch: str = "main") -> Dict[str, Any]:
        """Clone a repository in the sandbox"""
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}

        try:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repo_path = f"/home/user/{repo_name}"

            # Clone repository
            clone_result = await self.execute_command(
                session_id,
                f"git clone {repo_url} {repo_path}",
                timeout=120
            )

            if not clone_result["success"]:
                return clone_result

            self.session_repo_paths[session_id] = repo_path
            
            # Checkout branch if needed
            if branch != "main":
                await self.execute_command(session_id, f"git -C {repo_path} checkout {branch}")

            # Count files in the repository
            count_result = await self.execute_command(
                session_id,
                f"find {repo_path} -type f | wc -l"
            )
            
            file_count = 0
            if count_result["success"]:
                try:
                    file_count = int(count_result["stdout"].strip())
                except (ValueError, AttributeError):
                    file_count = 0

            return {
                "success": True,
                "repo_name": repo_name,
                "repo_path": repo_path,
                "branch": branch,
                "file_count": file_count
            }
            
        except Exception as e:
            logger.error(f"Repository clone failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def setup_git_auth(self, session_id: str, github_token: str) -> Dict[str, Any]:
        """Setup Git authentication for GitHub operations"""
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}

        try:
            self.session_tokens[session_id] = github_token

            # Setup git credentials
            credential_helper = f"!git config --global credential.helper 'store --file=/tmp/git-credentials'"
            await self.execute_command(session_id, credential_helper)
            
            # Store credentials
            credential_content = f"https://{github_token}:x-oauth-basic@github.com"
            await self.execute_command(session_id, f"!echo '{credential_content}' > /tmp/git-credentials")
            
            # Set git config
            await self.execute_command(session_id, "!git config --global user.email 'agent@backspace.com'")
            await self.execute_command(session_id, "!git config --global user.name 'Backspace Agent'")
            
            return {"success": True, "method": "git_credentials"}
            
        except Exception as e:
            logger.error(f"Git auth setup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def push_branch(self, session_id: str, branch_name: str) -> Dict[str, Any]:
        """Push a branch to GitHub"""
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}

        try:
            repo_path = self.session_repo_paths.get(session_id)
            if not repo_path:
                return {"success": False, "error": "Repository path not found"}

            # Push branch
            push_result = await self.execute_command(
                session_id,
                f"git -C {repo_path} push -u origin {branch_name}",
                timeout=120
            )

            if push_result["success"]:
                return {"success": True, "branch": branch_name}
            else:
                # Try with authenticated URL
                github_token = self.session_tokens.get(session_id)
                if github_token:
                    remote_result = await self.execute_command(
                        session_id,
                        f"git -C {repo_path} remote get-url origin"
                    )
                    
                    if remote_result["success"]:
                        remote_url = remote_result["stdout"].strip()
                        auth_url = remote_url.replace("https://github.com/", f"https://{github_token}@github.com/")
                        
                        auth_push = await self.execute_command(
                            session_id,
                            f"git -C {repo_path} push -u {auth_url} {branch_name}",
                            timeout=120
                        )
                        
                        return auth_push

            return push_result
            
        except Exception as e:
            logger.error(f"Branch push failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_pull_request(self, session_id: str, title: str,
                                body: str, branch_name: str) -> Dict[str, Any]:
        """Create a pull request"""
        if session_id not in self.active_sessions:
            return {"success": False, "error": "Session not found"}

        try:
            repo_path = self.session_repo_paths.get(session_id)
            if not repo_path:
                return {"success": False, "error": "Repository path not found"}

            github_token = self.session_tokens.get(session_id)
            if not github_token:
                return {"success": False, "error": "GitHub token not found"}

            # Get repository info
            remote_result = await self.execute_command(
                session_id,
                f"git -C {repo_path} remote get-url origin"
            )

            if not remote_result["success"]:
                return {"success": False, "error": "Failed to get remote URL"}

            remote_url = remote_result["stdout"].strip()

            # Extract owner and repo
            parts = remote_url.replace('https://github.com/', '').replace('.git', '').split('/')
            owner = parts[0]
            repo = parts[1]

            # Authenticate gh
            auth_result = await self.execute_command(
                session_id,
                f"gh auth login --with-token <<< {github_token}"
            )

            if not auth_result["success"]:
                return {"success": False, "error": f"GH auth failed: {auth_result['stderr']}"}

            # Create PR
            escaped_title = title.replace("'", "'\\''")
            escaped_body = body.replace("'", "'\\''")
            pr_command = f"gh pr create --title '{escaped_title}' --body '{escaped_body}' --head {branch_name} --base main"

            pr_result = await self.execute_command(
                session_id,
                pr_command,
                working_dir=repo_path
            )

            if pr_result["success"]:
                pr_url = pr_result["stdout"].strip()
                return {"success": True, "pr_url": pr_url}
            else:
                return {"success": False, "error": pr_result["stderr"]}

        except Exception as e:
            logger.error(f"PR creation failed: {e}")
            return {"success": False, "error": str(e)}

    async def cleanup_session(self, session_id: str) -> None:
        """Cleanup a specific sandbox session"""
        if session_id in self.active_sessions:
            try:
                sandbox = self.active_sessions[session_id]
                # E2B sandbox cleanup - it handles itself via garbage collection
                if hasattr(sandbox, 'close'):
                    sandbox.close()
                logger.info(f"Closed sandbox session: {session_id}")
            except Exception as e:
                logger.warning(f"Error closing session {session_id}: {e}")
            finally:
                del self.active_sessions[session_id]
                
            if session_id in self.session_repo_paths:
                del self.session_repo_paths[session_id]
            if session_id in self.session_tokens:
                del self.session_tokens[session_id]

    async def cleanup_all(self) -> None:
        """Cleanup all active sessions"""
        for session_id in list(self.active_sessions.keys()):
            await self.cleanup_session(session_id)
        logger.info("All sessions cleaned up")
