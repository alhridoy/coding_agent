"""
Environment Isolation and Cleanup Validator
Implements ABC checklist items T.4-T.6 for rigorous environment management
"""

import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentState:
    """Represents the state of an environment at a point in time"""
    timestamp: float
    session_id: str
    file_hashes: Dict[str, str]  # filepath -> hash
    environment_vars: Dict[str, str]
    running_processes: List[str]
    open_ports: List[int]
    memory_usage: float
    disk_usage: float
    sandbox_metadata: Dict[str, Any]


@dataclass
class IsolationViolation:
    """Represents a violation of environment isolation"""
    violation_type: str  # 'cross_contamination', 'state_leak', 'resource_leak'
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    affected_resources: List[str]
    recommendation: str


@dataclass
class EnvironmentValidationResult:
    """Result of environment validation"""
    is_clean: bool
    is_isolated: bool
    violations: List[IsolationViolation]
    state_before: Optional[EnvironmentState]
    state_after: Optional[EnvironmentState]
    cleanup_successful: bool
    isolation_score: float  # 0-1


class EnvironmentValidator:
    """
    Validates environment isolation and cleanup between test runs.
    Ensures no cross-contamination or state leakage.
    """
    
    def __init__(self, sandbox_manager: Optional[Any] = None):
        self.sandbox_manager = sandbox_manager
        self.baseline_states: Dict[str, EnvironmentState] = {}
        self.session_states: Dict[str, List[EnvironmentState]] = {}
        self.protected_resources: Set[str] = {
            '.env',
            'config.json', 
            'settings.py',
            'credentials',
            'secrets',
            '.git/config'
        }
        
    async def capture_environment_state(self, session_id: str) -> EnvironmentState:
        """Capture current environment state"""
        logger.info(f"Capturing environment state for session {session_id}")
        
        state = EnvironmentState(
            timestamp=time.time(),
            session_id=session_id,
            file_hashes={},
            environment_vars={},
            running_processes=[],
            open_ports=[],
            memory_usage=0.0,
            disk_usage=0.0,
            sandbox_metadata={}
        )
        
        if self.sandbox_manager:
            try:
                # Get file system state
                state.file_hashes = await self._capture_file_hashes(session_id)
                
                # Get environment variables
                state.environment_vars = await self._capture_env_vars(session_id)
                
                # Get process information
                state.running_processes = await self._capture_processes(session_id)
                
                # Get resource usage
                state.memory_usage, state.disk_usage = await self._capture_resource_usage(session_id)
                
                # Get sandbox metadata
                state.sandbox_metadata = await self._capture_sandbox_metadata(session_id)
                
            except Exception as e:
                logger.error(f"Error capturing environment state: {e}")
        
        # Store state
        if session_id not in self.session_states:
            self.session_states[session_id] = []
        self.session_states[session_id].append(state)
        
        return state
    
    async def validate_isolation(
        self,
        session_id: str,
        test_case: 'TestCase',
        state_before: EnvironmentState,
        state_after: EnvironmentState
    ) -> EnvironmentValidationResult:
        """Validate environment isolation for a test run"""
        logger.info(f"Validating environment isolation for session {session_id}")
        
        violations = []
        
        # Check 1: File system isolation
        file_violations = self._check_file_system_isolation(
            state_before,
            state_after,
            test_case
        )
        violations.extend(file_violations)
        
        # Check 2: Environment variable isolation
        env_violations = self._check_env_var_isolation(
            state_before,
            state_after
        )
        violations.extend(env_violations)
        
        # Check 3: Process isolation
        process_violations = self._check_process_isolation(
            state_before,
            state_after
        )
        violations.extend(process_violations)
        
        # Check 4: Resource cleanup
        resource_violations = self._check_resource_cleanup(
            state_before,
            state_after
        )
        violations.extend(resource_violations)
        
        # Check 5: No access to protected resources
        protected_violations = self._check_protected_resources(
            state_after,
            test_case
        )
        violations.extend(protected_violations)
        
        # Check 6: Cross-session contamination
        cross_violations = await self._check_cross_contamination(
            session_id,
            state_after
        )
        violations.extend(cross_violations)
        
        # Calculate isolation score
        isolation_score = self._calculate_isolation_score(violations)
        
        # Determine overall status
        is_clean = len([v for v in violations if v.severity in ['critical', 'high']]) == 0
        is_isolated = isolation_score > 0.8
        
        # Check cleanup
        cleanup_successful = await self._verify_cleanup(session_id, state_after)
        
        return EnvironmentValidationResult(
            is_clean=is_clean,
            is_isolated=is_isolated,
            violations=violations,
            state_before=state_before,
            state_after=state_after,
            cleanup_successful=cleanup_successful,
            isolation_score=isolation_score
        )
    
    def _check_file_system_isolation(
        self,
        state_before: EnvironmentState,
        state_after: EnvironmentState,
        test_case: 'TestCase'
    ) -> List[IsolationViolation]:
        """Check for file system isolation violations"""
        violations = []
        
        # Check for unexpected file modifications
        for filepath, after_hash in state_after.file_hashes.items():
            before_hash = state_before.file_hashes.get(filepath)
            
            # Skip expected modifications
            if self._is_expected_modification(filepath, test_case):
                continue
                
            if before_hash and before_hash != after_hash:
                violations.append(IsolationViolation(
                    violation_type="state_leak",
                    severity="high",
                    description=f"Unexpected file modification: {filepath}",
                    affected_resources=[filepath],
                    recommendation="Ensure test only modifies intended files"
                ))
        
        # Check for files created outside sandbox
        sandbox_paths = self._get_sandbox_paths(test_case)
        for filepath in state_after.file_hashes:
            if filepath not in state_before.file_hashes:
                if not any(filepath.startswith(sp) for sp in sandbox_paths):
                    violations.append(IsolationViolation(
                        violation_type="state_leak",
                        severity="critical",
                        description=f"File created outside sandbox: {filepath}",
                        affected_resources=[filepath],
                        recommendation="Restrict file operations to sandbox directory"
                    ))
        
        # Check for access to test framework files
        framework_files = ['expected_outcome', 'test_case', 'evaluation']
        for filepath in state_after.file_hashes:
            if any(fw in filepath.lower() for fw in framework_files):
                violations.append(IsolationViolation(
                    violation_type="state_leak",
                    severity="critical",
                    description=f"Access to test framework file: {filepath}",
                    affected_resources=[filepath],
                    recommendation="Isolate test framework from agent access"
                ))
        
        return violations
    
    def _check_env_var_isolation(
        self,
        state_before: EnvironmentState,
        state_after: EnvironmentState
    ) -> List[IsolationViolation]:
        """Check for environment variable isolation violations"""
        violations = []
        
        # Check for leaked credentials
        sensitive_patterns = [
            'KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL',
            'AUTH', 'PRIVATE', 'CERT'
        ]
        
        for var_name, var_value in state_after.environment_vars.items():
            if var_name not in state_before.environment_vars:
                # New environment variable
                if any(pattern in var_name.upper() for pattern in sensitive_patterns):
                    violations.append(IsolationViolation(
                        violation_type="state_leak",
                        severity="critical",
                        description=f"Sensitive environment variable created: {var_name}",
                        affected_resources=[var_name],
                        recommendation="Never store sensitive data in environment variables during tests"
                    ))
        
        # Check for modified system variables
        system_vars = ['PATH', 'HOME', 'USER', 'SHELL', 'LANG']
        for var in system_vars:
            if var in state_before.environment_vars and var in state_after.environment_vars:
                if state_before.environment_vars[var] != state_after.environment_vars[var]:
                    violations.append(IsolationViolation(
                        violation_type="state_leak",
                        severity="high",
                        description=f"System environment variable modified: {var}",
                        affected_resources=[var],
                        recommendation="Avoid modifying system environment variables"
                    ))
        
        return violations
    
    def _check_process_isolation(
        self,
        state_before: EnvironmentState,
        state_after: EnvironmentState
    ) -> List[IsolationViolation]:
        """Check for process isolation violations"""
        violations = []
        
        # Check for lingering processes
        before_procs = set(state_before.running_processes)
        after_procs = set(state_after.running_processes)
        
        new_processes = after_procs - before_procs
        if new_processes:
            violations.append(IsolationViolation(
                violation_type="resource_leak",
                severity="high",
                description=f"Processes not terminated: {', '.join(new_processes)}",
                affected_resources=list(new_processes),
                recommendation="Ensure all spawned processes are terminated"
            ))
        
        # Check for dangerous processes
        dangerous_processes = [
            'rm', 'dd', 'format', 'fdisk', 'mkfs',
            'nc', 'netcat', 'telnet', 'ssh'
        ]
        
        for proc in after_procs:
            if any(danger in proc.lower() for danger in dangerous_processes):
                violations.append(IsolationViolation(
                    violation_type="state_leak",
                    severity="critical",
                    description=f"Dangerous process detected: {proc}",
                    affected_resources=[proc],
                    recommendation="Block execution of potentially harmful processes"
                ))
        
        return violations
    
    def _check_resource_cleanup(
        self,
        state_before: EnvironmentState,
        state_after: EnvironmentState
    ) -> List[IsolationViolation]:
        """Check for resource cleanup violations"""
        violations = []
        
        # Check memory usage
        memory_increase = state_after.memory_usage - state_before.memory_usage
        if memory_increase > 100:  # MB
            violations.append(IsolationViolation(
                violation_type="resource_leak",
                severity="medium",
                description=f"Memory not released: {memory_increase:.1f}MB increase",
                affected_resources=["memory"],
                recommendation="Ensure proper memory cleanup"
            ))
        
        # Check disk usage
        disk_increase = state_after.disk_usage - state_before.disk_usage
        if disk_increase > 50:  # MB
            violations.append(IsolationViolation(
                violation_type="resource_leak",
                severity="medium",
                description=f"Disk space not cleaned: {disk_increase:.1f}MB increase",
                affected_resources=["disk"],
                recommendation="Remove temporary files after test"
            ))
        
        # Check for open ports
        before_ports = set(state_before.open_ports)
        after_ports = set(state_after.open_ports)
        
        lingering_ports = after_ports - before_ports
        if lingering_ports:
            violations.append(IsolationViolation(
                violation_type="resource_leak",
                severity="high",
                description=f"Ports left open: {', '.join(map(str, lingering_ports))}",
                affected_resources=[f"port:{p}" for p in lingering_ports],
                recommendation="Close all opened network connections"
            ))
        
        return violations
    
    def _check_protected_resources(
        self,
        state_after: EnvironmentState,
        test_case: 'TestCase'
    ) -> List[IsolationViolation]:
        """Check for access to protected resources"""
        violations = []
        
        # Check for access to protected files
        for filepath in state_after.file_hashes:
            for protected in self.protected_resources:
                if protected in filepath:
                    violations.append(IsolationViolation(
                        violation_type="state_leak",
                        severity="critical",
                        description=f"Access to protected resource: {filepath}",
                        affected_resources=[filepath],
                        recommendation="Block access to sensitive configuration files"
                    ))
        
        # Check for access to test expectations
        if hasattr(test_case, 'expected_outcome'):
            expectation_str = str(test_case.expected_outcome)
            for filepath, file_hash in state_after.file_hashes.items():
                # This is a simplified check - in practice would check file contents
                if 'expected' in filepath.lower():
                    violations.append(IsolationViolation(
                        violation_type="state_leak",
                        severity="critical",
                        description="Potential access to test expectations",
                        affected_resources=[filepath],
                        recommendation="Ensure test expectations are not accessible to agent"
                    ))
        
        return violations
    
    async def _check_cross_contamination(
        self,
        session_id: str,
        current_state: EnvironmentState
    ) -> List[IsolationViolation]:
        """Check for cross-session contamination"""
        violations = []
        
        # Compare with other session states
        for other_session, states in self.session_states.items():
            if other_session == session_id:
                continue
                
            for other_state in states:
                # Check for shared files
                shared_files = set(current_state.file_hashes.keys()) & set(other_state.file_hashes.keys())
                
                for filepath in shared_files:
                    if current_state.file_hashes[filepath] == other_state.file_hashes[filepath]:
                        # Same file content across sessions might indicate contamination
                        if not self._is_system_file(filepath):
                            violations.append(IsolationViolation(
                                violation_type="cross_contamination",
                                severity="high",
                                description=f"File shared across sessions: {filepath}",
                                affected_resources=[filepath, other_session],
                                recommendation="Ensure complete isolation between test sessions"
                            ))
        
        return violations
    
    def _calculate_isolation_score(self, violations: List[IsolationViolation]) -> float:
        """Calculate overall isolation score"""
        if not violations:
            return 1.0
        
        # Weight by severity
        severity_weights = {
            "critical": 0.4,
            "high": 0.2,
            "medium": 0.1,
            "low": 0.05
        }
        
        total_penalty = 0.0
        for violation in violations:
            total_penalty += severity_weights.get(violation.severity, 0.05)
        
        # Cap at 1.0
        total_penalty = min(total_penalty, 1.0)
        
        return 1.0 - total_penalty
    
    async def _verify_cleanup(
        self,
        session_id: str,
        final_state: EnvironmentState
    ) -> bool:
        """Verify environment was properly cleaned up"""
        try:
            if self.sandbox_manager:
                # Check if session still exists
                session_exists = await self.sandbox_manager.session_exists(session_id)
                if session_exists:
                    logger.warning(f"Session {session_id} still exists after test")
                    return False
                    
            # Check for residual files
            if final_state.file_hashes:
                test_files = [f for f in final_state.file_hashes if not self._is_system_file(f)]
                if test_files:
                    logger.warning(f"Residual files found: {test_files}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error verifying cleanup: {e}")
            return False
    
    async def _capture_file_hashes(self, session_id: str) -> Dict[str, str]:
        """Capture file hashes in the environment"""
        file_hashes = {}
        
        if self.sandbox_manager:
            try:
                # Get list of files
                files = await self.sandbox_manager.list_files(session_id)
                
                for filepath in files:
                    # Calculate hash
                    content = await self.sandbox_manager.read_file(session_id, filepath)
                    if content:
                        file_hash = hashlib.sha256(content.encode()).hexdigest()
                        file_hashes[filepath] = file_hash
                        
            except Exception as e:
                logger.error(f"Error capturing file hashes: {e}")
        
        return file_hashes
    
    async def _capture_env_vars(self, session_id: str) -> Dict[str, str]:
        """Capture environment variables"""
        env_vars = {}
        
        if self.sandbox_manager:
            try:
                # Run env command
                result = await self.sandbox_manager.run_command(session_id, "env")
                if result and result.get('output'):
                    for line in result['output'].split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key] = value
                            
            except Exception as e:
                logger.error(f"Error capturing env vars: {e}")
        
        return env_vars
    
    async def _capture_processes(self, session_id: str) -> List[str]:
        """Capture running processes"""
        processes = []
        
        if self.sandbox_manager:
            try:
                # Run ps command
                result = await self.sandbox_manager.run_command(session_id, "ps aux")
                if result and result.get('output'):
                    lines = result['output'].split('\n')[1:]  # Skip header
                    for line in lines:
                        if line.strip():
                            # Extract process name (simplified)
                            parts = line.split()
                            if len(parts) > 10:
                                processes.append(parts[10])
                                
            except Exception as e:
                logger.error(f"Error capturing processes: {e}")
        
        return processes
    
    async def _capture_resource_usage(self, session_id: str) -> tuple[float, float]:
        """Capture memory and disk usage"""
        memory_usage = 0.0
        disk_usage = 0.0
        
        if self.sandbox_manager:
            try:
                # Get memory usage
                mem_result = await self.sandbox_manager.run_command(
                    session_id, 
                    "free -m | grep Mem | awk '{print $3}'"
                )
                if mem_result and mem_result.get('output'):
                    memory_usage = float(mem_result['output'].strip())
                
                # Get disk usage
                disk_result = await self.sandbox_manager.run_command(
                    session_id,
                    "df -m / | tail -1 | awk '{print $3}'"
                )
                if disk_result and disk_result.get('output'):
                    disk_usage = float(disk_result['output'].strip())
                    
            except Exception as e:
                logger.error(f"Error capturing resource usage: {e}")
        
        return memory_usage, disk_usage
    
    async def _capture_sandbox_metadata(self, session_id: str) -> Dict[str, Any]:
        """Capture sandbox-specific metadata"""
        metadata = {}
        
        if self.sandbox_manager:
            try:
                metadata = await self.sandbox_manager.get_session_metadata(session_id)
            except Exception as e:
                logger.error(f"Error capturing sandbox metadata: {e}")
        
        return metadata
    
    def _is_expected_modification(self, filepath: str, test_case: 'TestCase') -> bool:
        """Check if a file modification is expected for this test"""
        # Repository files are expected to change
        if hasattr(test_case, 'repo_url'):
            repo_name = test_case.repo_url.split('/')[-1].replace('.git', '')
            if repo_name in filepath:
                return True
        
        # Check expected files from test case
        if hasattr(test_case, 'expected_outcome'):
            expected_files = test_case.expected_outcome.get('files_modified', [])
            if any(ef in filepath for ef in expected_files):
                return True
        
        return False
    
    def _get_sandbox_paths(self, test_case: 'TestCase') -> List[str]:
        """Get expected sandbox paths for a test"""
        paths = ['/tmp/', '/workspace/', '/sandbox/']
        
        # Add repo-specific paths
        if hasattr(test_case, 'repo_url'):
            repo_name = test_case.repo_url.split('/')[-1].replace('.git', '')
            paths.append(f'/workspace/{repo_name}/')
            paths.append(f'/tmp/{repo_name}/')
        
        return paths
    
    def _is_system_file(self, filepath: str) -> bool:
        """Check if a file is a system file"""
        system_paths = [
            '/etc/', '/usr/', '/bin/', '/sbin/', '/lib/',
            '/sys/', '/proc/', '/dev/', '/var/lib/'
        ]
        
        return any(filepath.startswith(sp) for sp in system_paths)
    
    def generate_isolation_report(
        self,
        validation_results: List[EnvironmentValidationResult]
    ) -> Dict[str, Any]:
        """Generate comprehensive isolation report"""
        total_tests = len(validation_results)
        clean_tests = sum(1 for r in validation_results if r.is_clean)
        isolated_tests = sum(1 for r in validation_results if r.is_isolated)
        
        # Aggregate violations by type
        violation_counts = {
            "cross_contamination": 0,
            "state_leak": 0,
            "resource_leak": 0
        }
        
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        for result in validation_results:
            for violation in result.violations:
                violation_counts[violation.violation_type] = violation_counts.get(
                    violation.violation_type, 0
                ) + 1
                severity_counts[violation.severity] = severity_counts.get(
                    violation.severity, 0
                ) + 1
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "clean_tests": clean_tests,
                "isolated_tests": isolated_tests,
                "isolation_rate": isolated_tests / total_tests if total_tests > 0 else 0,
                "average_isolation_score": sum(r.isolation_score for r in validation_results) / total_tests if total_tests > 0 else 0
            },
            "violations": {
                "by_type": violation_counts,
                "by_severity": severity_counts,
                "total": sum(violation_counts.values())
            },
            "cleanup": {
                "successful_cleanups": sum(1 for r in validation_results if r.cleanup_successful),
                "cleanup_rate": sum(1 for r in validation_results if r.cleanup_successful) / total_tests if total_tests > 0 else 0
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if severity_counts.get("critical", 0) > 0:
            report["recommendations"].append("URGENT: Fix critical isolation violations immediately")
            
        if report["summary"]["isolation_rate"] < 0.9:
            report["recommendations"].append("Improve test isolation - many tests show contamination")
            
        if report["cleanup"]["cleanup_rate"] < 0.95:
            report["recommendations"].append("Improve cleanup procedures - residual state detected")
        
        return report