"""
Tiered datasets for comprehensive evaluation of autonomous coding agents
"""

import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .eval_framework import TestCase


@dataclass
class DatasetMetadata:
    """Metadata for evaluation datasets"""
    name: str
    description: str
    difficulty: str
    expected_time_minutes: int
    success_rate_threshold: float
    categories: List[str]


class TieredDatasets:
    """Provides tiered datasets for systematic evaluation"""
    
    def __init__(self):
        self.datasets = {
            "tier1_basic": self._get_tier1_basic(),
            "tier2_intermediate": self._get_tier2_intermediate(), 
            "tier3_advanced": self._get_tier3_advanced(),
            "tier4_expert": self._get_tier4_expert(),
            "performance": self._get_performance_tests(),
            "edge_cases": self._get_edge_cases(),
            "security": self._get_security_tests()
        }
        
    def get_dataset(self, tier: str) -> List[TestCase]:
        """Get test cases for a specific tier"""
        return self.datasets.get(tier, [])
        
    def get_all_datasets(self) -> Dict[str, List[TestCase]]:
        """Get all available datasets"""
        return self.datasets
        
    def get_random_sample(self, tier: str, count: int = 5) -> List[TestCase]:
        """Get random sample from a tier"""
        dataset = self.get_dataset(tier)
        return random.sample(dataset, min(count, len(dataset)))
        
    def get_progressive_evaluation(self) -> List[TestCase]:
        """Get progressive evaluation starting from basic to advanced"""
        progressive_tests = []
        
        # Start with basic tests
        progressive_tests.extend(self.get_random_sample("tier1_basic", 3))
        progressive_tests.extend(self.get_random_sample("tier2_intermediate", 2))
        progressive_tests.extend(self.get_random_sample("tier3_advanced", 1))
        
        return progressive_tests
        
    def _get_tier1_basic(self) -> List[TestCase]:
        """Tier 1: Basic functionality tests"""
        
        return [
            TestCase(
                id="basic_readme_enhancement",
                name="README Enhancement",
                description="Add basic project information to README",
                repo_url="https://github.com/octocat/Hello-World",
                prompt="Add a project description, installation instructions, and usage example to the README",
                expected_outcome={
                    "pr_created": True,
                    "readme_updated": True,
                    "sections_added": ["description", "installation", "usage"]
                },
                timeout_seconds=300,
                tags=["documentation", "readme", "basic"],
                difficulty="easy",
                category="documentation"
            ),
            
            TestCase(
                id="simple_comment_addition",
                name="Add Code Comments",
                description="Add explanatory comments to code",
                repo_url="https://github.com/octocat/Hello-World",
                prompt="Add helpful comments explaining what the code does",
                expected_outcome={
                    "pr_created": True,
                    "comments_added": True
                },
                timeout_seconds=180,
                tags=["comments", "documentation", "basic"],
                difficulty="easy",
                category="code_quality"
            ),
            
            TestCase(
                id="license_addition",
                name="Add License File",
                description="Add MIT license to repository",
                repo_url="https://github.com/octocat/Hello-World",
                prompt="Add an MIT license file to the repository",
                expected_outcome={
                    "pr_created": True,
                    "license_added": True,
                    "proper_format": True
                },
                timeout_seconds=240,
                tags=["license", "legal", "basic"],
                difficulty="easy",
                category="documentation"
            ),
            
            TestCase(
                id="gitignore_creation",
                name="Create .gitignore",
                description="Add appropriate .gitignore file",
                repo_url="https://github.com/octocat/Hello-World", 
                prompt="Create a .gitignore file with common patterns for this type of project",
                expected_outcome={
                    "pr_created": True,
                    "gitignore_added": True,
                    "appropriate_patterns": True
                },
                timeout_seconds=200,
                tags=["gitignore", "configuration", "basic"],
                difficulty="easy",
                category="configuration"
            ),
            
            TestCase(
                id="basic_script_fix",
                name="Fix Simple Script Error",
                description="Fix a basic syntax or logic error",
                repo_url="https://github.com/python/cpython",
                prompt="Fix any simple syntax errors or typos in Python scripts",
                expected_outcome={
                    "pr_created": True,
                    "errors_fixed": True,
                    "syntax_valid": True
                },
                timeout_seconds=400,
                tags=["bugfix", "python", "basic"],
                difficulty="easy",
                category="bugfix"
            )
        ]
        
    def _get_tier2_intermediate(self) -> List[TestCase]:
        """Tier 2: Intermediate functionality tests"""
        
        return [
            TestCase(
                id="api_health_endpoint",
                name="Add API Health Check",
                description="Add health check endpoint to API",
                repo_url="https://github.com/fastapi/fastapi",
                prompt="Add a /health endpoint that returns API status, version, and uptime",
                expected_outcome={
                    "pr_created": True,
                    "endpoint_added": True,
                    "returns_json": True,
                    "includes_version": True
                },
                timeout_seconds=600,
                tags=["api", "health", "endpoint", "intermediate"],
                difficulty="medium",
                category="backend"
            ),
            
            TestCase(
                id="input_validation_middleware",
                name="Add Input Validation",
                description="Add input validation to API endpoints",
                repo_url="https://github.com/express/express",
                prompt="Add input validation middleware for all POST and PUT endpoints",
                expected_outcome={
                    "pr_created": True,
                    "validation_added": True,
                    "middleware_implemented": True,
                    "error_responses": True
                },
                timeout_seconds=800,
                tags=["validation", "middleware", "api", "intermediate"],
                difficulty="medium",
                category="backend"
            ),
            
            TestCase(
                id="logging_implementation",
                name="Add Structured Logging",
                description="Implement structured logging system",
                repo_url="https://github.com/winstonjs/winston",
                prompt="Add structured logging with different log levels and request tracking",
                expected_outcome={
                    "pr_created": True,
                    "logging_added": True,
                    "structured_format": True,
                    "multiple_levels": True
                },
                timeout_seconds=700,
                tags=["logging", "monitoring", "intermediate"],
                difficulty="medium",
                category="observability"
            ),
            
            TestCase(
                id="error_handling_enhancement",
                name="Enhance Error Handling",
                description="Add comprehensive error handling",
                repo_url="https://github.com/pallets/flask",
                prompt="Add comprehensive error handling with custom error pages and logging",
                expected_outcome={
                    "pr_created": True,
                    "error_handlers_added": True,
                    "custom_pages": True,
                    "error_logging": True
                },
                timeout_seconds=650,
                tags=["error-handling", "web", "intermediate"],
                difficulty="medium",
                category="reliability"
            ),
            
            TestCase(
                id="configuration_management",
                name="Add Configuration Management",
                description="Implement environment-based configuration",
                repo_url="https://github.com/kelektiv/node.bcrypt.js",
                prompt="Add configuration management with environment variables and validation",
                expected_outcome={
                    "pr_created": True,
                    "config_system": True,
                    "env_support": True,
                    "validation": True
                },
                timeout_seconds=750,
                tags=["configuration", "environment", "intermediate"],
                difficulty="medium",
                category="configuration"
            )
        ]
        
    def _get_tier3_advanced(self) -> List[TestCase]:
        """Tier 3: Advanced functionality tests"""
        
        return [
            TestCase(
                id="database_integration",
                name="Database Integration",
                description="Add database connection and basic operations",
                repo_url="https://github.com/sqlalchemy/sqlalchemy",
                prompt="Add database integration with connection pooling, migrations, and basic CRUD operations",
                expected_outcome={
                    "pr_created": True,
                    "db_integration": True,
                    "connection_pooling": True,
                    "migrations": True,
                    "crud_operations": True
                },
                timeout_seconds=1200,
                tags=["database", "orm", "advanced"],
                difficulty="hard",
                category="backend"
            ),
            
            TestCase(
                id="authentication_system",
                name="Authentication System",
                description="Implement JWT-based authentication",
                repo_url="https://github.com/auth0/node-jsonwebtoken",
                prompt="Implement JWT-based authentication with login, logout, and protected routes",
                expected_outcome={
                    "pr_created": True,
                    "auth_system": True,
                    "jwt_implementation": True,
                    "protected_routes": True,
                    "secure_storage": True
                },
                timeout_seconds=1500,
                tags=["authentication", "jwt", "security", "advanced"],
                difficulty="hard",
                category="security"
            ),
            
            TestCase(
                id="caching_layer",
                name="Implement Caching Layer",
                description="Add Redis-based caching system",
                repo_url="https://github.com/redis/redis",
                prompt="Implement a Redis-based caching layer with TTL, invalidation, and cache-aside pattern",
                expected_outcome={
                    "pr_created": True,
                    "caching_implemented": True,
                    "redis_integration": True,
                    "ttl_support": True,
                    "invalidation": True
                },
                timeout_seconds=1300,
                tags=["caching", "redis", "performance", "advanced"],
                difficulty="hard",
                category="performance"
            ),
            
            TestCase(
                id="api_rate_limiting",
                name="API Rate Limiting",
                description="Implement sophisticated rate limiting",
                repo_url="https://github.com/animir/node-rate-limiter-flexible",
                prompt="Implement rate limiting with sliding window, different limits per user type, and bypass mechanisms",
                expected_outcome={
                    "pr_created": True,
                    "rate_limiting": True,
                    "sliding_window": True,
                    "user_tiers": True,
                    "bypass_mechanism": True
                },
                timeout_seconds=1100,
                tags=["rate-limiting", "api", "security", "advanced"],
                difficulty="hard",
                category="security"
            ),
            
            TestCase(
                id="microservice_communication",
                name="Microservice Communication",
                description="Implement service-to-service communication",
                repo_url="https://github.com/grpc/grpc",
                prompt="Implement gRPC-based microservice communication with service discovery and health checks",
                expected_outcome={
                    "pr_created": True,
                    "grpc_implementation": True,
                    "service_discovery": True,
                    "health_checks": True,
                    "error_handling": True
                },
                timeout_seconds=1600,
                tags=["microservices", "grpc", "communication", "advanced"],
                difficulty="hard",
                category="architecture"
            )
        ]
        
    def _get_tier4_expert(self) -> List[TestCase]:
        """Tier 4: Expert-level tests"""
        
        return [
            TestCase(
                id="distributed_tracing",
                name="Distributed Tracing Implementation",
                description="Implement OpenTelemetry distributed tracing",
                repo_url="https://github.com/open-telemetry/opentelemetry-js",
                prompt="Implement distributed tracing with OpenTelemetry, including span propagation, sampling, and export to multiple backends",
                expected_outcome={
                    "pr_created": True,
                    "tracing_implemented": True,
                    "span_propagation": True,
                    "sampling_strategy": True,
                    "multiple_exporters": True
                },
                timeout_seconds=2000,
                tags=["tracing", "observability", "expert"],
                difficulty="expert",
                category="observability"
            ),
            
            TestCase(
                id="advanced_security_framework",
                name="Advanced Security Framework",
                description="Implement comprehensive security framework",
                repo_url="https://github.com/OWASP/CheatSheetSeries",
                prompt="Implement a comprehensive security framework with RBAC, audit logging, encryption at rest, and threat detection",
                expected_outcome={
                    "pr_created": True,
                    "rbac_system": True,
                    "audit_logging": True,
                    "encryption": True,
                    "threat_detection": True
                },
                timeout_seconds=2500,
                tags=["security", "rbac", "encryption", "expert"],
                difficulty="expert",
                category="security"
            ),
            
            TestCase(
                id="event_sourcing_system",
                name="Event Sourcing Implementation",
                description="Implement event sourcing with CQRS pattern",
                repo_url="https://github.com/EventStore/EventStore",
                prompt="Implement event sourcing with CQRS, event store, snapshots, and projections",
                expected_outcome={
                    "pr_created": True,
                    "event_sourcing": True,
                    "cqrs_pattern": True,
                    "event_store": True,
                    "snapshots": True,
                    "projections": True
                },
                timeout_seconds=2200,
                tags=["event-sourcing", "cqrs", "architecture", "expert"],
                difficulty="expert",
                category="architecture"
            )
        ]
        
    def _get_performance_tests(self) -> List[TestCase]:
        """Performance-focused test cases"""
        
        return [
            TestCase(
                id="perf_simple_change",
                name="Performance: Simple Change",
                description="Measure performance for simple changes",
                repo_url="https://github.com/octocat/Hello-World",
                prompt="Add a single line comment to the main file",
                expected_outcome={
                    "execution_time_under": 60,
                    "pr_created": True
                },
                timeout_seconds=120,
                tags=["performance", "speed"],
                difficulty="easy",
                category="performance"
            ),
            
            TestCase(
                id="perf_medium_change",
                name="Performance: Medium Change",
                description="Measure performance for medium complexity changes",
                repo_url="https://github.com/fastapi/fastapi",
                prompt="Add a new API endpoint with documentation",
                expected_outcome={
                    "execution_time_under": 300,
                    "pr_created": True
                },
                timeout_seconds=600,
                tags=["performance", "api"],
                difficulty="medium",
                category="performance"
            ),
            
            TestCase(
                id="perf_concurrent_load",
                name="Performance: Concurrent Load",
                description="Test performance under concurrent load",
                repo_url="https://github.com/octocat/Hello-World",
                prompt="Add project badges to README",
                expected_outcome={
                    "handles_concurrency": True,
                    "no_conflicts": True
                },
                timeout_seconds=400,
                tags=["performance", "concurrency"],
                difficulty="medium",
                category="performance"
            )
        ]
        
    def _get_edge_cases(self) -> List[TestCase]:
        """Edge case test scenarios"""
        
        return [
            TestCase(
                id="edge_empty_repo",
                name="Edge Case: Empty Repository",
                description="Handle completely empty repository",
                repo_url="https://github.com/octocat/Hello-World",
                prompt="Initialize a basic project structure in an empty repository",
                expected_outcome={
                    "pr_created": True,
                    "structure_created": True
                },
                timeout_seconds=400,
                tags=["edge-case", "empty-repo"],
                difficulty="medium",
                category="edge_cases"
            ),
            
            TestCase(
                id="edge_large_files",
                name="Edge Case: Large Files",
                description="Handle repository with large files",
                repo_url="https://github.com/git-lfs/git-lfs",
                prompt="Add documentation about handling large files",
                expected_outcome={
                    "pr_created": True,
                    "handles_large_files": True
                },
                timeout_seconds=800,
                tags=["edge-case", "large-files"],
                difficulty="medium",
                category="edge_cases"
            ),
            
            TestCase(
                id="edge_complex_structure",
                name="Edge Case: Complex Repository Structure",
                description="Handle repository with complex nested structure",
                repo_url="https://github.com/kubernetes/kubernetes",
                prompt="Add a simple improvement to the documentation",
                expected_outcome={
                    "pr_created": True,
                    "navigates_structure": True
                },
                timeout_seconds=900,
                tags=["edge-case", "complex-structure"],
                difficulty="hard",
                category="edge_cases"
            )
        ]
        
    def _get_security_tests(self) -> List[TestCase]:
        """Security-focused test cases"""
        
        return [
            TestCase(
                id="security_input_sanitization",
                name="Security: Input Sanitization",
                description="Add input sanitization to prevent XSS",
                repo_url="https://github.com/cure53/DOMPurify",
                prompt="Add input sanitization to prevent XSS attacks in user-generated content",
                expected_outcome={
                    "pr_created": True,
                    "sanitization_added": True,
                    "xss_prevention": True
                },
                timeout_seconds=700,
                tags=["security", "xss", "sanitization"],
                difficulty="medium",
                category="security"
            ),
            
            TestCase(
                id="security_sql_injection_prevention",
                name="Security: SQL Injection Prevention",
                description="Add parameterized queries to prevent SQL injection",
                repo_url="https://github.com/mysqljs/mysql",
                prompt="Replace string concatenation with parameterized queries to prevent SQL injection",
                expected_outcome={
                    "pr_created": True,
                    "parameterized_queries": True,
                    "sql_injection_prevention": True
                },
                timeout_seconds=800,
                tags=["security", "sql-injection", "database"],
                difficulty="medium",
                category="security"
            ),
            
            TestCase(
                id="security_secrets_management",
                name="Security: Secrets Management",
                description="Implement secure secrets management",
                repo_url="https://github.com/hashicorp/vault",
                prompt="Implement secure secrets management with encryption and rotation",
                expected_outcome={
                    "pr_created": True,
                    "secrets_encrypted": True,
                    "rotation_support": True,
                    "no_hardcoded_secrets": True
                },
                timeout_seconds=1200,
                tags=["security", "secrets", "encryption"],
                difficulty="hard",
                category="security"
            )
        ]
        
    def get_dataset_metadata(self, tier: str) -> Optional[DatasetMetadata]:
        """Get metadata for a specific dataset"""
        
        metadata_map = {
            "tier1_basic": DatasetMetadata(
                name="Basic Functionality",
                description="Simple documentation and configuration tasks",
                difficulty="easy",
                expected_time_minutes=5,
                success_rate_threshold=0.95,
                categories=["documentation", "configuration"]
            ),
            "tier2_intermediate": DatasetMetadata(
                name="Intermediate Development",
                description="API endpoints, middleware, and system integration",
                difficulty="medium", 
                expected_time_minutes=15,
                success_rate_threshold=0.85,
                categories=["backend", "api", "middleware"]
            ),
            "tier3_advanced": DatasetMetadata(
                name="Advanced Architecture",
                description="Database integration, authentication, and complex systems",
                difficulty="hard",
                expected_time_minutes=30,
                success_rate_threshold=0.70,
                categories=["architecture", "database", "security"]
            ),
            "tier4_expert": DatasetMetadata(
                name="Expert Systems",
                description="Distributed systems, advanced security, and complex patterns",
                difficulty="expert",
                expected_time_minutes=60,
                success_rate_threshold=0.50,
                categories=["distributed", "architecture", "security"]
            ),
            "performance": DatasetMetadata(
                name="Performance Tests",
                description="Speed and efficiency benchmarks",
                difficulty="varies",
                expected_time_minutes=10,
                success_rate_threshold=0.90,
                categories=["performance", "benchmarks"]
            ),
            "security": DatasetMetadata(
                name="Security Tests", 
                description="Security-focused implementation tasks",
                difficulty="medium-hard",
                expected_time_minutes=25,
                success_rate_threshold=0.80,
                categories=["security", "validation", "encryption"]
            )
        }
        
        return metadata_map.get(tier)
        
    def export_datasets(self, filepath: str):
        """Export all datasets to JSON file"""
        
        export_data = {
            "datasets": {},
            "metadata": {}
        }
        
        for tier, test_cases in self.datasets.items():
            export_data["datasets"][tier] = [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "description": tc.description,
                    "repo_url": tc.repo_url,
                    "prompt": tc.prompt,
                    "expected_outcome": tc.expected_outcome,
                    "timeout_seconds": tc.timeout_seconds,
                    "tags": tc.tags,
                    "difficulty": tc.difficulty,
                    "category": tc.category
                }
                for tc in test_cases
            ]
            
            metadata = self.get_dataset_metadata(tier)
            if metadata:
                export_data["metadata"][tier] = {
                    "name": metadata.name,
                    "description": metadata.description,
                    "difficulty": metadata.difficulty,
                    "expected_time_minutes": metadata.expected_time_minutes,
                    "success_rate_threshold": metadata.success_rate_threshold,
                    "categories": metadata.categories
                }
                
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
    def get_recommended_progression(self, current_success_rate: float) -> str:
        """Get recommended next tier based on current performance"""
        
        if current_success_rate >= 0.95:
            return "tier2_intermediate"
        elif current_success_rate >= 0.85:
            return "tier3_advanced"
        elif current_success_rate >= 0.70:
            return "tier4_expert"
        else:
            return "tier1_basic"  # Go back to basics