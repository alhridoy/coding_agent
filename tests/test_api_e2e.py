"""
End-to-end tests for the autonomous coding agent API
"""

import asyncio
import pytest
import json
import os
from unittest.mock import Mock, patch, AsyncMock
from httpx import AsyncClient
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.models import CodingRequest


class TestAPIEndToEnd:
    """End-to-end tests for the API"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for testing"""
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "GITHUB_TOKEN": "test-github-token",
            "E2B_API_KEY": "test-e2b-key"
        }):
            yield
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "components" in data
    
    def test_root_endpoint(self, client):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "Autonomous Coding Agent" in data["name"]
        assert "endpoints" in data
    
    @pytest.mark.asyncio
    async def test_streaming_api_mock(self, mock_env_vars):
        """Test the streaming API with mocked dependencies"""
        
        # Mock the coding agent and sandbox manager
        with patch('src.api.main.coding_agent') as mock_agent, \
             patch('src.api.main.sandbox_manager') as mock_sandbox:
            
            # Mock sandbox operations
            mock_sandbox.create_session.return_value = "test-session-123"
            mock_sandbox.clone_repository.return_value = {
                "success": True,
                "file_count": 10,
                "repo_name": "test-repo"
            }
            mock_sandbox.cleanup_session.return_value = None
            
            # Mock agent streaming
            async def mock_stream_coding_process(*args, **kwargs):
                from src.api.models import StatusEvent, AIMessageEvent, PRCreatedEvent
                
                yield StatusEvent(message="Starting analysis...")
                yield AIMessageEvent(message="Found Flask app", reasoning="Detected Flask patterns")
                yield PRCreatedEvent(
                    pr_url="https://github.com/test/test-repo/pull/123",
                    pr_number=123,
                    title="Test PR",
                    body="Test PR body"
                )
            
            mock_agent.stream_coding_process.return_value = mock_stream_coding_process()
            
            # Create test request
            request_data = {
                "repo_url": "https://github.com/test/test-repo",
                "prompt": "Add a health check endpoint"
            }
            
            # Make request using httpx AsyncClient
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post("/code", json=request_data)
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
                
                # Check that we get streaming response
                content = response.content.decode()
                assert "data: " in content
                assert "Starting analysis" in content
                assert "Found Flask app" in content
                assert "github.com" in content
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_request(self):
        """Test error handling for invalid requests"""
        
        # Test missing required fields
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/code", json={})
            assert response.status_code == 422  # Validation error
            
            # Test invalid URL
            response = await client.post("/code", json={
                "repo_url": "not-a-url",
                "prompt": "Test prompt"
            })
            assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_env_vars):
        """Test handling multiple concurrent requests"""
        
        with patch('src.api.main.coding_agent') as mock_agent, \
             patch('src.api.main.sandbox_manager') as mock_sandbox:
            
            # Mock quick responses
            mock_sandbox.create_session.return_value = "test-session"
            mock_sandbox.clone_repository.return_value = {"success": True, "file_count": 5}
            mock_sandbox.cleanup_session.return_value = None
            
            async def quick_mock_stream(*args, **kwargs):
                from src.api.models import StatusEvent
                yield StatusEvent(message="Quick test")
            
            mock_agent.stream_coding_process.return_value = quick_mock_stream()
            
            # Create multiple concurrent requests
            request_data = {
                "repo_url": "https://github.com/test/test-repo",
                "prompt": "Test prompt"
            }
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                # Make 3 concurrent requests
                tasks = [
                    client.post("/code", json=request_data),
                    client.post("/code", json=request_data),
                    client.post("/code", json=request_data)
                ]
                
                responses = await asyncio.gather(*tasks)
                
                # All should succeed
                for response in responses:
                    assert response.status_code == 200
                    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


class TestEventStreaming:
    """Test event streaming functionality"""
    
    @pytest.mark.asyncio
    async def test_stream_event_formatting(self):
        """Test that events are properly formatted for SSE"""
        
        from src.api.main import stream_event
        from src.api.models import (
            StatusEvent, AIMessageEvent, ToolCallEvent, 
            GitOperationEvent, PRCreatedEvent, ErrorEvent
        )
        
        # Test status event
        status_event = StatusEvent(message="Testing status")
        formatted = await stream_event(status_event)
        assert formatted == 'data: {"type": "Status", "message": "Testing status"}\n\n'
        
        # Test AI message event
        ai_event = AIMessageEvent(message="AI thinking", reasoning="Logic")
        formatted = await stream_event(ai_event)
        assert 'data: {"type": "AI Message"' in formatted
        assert '"message": "AI thinking"' in formatted
        
        # Test tool call event - read file
        tool_event = ToolCallEvent(
            tool_name="read_file",
            tool_input={"file": "app.py"},
            tool_output="File read"
        )
        formatted = await stream_event(tool_event)
        assert 'data: {"type": "Tool: Read"' in formatted
        assert '"filepath": "app.py"' in formatted
        
        # Test tool call event - write file
        write_event = ToolCallEvent(
            tool_name="write_file",
            tool_input={"file": "test.py", "old_str": "old", "new_str": "new"},
            tool_output="File written"
        )
        formatted = await stream_event(write_event)
        assert 'data: {"type": "Tool: Edit"' in formatted
        assert '"filepath": "test.py"' in formatted
        
        # Test git operation event
        git_event = GitOperationEvent(
            command="git commit -m 'test'",
            output="Commit successful",
            success=True
        )
        formatted = await stream_event(git_event)
        assert 'data: {"type": "Tool: Bash"' in formatted
        assert '"command": "git commit -m \'test\'"' in formatted
        
        # Test PR created event
        pr_event = PRCreatedEvent(
            pr_url="https://github.com/test/repo/pull/123",
            pr_number=123,
            title="Test PR",
            body="Test body"
        )
        formatted = await stream_event(pr_event)
        assert 'data: {' in formatted
        assert '"pr_url": "https://github.com/test/repo/pull/123"' in formatted
        
        # Test error event
        error_event = ErrorEvent(
            error_type="TestError",
            error_message="Test error message"
        )
        formatted = await stream_event(error_event)
        assert 'data: {' in formatted
        assert '"error_type": "TestError"' in formatted


class TestWorkflowIntegration:
    """Test the complete workflow integration"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_simulation(self, mock_env_vars):
        """Simulate a complete workflow with all components mocked"""
        
        from src.api.main import coding_process_stream
        
        # Mock all dependencies
        with patch('src.api.main.sandbox_manager') as mock_sandbox, \
             patch('src.api.main.coding_agent') as mock_agent:
            
            # Mock sandbox operations
            mock_sandbox.create_session.return_value = "session-123"
            mock_sandbox.clone_repository.return_value = {
                "success": True,
                "file_count": 15,
                "repo_name": "test-repo"
            }
            mock_sandbox.cleanup_session.return_value = None
            
            # Mock agent workflow
            async def mock_workflow(*args, **kwargs):
                from src.api.models import (
                    StatusEvent, AIMessageEvent, ToolCallEvent,
                    GitOperationEvent, PRCreatedEvent
                )
                
                yield StatusEvent(message="Analyzing repository...")
                yield AIMessageEvent(message="Found Python Flask app", reasoning="Detected Flask patterns")
                yield ToolCallEvent(
                    tool_name="read_file",
                    tool_input={"file": "app.py"},
                    tool_output="File read successfully"
                )
                yield ToolCallEvent(
                    tool_name="write_file",
                    tool_input={"file": "app.py", "old_str": "old_code", "new_str": "new_code"},
                    tool_output="File modified"
                )
                yield GitOperationEvent(
                    command="git checkout -b feature/health-check",
                    output="Switched to branch",
                    success=True
                )
                yield GitOperationEvent(
                    command="git commit -m 'Add health check'",
                    output="Commit created",
                    success=True
                )
                yield PRCreatedEvent(
                    pr_url="https://github.com/test/test-repo/pull/456",
                    pr_number=456,
                    title="Add health check endpoint",
                    body="Added health check endpoint as requested"
                )
            
            mock_agent.stream_coding_process.return_value = mock_workflow()
            
            # Execute the workflow
            events = []
            async for event_str in coding_process_stream(
                repo_url="https://github.com/test/test-repo",
                prompt="Add a health check endpoint"
            ):
                events.append(event_str)
            
            # Verify we got all expected events
            event_content = "".join(events)
            
            # Check for key workflow stages
            assert "Initializing coding session" in event_content
            assert "Cloning repository" in event_content
            assert "Analyzing repository" in event_content
            assert "Starting AI coding agent" in event_content
            assert "Tool: Read" in event_content
            assert "Tool: Edit" in event_content
            assert "Tool: Bash" in event_content
            assert "github.com" in event_content
            assert "Session cleaned up" in event_content
            
            # Verify session was cleaned up
            mock_sandbox.cleanup_session.assert_called_once_with("session-123")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])