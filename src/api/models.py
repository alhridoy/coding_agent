from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Literal, Dict, Any
from datetime import datetime


class CodingRequest(BaseModel):
    """Request model for the coding endpoint"""
    repo_url: HttpUrl = Field(..., description="Public GitHub repository URL")
    prompt: str = Field(..., min_length=10, max_length=2000, description="Coding task description")
    branch_name: Optional[str] = Field(None, description="Custom branch name (auto-generated if not provided)")
    pr_title: Optional[str] = Field(None, description="Custom PR title (auto-generated if not provided)")


class StreamEvent(BaseModel):
    """Base model for streaming events"""
    type: Literal[
        "status", 
        "tool_call", 
        "ai_message", 
        "git_operation", 
        "pr_created", 
        "error",
        "completed"
    ]
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)


class StatusEvent(StreamEvent):
    """Status update event"""
    type: Literal["status"] = "status"
    message: str


class ToolCallEvent(StreamEvent):
    """Tool execution event"""
    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Optional[str] = None


class AIMessageEvent(StreamEvent):
    """AI reasoning/message event"""
    type: Literal["ai_message"] = "ai_message"
    message: str
    reasoning: Optional[str] = None


class GitOperationEvent(StreamEvent):
    """Git operation event"""
    type: Literal["git_operation"] = "git_operation"
    command: str
    output: str
    success: bool = True


class PRCreatedEvent(StreamEvent):
    """PR creation event"""
    type: Literal["pr_created"] = "pr_created"
    pr_url: str
    pr_number: int
    title: str
    body: str


class ErrorEvent(StreamEvent):
    """Error event"""
    type: Literal["error"] = "error"
    error_type: str
    error_message: str
    traceback: Optional[str] = None


class CompletedEvent(StreamEvent):
    """Completion event"""
    type: Literal["completed"] = "completed"
    pr_url: str
    summary: str
    changes_made: list[str]
