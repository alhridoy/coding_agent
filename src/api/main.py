from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
import logging
from typing import AsyncGenerator
import traceback

from .models import CodingRequest, StreamEvent, StatusEvent, ErrorEvent, CompletedEvent
from ..agent.coding_agent import CodingAgent
from ..sandbox.e2b_sandbox import E2BSandbox
from ..utils.config import get_settings

# Configure detailed logging for tracing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
coding_agent = None
sandbox_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup global resources"""
    global coding_agent, sandbox_manager
    
    settings = get_settings()
    
    # Initialize components with observability
    coding_agent = CodingAgent(
        anthropic_api_key=settings.anthropic_api_key,
        github_token=settings.github_token
    )
    
    # Initialize E2B sandbox
    try:
        if not settings.e2b_api_key:
            raise ValueError("E2B_API_KEY is required for sandbox operations")
        
        sandbox_manager = E2BSandbox()
        logger.info("🚀 API: Using E2BSandbox for secure code execution")

    except Exception as e:
        logger.error(f"Failed to initialize E2B sandbox: {e}")
        raise Exception("E2B sandbox initialization failed. Please check your E2B_API_KEY.")
    
    logger.info("Application started successfully")
    yield
    
    # Cleanup
    if sandbox_manager:
        await sandbox_manager.cleanup_all()
    
    logger.info("Application shutting down")


# Create FastAPI app
app = FastAPI(
    title="Autonomous Coding Agent",
    description="A service that creates pull requests from natural language coding prompts",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def stream_event(event: StreamEvent) -> str:
    """Format event as Server-Sent Event in Backspace specification format"""
    
    
    if event.type == "status":
        return f'data: {json.dumps({"type": "Status", "message": event.message})}\n\n'
    elif event.type == "ai_message":
        return f'data: {json.dumps({"type": "AI Message", "message": event.message})}\n\n'
    elif event.type == "tool_call":
        if event.tool_name == "read_file":
            return f'data: {json.dumps({"type": "Tool: Read", "filepath": event.tool_input.get("file", "")})}\n\n'
        elif event.tool_name in ["write_file", "apply_change"]:
            filepath = event.tool_input.get("file", "")
            old_str = event.tool_input.get("old_str", "")
            new_str = event.tool_input.get("new_str", "")
            return f'data: {json.dumps({"type": "Tool: Edit", "filepath": filepath, "old_str": old_str, "new_str": new_str})}\n\n'
        else:
            return f'data: {json.dumps({"type": "AI Message", "message": f"Using tool: {event.tool_name}"})}\n\n'
    elif event.type == "git_operation":
        return f'data: {json.dumps({"type": "Tool: Bash", "command": event.command, "output": event.output})}\n\n'
    else:
        # Fallback to original format
        return f"data: {event.model_dump_json()}\n\n"


async def coding_process_stream(
    repo_url: str, 
    prompt: str, 
    branch_name: str = None,
    pr_title: str = None
) -> AsyncGenerator[str, None]:
    """Main streaming function that orchestrates the coding process"""
    session_id = None
    
    try:
        # Step 1: Initialize session
        yield await stream_event(StatusEvent(message="Initializing coding session..."))
        
        session_id = await sandbox_manager.create_session()
        
        # Step 2: Clone repository
        yield await stream_event(StatusEvent(message=f"Cloning repository: {repo_url}"))
        
        repo_info = await sandbox_manager.clone_repository(session_id, str(repo_url))
        
        yield await stream_event(StatusEvent(
            message=f"Repository cloned successfully. Found {repo_info.get('file_count', 0)} files"
        ))
        
        # Step 3: Analyze repository
        yield await stream_event(StatusEvent(message="Analyzing repository structure..."))
        
        # Step 4: Execute coding agent
        yield await stream_event(StatusEvent(message="Starting AI coding agent..."))
        
        async for event in coding_agent.stream_coding_process(
            session_id=session_id,
            repo_info=repo_info,
            prompt=prompt,
            branch_name=branch_name,
            pr_title=pr_title,
            sandbox_manager=sandbox_manager
        ):
            yield await stream_event(event)
        
        # Step 5: Completion - will be handled by individual PR creation events
        
    except Exception as e:
        logger.error(f"Error in coding process: {str(e)}", exc_info=True)
        yield await stream_event(ErrorEvent(
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=traceback.format_exc()
        ))
        
    finally:
        # Cleanup sandbox session
        if session_id and sandbox_manager:
            try:
                await sandbox_manager.cleanup_session(session_id)
                yield await stream_event(StatusEvent(message="Session cleaned up successfully"))
            except Exception as e:
                logger.error(f"Error cleaning up session: {str(e)}")


@app.post("/code")
async def stream_coding_process(request: CodingRequest):
    """
    Stream the coding process in real-time via Server-Sent Events.
    
    Takes a GitHub repository URL and a coding prompt, then:
    1. Clones the repo in a secure sandbox
    2. Runs an AI coding agent to implement changes
    3. Creates a pull request with the changes
    4. Streams the entire process in real-time
    """
    try:
        return StreamingResponse(
            coding_process_stream(
                repo_url=str(request.repo_url),
                prompt=request.prompt,
                branch_name=request.branch_name,
                pr_title=request.pr_title
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    except Exception as e:
        logger.error(f"Error starting coding process: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start coding process: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "coding_agent": coding_agent is not None,
            "sandbox_manager": sandbox_manager is not None
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Autonomous Coding Agent API",
        "version": "0.1.0",
        "description": "Stream coding changes and create PRs from natural language prompts",
        "endpoints": {
            "POST /code": "Stream coding process",
            "GET /health": "Health check",
            "GET /": "This endpoint"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
