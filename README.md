# Autonomous Coding Agent

A sandboxed coding agent that creates pull requests automatically. This service takes a GitHub repository URL and coding prompt, then autonomously implements the requested changes and creates a pull request with real-time streaming updates.


## ⚡ Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd autonomous-coding-agent
pip install -e .

# 2. Set environment variables in .env
ANTHROPIC_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here

# 3. Run the server
python -m src.api.main

# 4. Test the API
curl -X POST "http://localhost:8000/code" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/owner/repo", "prompt": "Add input validation"}'

# 5. Or run the demo
python demo.py
```

## 🚀 Features

- **Streaming API**: Real-time updates via Server-Sent Events (matches Backspace spec exactly)
- **Secure Sandboxing**: Isolated environment for code execution
- **Claude Sonnet 4**: Latest AI model for intelligent code analysis and modification
- **LangGraph Architecture**: Production-ready StateGraph workflow orchestration
- **GitHub Integration**: Automatic PR creation with multiple fallback methods
- **Real-time Observability**: LangSmith tracing for full workflow visibility

## 🏗️ Architecture

- **FastAPI Backend**: Streaming API with SSE support matching Backspace specification
- **LangGraph StateGraph**: Advanced workflow orchestration with retry policies
- **E2B Sandbox**: Secure cloud-based isolated environment for repository operations
- **Claude Sonnet 4**: Latest model (claude-sonnet-4-20250514) for code generation
- **Multi-tier GitHub Integration**: API + CLI + manual fallbacks for reliable PR creation
- **LangSmith Observability**: Complete tracing and monitoring

## 🛠️ Setup

### Prerequisites

- Python 3.9+
- Git
- GitHub CLI (optional, for PR creation)
- GitHub Personal Access Token

---

