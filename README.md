# Backspace Autonomous Coding Agent

A production-ready sandboxed coding agent that creates pull requests automatically. This service takes a GitHub repository URL and coding prompt, then autonomously implements the requested changes and creates a pull request with real-time streaming updates.

**Built for the Backspace coding agent takehome challenge.**

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

### Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd autonomous-coding-agent
```

2. **Install dependencies:**
```bash
pip install -e .
```

3. **Set up environment variables:**
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GITHUB_TOKEN=your_github_personal_access_token
E2B_API_KEY=your_e2b_api_key_here
```

### Getting API Keys

1. **Anthropic API Key**: 
   - Go to [Anthropic Console](https://console.anthropic.com/)
   - Create an API key

2. **GitHub Token**:
   - Go to [GitHub Settings > Personal Access Tokens](https://github.com/settings/tokens)
   - Create a classic token with `repo` scope

3. **E2B API Key** (optional):
   - Go to [E2B Dashboard](https://e2b.dev/)
   - Create an API key

## 🚀 Running the Application

### Local Development

```bash
# Start the server
python -m src.api.main

# The server will be available at http://localhost:8000
```

### Testing the API

```bash
# Example API call
curl -X POST "http://localhost:8000/code" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/yourusername/your-repo",
    "prompt": "Add input validation to all POST endpoints",
    "branch": "feature/add-validation"
  }'
```

## 📡 API Reference

### POST /code

Creates a new coding session and streams the implementation process.

**Request Body:**
```json
{
  "repo_url": "https://github.com/owner/repo",
  "prompt": "Description of changes to make",
  "branch": "feature/branch-name"
}
```

**Response:**
Server-Sent Events stream with real-time updates:

```
data: {"type": "Status", "message": "Cloning repository..."}
data: {"type": "Tool: Read", "filepath": "app.py"}
data: {"type": "AI Message", "message": "Found 3 POST endpoints..."}
data: {"type": "Tool: Edit", "filepath": "models.py", "old_str": "", "new_str": "class UserModel(BaseModel):..."}
data: {"type": "Tool: Bash", "command": "git checkout -b feature/validation", "output": "Switched to branch..."}
data: {"type": "PR Created", "pr_url": "https://github.com/owner/repo/pull/123"}
```

## 🎯 Usage Examples

### Example 1: Add Health Check Endpoint

```bash
curl -X POST "http://localhost:8000/code" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/yourusername/my-api",
    "prompt": "Add a health check endpoint that returns API status and version",
    "branch": "feature/health-check"
  }'
```

### Example 2: Add Input Validation

```bash
curl -X POST "http://localhost:8000/code" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/yourusername/my-app",
    "prompt": "Add Pydantic validation models for all API endpoints",
    "branch": "feature/validation"
  }'
```

### Example 3: Add Logging

```bash
curl -X POST "http://localhost:8000/code" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/yourusername/ml-project",
    "prompt": "Add structured logging to track model inference times",
    "branch": "feature/logging"
  }'
```

## 🔧 Configuration

### Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude AI
- `GITHUB_TOKEN`: GitHub Personal Access Token for repository access
- `E2B_API_KEY`: E2B API key for secure cloud sandboxing (required)

### Sandbox Configuration

The system uses E2B cloud sandbox that:
- Clones repositories in secure cloud environment
- Executes git operations in complete isolation
- Automatically cleans up after each session
- Supports authentication for pushing changes

## 🧪 Testing

Run the test suite:

```bash
# Test core components
python test_basic_working.py

# Test sandbox functionality
python test_e2b.py

# Test API server
python test_server.py
```

## 🔍 Coding Agent Approach

This implementation uses **Claude AI (Anthropic)** as the coding agent because:

1. **Advanced Code Understanding**: Claude excels at analyzing code structure and context
2. **Strong Reasoning**: Can create comprehensive implementation plans
3. **Safety Focus**: Built-in safety measures for code generation
4. **Long Context**: Can handle large codebases effectively

### Agent Workflow

1. **Repository Analysis**: Analyzes codebase structure and patterns
2. **Implementation Planning**: Creates detailed modification plan
3. **Code Generation**: Generates context-aware code changes
4. **Validation**: Validates changes before applying
5. **Git Operations**: Handles branching, commits, and PR creation

## 🔐 Security

- **Sandboxed Execution**: All operations run in isolated environments
- **Token Management**: Secure handling of API keys and tokens
- **Input Validation**: Comprehensive validation of all inputs
- **Cleanup**: Automatic cleanup of temporary files and sessions

## 🐛 Troubleshooting

### Common Issues

1. **GitHub Authentication Failed**:
   - Ensure your GitHub token has `repo` scope
   - Check token expiration
   - Verify repository access permissions

2. **API Key Issues**:
   - Verify all API keys are set in `.env`
   - Check API key validity and quotas

3. **Git Operations Fail**:
   - Ensure Git is installed and configured
   - Check repository permissions
   - Verify branch naming conventions

### Debug Mode

Set environment variable for verbose logging:
```bash
export LOG_LEVEL=DEBUG
```

## 📈 Performance

- **Concurrent Sessions**: Supports multiple simultaneous coding sessions
- **Streaming Updates**: Real-time progress updates via SSE
- **Efficient Sandboxing**: Optimized temporary directory management
- **Smart Caching**: Reuses analysis results where possible

## 🔄 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Anthropic**: For providing the Claude AI API
- **GitHub**: For repository hosting and API
- **FastAPI**: For the excellent web framework
- **E2B**: For sandboxing inspiration and tools

---

