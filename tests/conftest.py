"""
Pytest configuration and fixtures for autonomous coding agent tests
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    with patch.dict(os.environ, {
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GITHUB_TOKEN": "test-github-token",
        "E2B_API_KEY": "test-e2b-key",
        "LANGSMITH_API_KEY": "test-langsmith-key"
    }):
        yield


@pytest.fixture
def sample_repo_files():
    """Sample files for testing repository operations"""
    return {
        "README.md": """# Test Repository

This is a test repository for the autonomous coding agent.

## Features

- Flask web application
- Basic routing
- Health check endpoint needed

## Installation

```bash
pip install -r requirements.txt
python app.py
```
""",
        "app.py": """
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/api/users')
def get_users():
    return jsonify([
        {'id': 1, 'name': 'John'},
        {'id': 2, 'name': 'Jane'}
    ])

if __name__ == '__main__':
    app.run(debug=True)
""",
        "requirements.txt": """
Flask==2.0.1
requests==2.25.1
pytest==6.2.4
""",
        "tests/test_app.py": """
import pytest
from app import app

def test_hello():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'Hello World!' in response.data

def test_get_users():
    client = app.test_client()
    response = client.get('/api/users')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data) == 2
"""
    }


@pytest.fixture
def mock_github_repo(temp_dir, sample_repo_files):
    """Create a mock GitHub repository with sample files"""
    repo_dir = os.path.join(temp_dir, "test-repo")
    os.makedirs(repo_dir)
    
    # Create sample files
    for file_path, content in sample_repo_files.items():
        full_path = os.path.join(repo_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
    
    # Initialize git repo
    os.system(f"cd {repo_dir} && git init")
    os.system(f"cd {repo_dir} && git config user.email 'test@example.com'")
    os.system(f"cd {repo_dir} && git config user.name 'Test User'")
    os.system(f"cd {repo_dir} && git add .")
    os.system(f"cd {repo_dir} && git commit -m 'Initial commit'")
    
    yield repo_dir


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic API response"""
    class MockResponse:
        def __init__(self, text):
            self.text = text
    
    class MockContent:
        def __init__(self, text):
            self.content = [MockResponse(text)]
    
    return MockContent("""
I'll add a health check endpoint to the Flask application.

Looking at the code, I need to:
1. Add a new route `/health` that returns the API status
2. Include version information and timestamp
3. Make it return JSON format

Here's the implementation:

```python
@app.route('/health')
def health_check():
    import datetime
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.datetime.now().isoformat()
    })
```

This will provide a simple health check endpoint that applications can use to verify the API is running properly.
""")


# Configure pytest
def pytest_configure(config):
    """Configure pytest settings"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )