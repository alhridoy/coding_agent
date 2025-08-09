# Evaluation Framework for Autonomous Coding Agents

This comprehensive evaluation framework provides systematic testing and assessment for autonomous coding agents, specifically designed for Backspace-style requirements.

## 🎯 Overview

The evaluation framework consists of multiple components that work together to provide comprehensive testing:

- **Core Framework** (`eval_framework.py`) - Base evaluation infrastructure
- **Backspace Suite** (`backspace_eval_suite.py`) - Backspace-specific test scenarios
- **Enhanced Evaluators** (`enhanced_evaluators.py`) - Code quality, PR quality, and security analysis
- **Tiered Datasets** (`tiered_datasets.py`) - Progressive difficulty test cases
- **Regression Detection** (`regression_detection.py`) - Performance regression monitoring
- **Test Runner** (`backspace_test_runner.py`) - Orchestrates comprehensive evaluations

## 🚀 Quick Start

### Basic Evaluation

```bash
# Run quick evaluation (3 basic tests)
python -m src.evaluation.eval_runner --mode quick

# Run comprehensive evaluation (all tiers)
python -m src.evaluation.eval_runner --mode comprehensive

# Run progressive evaluation (tier by tier)
python -m src.evaluation.eval_runner --mode progressive
```

### Specific Evaluations

```bash
# Performance benchmarking
python -m src.evaluation.eval_runner --mode performance

# Security-focused testing
python -m src.evaluation.eval_runner --mode security

# Test specific tier only
python -m src.evaluation.eval_runner --tier tier1_basic
```

## 📊 Evaluation Modes

### 1. Quick Evaluation
- **Purpose**: Fast validation of basic functionality
- **Duration**: ~5-10 minutes
- **Tests**: 3 basic test cases
- **Use Case**: Development testing, CI/CD pipelines

### 2. Comprehensive Evaluation
- **Purpose**: Full system assessment
- **Duration**: ~30-60 minutes
- **Tests**: All tiers + performance + regression analysis
- **Use Case**: Release validation, comprehensive assessment

### 3. Progressive Evaluation
- **Purpose**: Tier-by-tier capability assessment
- **Duration**: ~15-30 minutes
- **Tests**: Sequential tier testing (stops on failure)
- **Use Case**: Capability assessment, identifying improvement areas

### 4. Performance Benchmark
- **Purpose**: Speed and efficiency testing
- **Duration**: ~10-15 minutes
- **Tests**: Performance-focused scenarios (multiple runs)
- **Use Case**: Performance optimization, bottleneck identification

### 5. Security Evaluation
- **Purpose**: Security vulnerability assessment
- **Duration**: ~20-30 minutes
- **Tests**: Security-focused test cases
- **Use Case**: Security validation, compliance checking

## 🏗️ Test Tiers

### Tier 1: Basic (Easy)
- **Success Threshold**: 95%
- **Examples**: README updates, comments, license files
- **Focus**: Basic file operations and documentation

### Tier 2: Intermediate (Medium)
- **Success Threshold**: 85%
- **Examples**: API endpoints, validation, logging
- **Focus**: Backend functionality and middleware

### Tier 3: Advanced (Hard)
- **Success Threshold**: 70%
- **Examples**: Database integration, authentication, caching
- **Focus**: Complex system integration

### Tier 4: Expert (Expert)
- **Success Threshold**: 50%
- **Examples**: Distributed systems, advanced security
- **Focus**: Enterprise-level architecture patterns

## 📈 Metrics and Analysis

### Core Metrics
- **Success Rate**: Percentage of tests that create successful PRs
- **Execution Time**: Average time to complete tasks
- **Code Quality Score**: AST-based code quality assessment
- **PR Quality Score**: Pull request title, description, and changes quality
- **Error Rate**: Frequency and types of errors encountered

### Advanced Analysis
- **Streaming Compliance**: Adherence to Backspace streaming specification
- **Security Assessment**: Vulnerability detection and security best practices
- **Performance Insights**: Efficiency patterns and bottleneck identification
- **Regression Detection**: Performance degradation over time

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
API_BASE_URL=http://localhost:8000

# Output Configuration
EVAL_OUTPUT_DIR=evaluation_results

# Execution Configuration
MAX_CONCURRENT_TESTS=2
TEST_TIMEOUT_SECONDS=600

# Regression Thresholds
SUCCESS_RATE_DROP_THRESHOLD=0.10
EXECUTION_TIME_INCREASE_THRESHOLD=0.30
```

### Custom Test Cases

Create custom test cases by extending the `TestCase` class:

```python
from src.evaluation.eval_framework import TestCase

custom_test = TestCase(
    id="custom_feature_test",
    name="Custom Feature Implementation",
    description="Test custom feature implementation",
    repo_url="https://github.com/your/repo",
    prompt="Implement custom feature X with Y requirements",
    expected_outcome={
        "pr_created": True,
        "feature_implemented": True,
        "tests_included": True
    },
    timeout_seconds=600,
    tags=["custom", "feature"],
    difficulty="medium",
    category="backend"
)
```

## 📋 Report Structure

### Summary Report
```json
{
  "summary": {
    "total_tests": 10,
    "passed": 8,
    "failed": 2,
    "success_rate": 80.0,
    "average_execution_time": 245.5
  },
  "backspace_metrics": {
    "streaming_compliance": 0.95,
    "pr_quality_score": 78.5,
    "specification_adherence": 0.88,
    "ready_for_production": true
  },
  "recommendations": [
    "Improve error handling for edge cases",
    "Optimize performance for complex tasks"
  ]
}
```

### Detailed Analysis
- **Code Quality Trends**: Complexity, maintainability, readability scores
- **PR Quality Analysis**: Title/description quality, change scope appropriateness
- **Workflow Efficiency**: Step completion rates, error recovery patterns
- **Performance Insights**: Execution time patterns, scalability assessment

## 🔄 Regression Detection

The framework automatically tracks performance baselines and detects regressions:

### Baseline Management
```python
from src.evaluation.regression_detection import RegressionDetector

detector = RegressionDetector()

# Update baselines with new results
detector.update_baseline(evaluation_results, version="v1.2.0")

# Detect regressions
regressions = detector.detect_regressions(current_results)
```

### Regression Thresholds
- **Success Rate Drop**: 10% decrease triggers regression alert
- **Execution Time Increase**: 30% increase triggers performance regression
- **Quality Score Drop**: 15% decrease triggers quality regression
- **Error Rate Increase**: 2x increase triggers reliability regression

## 🎯 Backspace Specification Compliance

The framework specifically validates Backspace requirements:

### Streaming API Compliance
- Server-Sent Events format validation
- Required event types: `Status`, `Tool: Read`, `Tool: Edit`, `Tool: Bash`
- Event ordering and content validation

### Core Functionality Requirements
- Repository cloning and analysis
- Code modification and file operations
- Git workflow automation (branch, commit, push)
- Pull request creation with proper formatting

### Quality Standards
- PR title and description quality
- Code change appropriateness and scope
- Error handling and recovery
- Performance within acceptable bounds

## 📊 Usage Examples

### Programmatic Usage

```python
import asyncio
from src.evaluation.backspace_test_runner import BackspaceTestRunner

async def run_evaluation():
    runner = BackspaceTestRunner("http://localhost:8000")
    
    # Quick evaluation
    quick_report = await runner.run_quick_evaluation()
    
    # Check if ready for production
    is_ready = quick_report["backspace_metrics"]["ready_for_production"]
    
    if is_ready:
        # Run comprehensive evaluation
        full_report = await runner.run_comprehensive_evaluation()
        return full_report
    else:
        print("System not ready - focus on basic functionality")
        return quick_report

# Run evaluation
report = asyncio.run(run_evaluation())
```

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Run Autonomous Agent Evaluation
  run: |
    python -m src.evaluation.eval_runner --mode quick --api-url ${{ env.API_URL }}
    
- name: Upload Evaluation Results
  uses: actions/upload-artifact@v3
  with:
    name: evaluation-results
    path: evaluation_results/
```

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Failures**
   - Verify API server is running on specified URL
   - Check network connectivity and firewall settings
   - Ensure API endpoints are accessible

2. **Test Timeouts**
   - Increase timeout values for complex tests
   - Check system resources and performance
   - Review test complexity and expectations

3. **Low Success Rates**
   - Review error logs in evaluation results
   - Check API key configurations
   - Validate repository access permissions

4. **Inconsistent Results**
   - Run multiple evaluation rounds
   - Check for external dependencies
   - Review test case stability

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```bash
python -m src.evaluation.eval_runner --mode quick --verbose
```

## 🚀 Production Readiness Criteria

The framework determines production readiness based on:

### Basic Requirements (Must Pass)
- **Tier 1 Success Rate**: ≥ 90%
- **Core Functionality**: Repository operations, file modifications, PR creation
- **Streaming Compliance**: ≥ 80% adherence to specification

### Advanced Requirements (Recommended)
- **Tier 2 Success Rate**: ≥ 75%
- **Performance**: Average execution time < 10 minutes
- **Error Rate**: < 2 errors per test on average
- **Security Score**: ≥ 80/100

### Quality Indicators
- **Code Quality**: Average score ≥ 70
- **PR Quality**: Average score ≥ 60
- **Regression Rate**: < 10% across test runs

## 📈 Continuous Improvement

Use evaluation results to drive continuous improvement:

1. **Identify Weak Areas**: Focus on tiers/categories with low success rates
2. **Performance Optimization**: Address slow-performing test cases
3. **Quality Enhancement**: Improve code and PR quality scores
4. **Regression Prevention**: Monitor and address performance degradations
5. **Security Hardening**: Address security vulnerabilities and improve practices

---

**For more detailed information, see the individual module documentation and example usage in the test files.**