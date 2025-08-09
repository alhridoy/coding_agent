"""
Evaluation framework for autonomous coding agents
"""

from .eval_framework import EvaluationFramework, EvaluationResult, TestCase
from .backspace_eval_suite import BackspaceEvaluationSuite
from .enhanced_evaluators import (
    CodeQualityEvaluator,
    PRQualityEvaluator, 
    WorkflowEvaluator,
    SecurityEvaluator
)
from .tiered_datasets import TieredDatasets
from .regression_detection import RegressionDetector

__all__ = [
    'EvaluationFramework',
    'EvaluationResult', 
    'TestCase',
    'BackspaceEvaluationSuite',
    'CodeQualityEvaluator',
    'PRQualityEvaluator',
    'WorkflowEvaluator',
    'SecurityEvaluator',
    'TieredDatasets',
    'RegressionDetector'
]