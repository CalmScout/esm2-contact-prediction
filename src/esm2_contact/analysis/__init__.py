"""
Analysis Module for ESM2 Contact Prediction

This module provides tools for analyzing MLflow experiment data,
visualizing training performance, and generating insights about model behavior.

Components:
- MLflowAnalyzer: Extract and analyze MLflow experiment data
- PerformanceAnalyzer: Specialized analysis for contact prediction metrics
- Visualization utilities: Plotting and chart generation
- Analysis tools: Command-line and notebook-based analysis

Usage:
    from esm2_contact.analysis import MLflowAnalyzer, PerformanceAnalyzer

    # Analyze MLflow experiment
    analyzer = MLflowAnalyzer()
    run_data = analyzer.load_experiment("protein_contact_prediction")

    # Analyze performance
    perf_analyzer = PerformanceAnalyzer()
    insights = perf_analyzer.analyze_performance(run_data)
"""

from .mlflow_analyzer import MLflowAnalyzer
from .performance_analyzer import PerformanceAnalyzer

__all__ = [
    'MLflowAnalyzer',
    'PerformanceAnalyzer'
]

__version__ = '1.0.0'