"""Normalized benchmark contracts for llm-wiki evaluations."""

from .schema import BenchmarkCase, Prediction, validate_case, validate_prediction

__all__ = ["BenchmarkCase", "Prediction", "validate_case", "validate_prediction"]
