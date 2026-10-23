"""Public model-validation API."""

from .models.pipeline_validation import ValidationResult, validate_model

__all__ = ["ValidationResult", "validate_model"]
