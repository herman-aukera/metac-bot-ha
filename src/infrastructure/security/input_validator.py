"""Comprehensive input validation with XSS protection, SQL injection prevention, and data sanitization."""

import re
import html
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import bleach
from urllib.parse import urlparse
import ipaddress

from ...domain.exceptions.infrastructure_exceptions import ValidationError


class ValidationType(Enum):
    """Types of validation to perform."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    IP_ADDRESS = "ip_address"
    JSON = "json"
    HTML = "html"
    SQL_SAFE = "sql_safe"
    FILENAME = "filename"
    UUID = "uuid"
    PREDICTION_VALUE = "prediction_value"
    QUESTION_TEXT = "question_text"
    API_RESPONSE = "api_response"


@dataclass
class ValidationRule:
    """Validation rule configuration."""
    field_name: str
    validation_type: ValidationType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    sanitize: bool = True


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    sanitized_value: Any
    errors: List[str]
    warnings: List[str]


class InputValidator:
    """Comprehensive input validation and sanitization."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # XSS protection configuration
        self.allowed_html_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'blockquote'
        ]
        self.allowed_html_attributes = {
            '*': ['class'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'width', 'height']
        }

        # SQL injection patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+['\"].*['\"])",
            r"(;|\|\||&&)",
            r"(\bxp_cmdshell\b|\bsp_executesql\b)"
        ]

        # Common validation patterns
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            'filename': r'^[a-zA-Z0-9._-]+$',
            'safe_string': r'^[a-zA-Z0-9\s\-_.,:;!?()]+$'
        }

        # Dangerous file extensions
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
            '.jar', '.php', '.asp', '.aspx', '.jsp', '.py', '.rb', '.pl'
        }

    def validate_field(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate a single field according to its rule."""
        errors = []
        warnings = []
        sanitized_value = value

        try:
            # Check if required field is present
            if rule.required and (value is None or value == ""):
                errors.append(f"Field '{rule.field_name}' is required")
                return ValidationResult(False, None, errors, warnings)

            # Skip validation for optional empty fields
            if not rule.required and (value is None or value == ""):
                return ValidationResult(True, None, errors, warnings)

            # Type-specific validation
            if rule.validation_type == ValidationType.STRING:
                sanitized_value = self._validate_string(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.INTEGER:
                sanitized_value = self._validate_integer(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.FLOAT:
                sanitized_value = self._validate_float(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.BOOLEAN:
                sanitized_value = self._validate_boolean(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.EMAIL:
                sanitized_value = self._validate_email(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.URL:
                sanitized_value = self._validate_url(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.IP_ADDRESS:
                sanitized_value = self._validate_ip_address(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.JSON:
                sanitized_value = self._validate_json(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.HTML:
                sanitized_value = self._validate_html(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.SQL_SAFE:
                sanitized_value = self._validate_sql_safe(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.FILENAME:
                sanitized_value = self._validate_filename(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.UUID:
                sanitized_value = self._validate_uuid(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.PREDICTION_VALUE:
                sanitized_value = self._validate_prediction_value(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.QUESTION_TEXT:
                sanitized_value = self._validate_question_text(value, rule, errors, warnings)
            elif rule.validation_type == ValidationType.API_RESPONSE:
                sanitized_value = self._validate_api_response(value, rule, errors, warnings)

            # Custom validation
            if rule.custom_validator and not rule.custom_validator(sanitized_value):
                errors.append(f"Field '{rule.field_name}' failed custom validation")

            is_valid = len(errors) == 0
            return ValidationResult(is_valid, sanitized_value, errors, warnings)

        except Exception as e:
            self.logger.error(f"Validation error for field '{rule.field_name}': {e}")
            errors.append(f"Validation failed for field '{rule.field_name}': {str(e)}")
            return ValidationResult(False, value, errors, warnings)

    def validate_data(self, data: Dict[str, Any], rules: List[ValidationRule]) -> Dict[str, ValidationResult]:
        """Validate multiple fields according to their rules."""
        results = {}

        for rule in rules:
            field_value = data.get(rule.field_name)
            results[rule.field_name] = self.validate_field(field_value, rule)

        return results

    def _validate_string(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> str:
        """Validate and sanitize string value."""
        if not isinstance(value, str):
            try:
                value = str(value)
            except Exception:
                errors.append(f"Cannot convert '{rule.field_name}' to string")
                return value

        # Length validation
        if rule.min_length is not None and len(value) < rule.min_length:
            errors.append(f"Field '{rule.field_name}' must be at least {rule.min_length} characters")

        if rule.max_length is not None and len(value) > rule.max_length:
            errors.append(f"Field '{rule.field_name}' must be at most {rule.max_length} characters")

        # Pattern validation
        if rule.pattern and not re.match(rule.pattern, value):
            errors.append(f"Field '{rule.field_name}' does not match required pattern")

        # Allowed values validation
        if rule.allowed_values and value not in rule.allowed_values:
            errors.append(f"Field '{rule.field_name}' must be one of: {rule.allowed_values}")

        # Sanitization
        if rule.sanitize:
            # Remove null bytes
            value = value.replace('\x00', '')

            # Normalize whitespace
            value = ' '.join(value.split())

            # HTML escape for safety
            value = html.escape(value)

        return value

    def _validate_integer(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> int:
        """Validate integer value."""
        try:
            if isinstance(value, str):
                value = int(value)
            elif isinstance(value, float):
                if value.is_integer():
                    value = int(value)
                else:
                    errors.append(f"Field '{rule.field_name}' must be an integer")
                    return value
            elif not isinstance(value, int):
                errors.append(f"Field '{rule.field_name}' must be an integer")
                return value
        except ValueError:
            errors.append(f"Field '{rule.field_name}' must be a valid integer")
            return value

        # Range validation
        if rule.min_value is not None and value < rule.min_value:
            errors.append(f"Field '{rule.field_name}' must be at least {rule.min_value}")

        if rule.max_value is not None and value > rule.max_value:
            errors.append(f"Field '{rule.field_name}' must be at most {rule.max_value}")

        return value

    def _validate_float(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> float:
        """Validate float value."""
        try:
            value = float(value)
        except (ValueError, TypeError):
            errors.append(f"Field '{rule.field_name}' must be a valid number")
            return value

        # Range validation
        if rule.min_value is not None and value < rule.min_value:
            errors.append(f"Field '{rule.field_name}' must be at least {rule.min_value}")

        if rule.max_value is not None and value > rule.max_value:
            errors.append(f"Field '{rule.field_name}' must be at most {rule.max_value}")

        return value

    def _validate_boolean(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> bool:
        """Validate boolean value."""
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ['true', '1', 'yes', 'on']:
                return True
            elif value_lower in ['false', '0', 'no', 'off']:
                return False

        if isinstance(value, (int, float)):
            return bool(value)

        errors.append(f"Field '{rule.field_name}' must be a valid boolean")
        return value

    def _validate_email(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> str:
        """Validate email address."""
        if not isinstance(value, str):
            errors.append(f"Field '{rule.field_name}' must be a string")
            return value

        # Basic email pattern validation
        if not re.match(self.patterns['email'], value):
            errors.append(f"Field '{rule.field_name}' must be a valid email address")

        # Sanitize
        if rule.sanitize:
            value = value.strip().lower()

        return value

    def _validate_url(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> str:
        """Validate URL."""
        if not isinstance(value, str):
            errors.append(f"Field '{rule.field_name}' must be a string")
            return value

        try:
            parsed = urlparse(value)
            if not parsed.scheme or not parsed.netloc:
                errors.append(f"Field '{rule.field_name}' must be a valid URL")
        except Exception:
            errors.append(f"Field '{rule.field_name}' must be a valid URL")

        return value

    def _validate_ip_address(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> str:
        """Validate IP address."""
        if not isinstance(value, str):
            errors.append(f"Field '{rule.field_name}' must be a string")
            return value

        try:
            ipaddress.ip_address(value)
        except ValueError:
            errors.append(f"Field '{rule.field_name}' must be a valid IP address")

        return value

    def _validate_json(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> Any:
        """Validate JSON data."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                errors.append(f"Field '{rule.field_name}' must be valid JSON")
                return value

        # Additional JSON structure validation could be added here
        return value

    def _validate_html(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> str:
        """Validate and sanitize HTML content."""
        if not isinstance(value, str):
            errors.append(f"Field '{rule.field_name}' must be a string")
            return value

        # Sanitize HTML using bleach
        if rule.sanitize:
            value = bleach.clean(
                value,
                tags=self.allowed_html_tags,
                attributes=self.allowed_html_attributes,
                strip=True
            )

        return value

    def _validate_sql_safe(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> str:
        """Validate that string is safe from SQL injection."""
        if not isinstance(value, str):
            errors.append(f"Field '{rule.field_name}' must be a string")
            return value

        # Check for SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                errors.append(f"Field '{rule.field_name}' contains potentially dangerous SQL patterns")
                break

        # Sanitize by escaping single quotes
        if rule.sanitize:
            value = value.replace("'", "''")

        return value

    def _validate_filename(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> str:
        """Validate filename for security."""
        if not isinstance(value, str):
            errors.append(f"Field '{rule.field_name}' must be a string")
            return value

        # Check for dangerous patterns
        if '..' in value or '/' in value or '\\' in value:
            errors.append(f"Field '{rule.field_name}' contains invalid path characters")

        # Check for dangerous extensions
        for ext in self.dangerous_extensions:
            if value.lower().endswith(ext):
                errors.append(f"Field '{rule.field_name}' has a potentially dangerous file extension")
                break

        # Pattern validation
        if not re.match(self.patterns['filename'], value):
            errors.append(f"Field '{rule.field_name}' contains invalid filename characters")

        return value

    def _validate_uuid(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> str:
        """Validate UUID format."""
        if not isinstance(value, str):
            errors.append(f"Field '{rule.field_name}' must be a string")
            return value

        if not re.match(self.patterns['uuid'], value.lower()):
            errors.append(f"Field '{rule.field_name}' must be a valid UUID")

        return value.lower()

    def _validate_prediction_value(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> Any:
        """Validate prediction value (binary, numeric, or multiple choice)."""
        # Binary prediction (0.0 to 1.0)
        if isinstance(value, (int, float)):
            if 0.0 <= value <= 1.0:
                return float(value)
            else:
                errors.append(f"Field '{rule.field_name}' must be between 0.0 and 1.0 for binary predictions")

        # Multiple choice prediction (dictionary with probabilities)
        elif isinstance(value, dict):
            total_prob = 0.0
            for choice, prob in value.items():
                if not isinstance(prob, (int, float)):
                    errors.append(f"Probability for choice '{choice}' must be a number")
                    continue
                if not 0.0 <= prob <= 1.0:
                    errors.append(f"Probability for choice '{choice}' must be between 0.0 and 1.0")
                total_prob += prob

            if abs(total_prob - 1.0) > 0.001:  # Allow small floating point errors
                warnings.append(f"Probabilities should sum to 1.0, got {total_prob}")

        else:
            errors.append(f"Field '{rule.field_name}' must be a number (binary) or dictionary (multiple choice)")

        return value

    def _validate_question_text(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> str:
        """Validate question text for forecasting."""
        if not isinstance(value, str):
            errors.append(f"Field '{rule.field_name}' must be a string")
            return value

        # Length validation
        if len(value.strip()) < 10:
            errors.append(f"Field '{rule.field_name}' must be at least 10 characters")

        if len(value) > 10000:
            errors.append(f"Field '{rule.field_name}' must be at most 10000 characters")

        # Check for question mark (optional warning)
        if not value.strip().endswith('?'):
            warnings.append(f"Field '{rule.field_name}' should typically end with a question mark")

        # Sanitize HTML and remove excessive whitespace
        if rule.sanitize:
            value = bleach.clean(value, tags=[], strip=True)
            value = ' '.join(value.split())

        return value

    def _validate_api_response(self, value: Any, rule: ValidationRule, errors: List[str], warnings: List[str]) -> Any:
        """Validate API response data."""
        # Check for common API response structure
        if isinstance(value, dict):
            # Check for error indicators
            if 'error' in value and value['error']:
                warnings.append(f"API response contains error: {value.get('error')}")

            # Validate common fields
            if 'status' in value and value['status'] not in ['success', 'ok', 200]:
                warnings.append(f"API response has non-success status: {value['status']}")

        return value

    def sanitize_for_logging(self, data: Any) -> Any:
        """Sanitize data for safe logging."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                key_lower = key.lower()

                # Mask sensitive fields
                if any(sensitive in key_lower for sensitive in ['password', 'token', 'key', 'secret']):
                    sanitized[key] = "***MASKED***"
                elif isinstance(value, (dict, list)):
                    sanitized[key] = self.sanitize_for_logging(value)
                else:
                    sanitized[key] = value
            return sanitized

        elif isinstance(data, list):
            return [self.sanitize_for_logging(item) for item in data]

        else:
            return data

    def detect_xss_attempt(self, value: str) -> bool:
        """Detect potential XSS attack patterns."""
        if not isinstance(value, str):
            return False

        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'<link[^>]*>',
            r'<meta[^>]*>',
            r'vbscript:',
            r'data:text/html'
        ]

        for pattern in xss_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        return False

    def detect_sql_injection_attempt(self, value: str) -> bool:
        """Detect potential SQL injection attack patterns."""
        if not isinstance(value, str):
            return False

        for pattern in self.sql_injection_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True

        return False

    def create_validation_rules_for_question(self) -> List[ValidationRule]:
        """Create validation rules for question data."""
        return [
            ValidationRule(
                field_name="title",
                validation_type=ValidationType.STRING,
                required=True,
                min_length=5,
                max_length=500,
                sanitize=True
            ),
            ValidationRule(
                field_name="description",
                validation_type=ValidationType.QUESTION_TEXT,
                required=True,
                sanitize=True
            ),
            ValidationRule(
                field_name="question_type",
                validation_type=ValidationType.STRING,
                required=True,
                allowed_values=["binary", "numeric", "multiple_choice"]
            ),
            ValidationRule(
                field_name="close_time",
                validation_type=ValidationType.STRING,
                required=True
            ),
            ValidationRule(
                field_name="categories",
                validation_type=ValidationType.JSON,
                required=False
            )
        ]

    def create_validation_rules_for_prediction(self) -> List[ValidationRule]:
        """Create validation rules for prediction data."""
        return [
            ValidationRule(
                field_name="question_id",
                validation_type=ValidationType.UUID,
                required=True
            ),
            ValidationRule(
                field_name="prediction_value",
                validation_type=ValidationType.PREDICTION_VALUE,
                required=True
            ),
            ValidationRule(
                field_name="confidence",
                validation_type=ValidationType.FLOAT,
                required=True,
                min_value=0.0,
                max_value=1.0
            ),
            ValidationRule(
                field_name="reasoning",
                validation_type=ValidationType.STRING,
                required=False,
                max_length=5000,
                sanitize=True
            )
        ]
