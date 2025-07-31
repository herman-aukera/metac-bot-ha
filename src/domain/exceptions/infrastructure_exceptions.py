"""
Infrastructure-specific exceptions for the tournament optimization system.
"""

from typing import Optional, Dict, Any, List
from .base_exceptions import TournamentOptimizationError


class InfrastructureError(TournamentOptimizationError):
    """Base class for infrastructure-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, **kwargs)
        self.context.component = "infrastructure"


class ExternalServiceError(InfrastructureError):
    """
    Raised when external service operations fail.

    Includes information about the service, error type,
    and whether the error is recoverable.
    """

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.endpoint = endpoint
        self.status_code = status_code
        self.response_body = response_body
        self.context.metadata.update({
            "service_name": service_name,
            "endpoint": endpoint,
            "status_code": status_code,
            "response_body": response_body,
        })
        self.context.operation = "external_service_request"

        # Determine if error is recoverable based on status code
        if status_code:
            self.recoverable = status_code in [429, 500, 502, 503, 504]
            if status_code == 429:  # Rate limit
                self.retry_after = 60
            elif status_code in [500, 502, 503, 504]:  # Server errors
                self.retry_after = 30


class NetworkError(InfrastructureError):
    """
    Raised when network operations fail.

    Includes information about the network operation and
    connection details.
    """

    def __init__(
        self,
        message: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.host = host
        self.port = port
        self.operation = operation
        self.context.metadata.update({
            "host": host,
            "port": port,
            "operation": operation,
        })
        self.context.operation = "network_request"
        self.recoverable = True
        self.retry_after = 30


class TimeoutError(InfrastructureError):
    """
    Raised when operations exceed timeout limits.

    Includes information about timeout duration and
    the operation that timed out.
    """

    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration
        self.operation_type = operation_type
        self.context.metadata.update({
            "timeout_duration": timeout_duration,
            "operation_type": operation_type,
        })
        self.context.operation = "timeout"
        self.recoverable = True
        self.retry_after = 60


class RateLimitError(InfrastructureError):
    """
    Raised when rate limits are exceeded.

    Includes information about rate limit details and
    when the limit will reset.
    """

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        limit: Optional[int] = None,
        remaining: Optional[int] = None,
        reset_time: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.limit = limit
        self.remaining = remaining
        self.reset_time = reset_time
        self.context.metadata.update({
            "service_name": service_name,
            "limit": limit,
            "remaining": remaining,
            "reset_time": reset_time,
        })
        self.context.operation = "rate_limit_check"
        self.recoverable = True
        self.retry_after = reset_time or 60


class AuthenticationError(InfrastructureError):
    """
    Raised when authentication fails.

    Includes information about the authentication method and
    the service that rejected the authentication.
    """

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        auth_method: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.service_name = service_name
        self.auth_method = auth_method
        self.context.metadata.update({
            "service_name": service_name,
            "auth_method": auth_method,
        })
        self.context.operation = "authentication"
        self.recoverable = False  # Usually requires manual intervention


class ConfigurationError(InfrastructureError):
    """
    Raised when configuration is invalid or missing.

    Includes information about the configuration issue and
    the specific configuration that is problematic.
    """

    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.config_section = config_section
        self.config_key = config_key
        self.config_value = config_value
        self.context.metadata.update({
            "config_section": config_section,
            "config_key": config_key,
            "config_value": str(config_value) if config_value is not None else None,
        })
        self.context.operation = "configuration_validation"
        self.recoverable = False  # Usually requires configuration fix


class DatabaseError(InfrastructureError):
    """
    Raised when database operations fail.

    Includes information about the database operation and
    connection details.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        table: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.table = table
        self.query = query
        self.context.metadata.update({
            "operation": operation,
            "table": table,
            "query": query,
        })
        self.context.operation = "database_operation"
        self.recoverable = True
        self.retry_after = 30


class CacheError(InfrastructureError):
    """
    Raised when cache operations fail.

    Includes information about the cache operation and
    the cache key that was involved.
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.operation = operation
        self.cache_key = cache_key
        self.context.metadata.update({
            "operation": operation,
            "cache_key": cache_key,
        })
        self.context.operation = "cache_operation"
        self.recoverable = True  # Cache failures are usually non-critical


# Security-related exceptions

class SecurityError(InfrastructureError):
    """
    Raised when security checks fail.

    Includes information about the security violation and
    the context in which it occurred.
    """

    def __init__(
        self,
        message: str,
        security_level: Optional[str] = None,
        violation_type: Optional[str] = None,
        client_ip: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.security_level = security_level
        self.violation_type = violation_type
        self.client_ip = client_ip
        self.context.metadata.update({
            "security_level": security_level,
            "violation_type": violation_type,
            "client_ip": client_ip,
        })
        self.context.operation = "security_check"
        self.recoverable = False


class AuthorizationError(InfrastructureError):
    """
    Raised when authorization checks fail.

    Includes information about the required permissions and
    the user's actual permissions.
    """

    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        required_permissions: Optional[List[str]] = None,
        user_permissions: Optional[List[str]] = None,
        resource: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.user_id = user_id
        self.required_permissions = required_permissions or []
        self.user_permissions = user_permissions or []
        self.resource = resource
        self.context.metadata.update({
            "user_id": user_id,
            "required_permissions": required_permissions,
            "user_permissions": user_permissions,
            "resource": resource,
        })
        self.context.operation = "authorization_check"
        self.recoverable = False


class ValidationError(InfrastructureError):
    """
    Raised when input validation fails.

    Includes information about the validation errors and
    the fields that failed validation.
    """

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.validation_errors = validation_errors or []
        self.context.metadata.update({
            "field_name": field_name,
            "validation_errors": validation_errors,
        })
        self.context.operation = "input_validation"
        self.recoverable = False


class RateLimitExceededError(InfrastructureError):
    """
    Raised when rate limits are exceeded.

    Includes information about the rate limit rule and
    when the limit will reset.
    """

    def __init__(
        self,
        message: str,
        rule_name: Optional[str] = None,
        identifier: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.rule_name = rule_name
        self.identifier = identifier
        self.retry_after = retry_after
        self.context.metadata.update({
            "rule_name": rule_name,
            "identifier": identifier,
            "retry_after": retry_after,
        })
        self.context.operation = "rate_limit_check"
        self.recoverable = True


class CredentialError(InfrastructureError):
    """
    Raised when credential operations fail.

    Includes information about the credential service and
    the type of credential that failed.
    """

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        credential_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.service = service
        self.credential_type = credential_type
        self.context.metadata.update({
            "service": service,
            "credential_type": credential_type,
        })
        self.context.operation = "credential_management"
        self.recoverable = False


class VaultConnectionError(InfrastructureError):
    """
    Raised when vault connection operations fail.

    Includes information about the vault operation and
    connection details.
    """

    def __init__(
        self,
        message: str,
        vault_url: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.vault_url = vault_url
        self.operation = operation
        self.context.metadata.update({
            "vault_url": vault_url,
            "operation": operation,
        })
        self.context.operation = "vault_connection"
        self.recoverable = True
        self.retry_after = 30


class CredentialRotationError(InfrastructureError):
    """
    Raised when credential rotation operations fail.

    Includes information about the credential that failed to rotate
    and the rotation attempt details.
    """

    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        credential_type: Optional[str] = None,
        rotation_attempt: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.service = service
        self.credential_type = credential_type
        self.rotation_attempt = rotation_attempt
        self.context.metadata.update({
            "service": service,
            "credential_type": credential_type,
            "rotation_attempt": rotation_attempt,
        })
        self.context.operation = "credential_rotation"
        self.recoverable = True
        self.retry_after = 300  # 5 minutes
