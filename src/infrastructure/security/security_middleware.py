"""Security middleware for request processing with comprehensive security checks."""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import ipaddress
from urllib.parse import urlparse

from .input_validator import InputValidator, ValidationRule, ValidationType
from .rate_limiter import RateLimiter
from .audit_logger import AuditLogger
from ...domain.exceptions.infrastructure_exceptions import (
    SecurityError,
    AuthenticationError,
    AuthorizationError,
    RateLimitExceededError
)


class SecurityLevel(Enum):
    """Security levels for different endpoints."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityConfig:
    """Security configuration for middleware."""
    level: SecurityLevel
    require_authentication: bool = True
    require_authorization: bool = True
    enable_rate_limiting: bool = True
    rate_limit_rule: Optional[str] = None
    validate_input: bool = True
    validation_rules: Optional[List[ValidationRule]] = None
    allowed_origins: Optional[List[str]] = None
    allowed_ips: Optional[List[str]] = None
    require_https: bool = True
    max_request_size: int = 1024 * 1024  # 1MB
    enable_csrf_protection: bool = True
    custom_checks: Optional[List[Callable]] = None


@dataclass
class SecurityContext:
    """Security context for request processing."""
    request_id: str
    client_ip: str
    user_agent: str
    origin: Optional[str]
    authenticated_user: Optional[str]
    user_roles: List[str]
    request_timestamp: float
    security_level: SecurityLevel
    metadata: Dict[str, Any]


class SecurityMiddleware:
    """Security middleware for request processing with comprehensive security checks."""

    def __init__(self,
                 input_validator: Optional[InputValidator] = None,
                 rate_limiter: Optional[RateLimiter] = None,
                 audit_logger: Optional[AuditLogger] = None):
        self.input_validator = input_validator or InputValidator()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.audit_logger = audit_logger or AuditLogger()
        self.logger = logging.getLogger(__name__)

        # Security configurations by endpoint
        self.endpoint_configs: Dict[str, SecurityConfig] = {}

        # Blocked IPs and user agents
        self.blocked_ips: set = set()
        self.blocked_user_agents: set = set()

        # CSRF token storage (in production, use Redis or database)
        self.csrf_tokens: Dict[str, float] = {}

        # API key validation
        self.api_keys: Dict[str, Dict[str, Any]] = {}

        # Initialize default configurations
        self._setup_default_configs()

    def _setup_default_configs(self) -> None:
        """Setup default security configurations."""
        # High security for authentication endpoints
        self.add_endpoint_config("/auth/login", SecurityConfig(
            level=SecurityLevel.HIGH,
            require_authentication=False,  # Login endpoint
            enable_rate_limiting=True,
            rate_limit_rule="auth_attempts",
            validate_input=True,
            require_https=True,
            enable_csrf_protection=True
        ))

        # Critical security for admin endpoints
        self.add_endpoint_config("/admin/*", SecurityConfig(
            level=SecurityLevel.CRITICAL,
            require_authentication=True,
            require_authorization=True,
            enable_rate_limiting=True,
            rate_limit_rule="admin_operations",
            validate_input=True,
            require_https=True,
            enable_csrf_protection=True
        ))

        # Medium security for API endpoints
        self.add_endpoint_config("/api/*", SecurityConfig(
            level=SecurityLevel.MEDIUM,
            require_authentication=True,
            enable_rate_limiting=True,
            rate_limit_rule="api_general",
            validate_input=True,
            require_https=True
        ))

        # Low security for public endpoints
        self.add_endpoint_config("/public/*", SecurityConfig(
            level=SecurityLevel.LOW,
            require_authentication=False,
            require_authorization=False,
            enable_rate_limiting=True,
            rate_limit_rule="public_access",
            validate_input=True
        ))

    def add_endpoint_config(self, endpoint_pattern: str, config: SecurityConfig) -> None:
        """Add security configuration for an endpoint pattern."""
        self.endpoint_configs[endpoint_pattern] = config
        self.logger.info(f"Added security config for {endpoint_pattern}: {config.level.value}")

    def get_endpoint_config(self, endpoint: str) -> SecurityConfig:
        """Get security configuration for an endpoint."""
        # Exact match first
        if endpoint in self.endpoint_configs:
            return self.endpoint_configs[endpoint]

        # Pattern matching
        for pattern, config in self.endpoint_configs.items():
            if pattern.endswith('*'):
                prefix = pattern[:-1]
                if endpoint.startswith(prefix):
                    return config
            elif pattern.startswith('*'):
                suffix = pattern[1:]
                if endpoint.endswith(suffix):
                    return config

        # Default configuration
        return SecurityConfig(
            level=SecurityLevel.MEDIUM,
            require_authentication=True,
            enable_rate_limiting=True,
            validate_input=True
        )

    async def process_request(self,
                            endpoint: str,
                            request_data: Dict[str, Any],
                            headers: Dict[str, str],
                            client_ip: str,
                            method: str = "POST") -> SecurityContext:
        """Process request through security middleware."""
        request_id = self._generate_request_id()
        start_time = time.time()

        try:
            # Get security configuration
            config = self.get_endpoint_config(endpoint)

            # Create security context
            context = SecurityContext(
                request_id=request_id,
                client_ip=client_ip,
                user_agent=headers.get('User-Agent', ''),
                origin=headers.get('Origin'),
                authenticated_user=None,
                user_roles=[],
                request_timestamp=start_time,
                security_level=config.level,
                metadata={}
            )

            # Log security event
            await self.audit_logger.log_security_event(
                event_type="REQUEST_START",
                endpoint=endpoint,
                client_ip=client_ip,
                user_agent=context.user_agent,
                security_level=config.level.value,
                request_id=request_id
            )

            # Perform security checks
            await self._check_blocked_clients(context)
            await self._check_https_requirement(config, headers)
            await self._check_request_size(config, request_data)
            await self._check_cors(config, context)
            await self._check_rate_limiting(config, context, endpoint)
            await self._check_authentication(config, context, headers)
            await self._check_authorization(config, context, endpoint)
            await self._check_csrf_protection(config, context, headers, method)
            await self._validate_input(config, request_data)
            await self._run_custom_checks(config, context, request_data)

            # Log successful security check
            await self.audit_logger.log_security_event(
                event_type="SECURITY_CHECK_PASSED",
                endpoint=endpoint,
                client_ip=client_ip,
                user=context.authenticated_user,
                security_level=config.level.value,
                request_id=request_id,
                processing_time=time.time() - start_time
            )

            return context

        except Exception as e:
            # Log security failure
            await self.audit_logger.log_security_event(
                event_type="SECURITY_CHECK_FAILED",
                endpoint=endpoint,
                client_ip=client_ip,
                error=str(e),
                security_level=config.level.value if 'config' in locals() else "unknown",
                request_id=request_id,
                processing_time=time.time() - start_time
            )

            # Re-raise the exception
            raise

    async def _check_blocked_clients(self, context: SecurityContext) -> None:
        """Check if client IP or user agent is blocked."""
        if context.client_ip in self.blocked_ips:
            raise SecurityError(
                f"IP address {context.client_ip} is blocked",
                error_code="IP_BLOCKED",
                context={"client_ip": context.client_ip}
            )

        if context.user_agent in self.blocked_user_agents:
            raise SecurityError(
                f"User agent is blocked",
                error_code="USER_AGENT_BLOCKED",
                context={"user_agent": context.user_agent}
            )

    async def _check_https_requirement(self, config: SecurityConfig, headers: Dict[str, str]) -> None:
        """Check HTTPS requirement."""
        if config.require_https:
            # Check various headers that indicate HTTPS
            is_https = (
                headers.get('X-Forwarded-Proto') == 'https' or
                headers.get('X-Forwarded-Ssl') == 'on' or
                headers.get('X-Url-Scheme') == 'https' or
                headers.get('Forwarded', '').find('proto=https') != -1
            )

            if not is_https:
                raise SecurityError(
                    "HTTPS is required for this endpoint",
                    error_code="HTTPS_REQUIRED"
                )

    async def _check_request_size(self, config: SecurityConfig, request_data: Dict[str, Any]) -> None:
        """Check request size limits."""
        import sys
        request_size = sys.getsizeof(str(request_data))

        if request_size > config.max_request_size:
            raise SecurityError(
                f"Request size {request_size} exceeds maximum {config.max_request_size}",
                error_code="REQUEST_TOO_LARGE",
                context={"request_size": request_size, "max_size": config.max_request_size}
            )

    async def _check_cors(self, config: SecurityConfig, context: SecurityContext) -> None:
        """Check CORS policy."""
        if config.allowed_origins and context.origin:
            if context.origin not in config.allowed_origins:
                raise SecurityError(
                    f"Origin {context.origin} not allowed",
                    error_code="CORS_VIOLATION",
                    context={"origin": context.origin, "allowed_origins": config.allowed_origins}
                )

    async def _check_rate_limiting(self, config: SecurityConfig, context: SecurityContext, endpoint: str) -> None:
        """Check rate limiting."""
        if config.enable_rate_limiting and config.rate_limit_rule:
            try:
                await self.rate_limiter.middleware_check(
                    config.rate_limit_rule,
                    context.client_ip
                )
            except RateLimitExceededError as e:
                # Add additional context
                e.context.update({
                    "endpoint": endpoint,
                    "client_ip": context.client_ip,
                    "user_agent": context.user_agent
                })
                raise

    async def _check_authentication(self, config: SecurityConfig, context: SecurityContext, headers: Dict[str, str]) -> None:
        """Check authentication requirements."""
        if not config.require_authentication:
            return

        # Check for API key
        api_key = headers.get('X-API-Key') or headers.get('Authorization', '').replace('Bearer ', '')

        if not api_key:
            raise AuthenticationError(
                "Authentication required",
                error_code="AUTHENTICATION_REQUIRED"
            )

        # Validate API key
        user_info = await self._validate_api_key(api_key)
        if not user_info:
            raise AuthenticationError(
                "Invalid authentication credentials",
                error_code="INVALID_CREDENTIALS"
            )

        # Update context with authenticated user
        context.authenticated_user = user_info.get('user_id')
        context.user_roles = user_info.get('roles', [])
        context.metadata.update(user_info)

    async def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user information."""
        # Hash the API key for lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Check in-memory store (in production, use database)
        if key_hash in self.api_keys:
            key_info = self.api_keys[key_hash]

            # Check if key is active and not expired
            if key_info.get('active', False):
                expiry = key_info.get('expires_at')
                if not expiry or time.time() < expiry:
                    return key_info

        return None

    async def _check_authorization(self, config: SecurityConfig, context: SecurityContext, endpoint: str) -> None:
        """Check authorization requirements."""
        if not config.require_authorization or not context.authenticated_user:
            return

        # Check if user has required roles for this endpoint
        required_roles = self._get_required_roles(endpoint)

        if required_roles:
            user_roles = set(context.user_roles)
            required_roles_set = set(required_roles)

            if not user_roles.intersection(required_roles_set):
                raise AuthorizationError(
                    f"Insufficient permissions for {endpoint}",
                    error_code="INSUFFICIENT_PERMISSIONS",
                    context={
                        "endpoint": endpoint,
                        "user_roles": context.user_roles,
                        "required_roles": required_roles
                    }
                )

    def _get_required_roles(self, endpoint: str) -> List[str]:
        """Get required roles for an endpoint."""
        # This would typically be configured in a database or config file
        role_mappings = {
            "/admin/*": ["admin"],
            "/api/forecast/*": ["forecaster", "admin"],
            "/api/tournament/*": ["participant", "admin"],
            "/api/research/*": ["researcher", "forecaster", "admin"]
        }

        for pattern, roles in role_mappings.items():
            if pattern.endswith('*'):
                prefix = pattern[:-1]
                if endpoint.startswith(prefix):
                    return roles
            elif endpoint == pattern:
                return roles

        return []

    async def _check_csrf_protection(self, config: SecurityConfig, context: SecurityContext, headers: Dict[str, str], method: str) -> None:
        """Check CSRF protection."""
        if not config.enable_csrf_protection or method in ['GET', 'HEAD', 'OPTIONS']:
            return

        csrf_token = headers.get('X-CSRF-Token')
        if not csrf_token:
            raise SecurityError(
                "CSRF token required",
                error_code="CSRF_TOKEN_REQUIRED"
            )

        # Validate CSRF token
        if not self._validate_csrf_token(csrf_token, context.client_ip):
            raise SecurityError(
                "Invalid CSRF token",
                error_code="INVALID_CSRF_TOKEN"
            )

    def _validate_csrf_token(self, token: str, client_ip: str) -> bool:
        """Validate CSRF token."""
        # Simple implementation - in production, use more sophisticated approach
        expected_token = self._generate_csrf_token(client_ip)
        return hmac.compare_digest(token, expected_token)

    def _generate_csrf_token(self, client_ip: str) -> str:
        """Generate CSRF token for client."""
        # Simple implementation - in production, include timestamp and user session
        secret = "csrf_secret_key"  # Should be from config
        data = f"{client_ip}:{int(time.time() // 3600)}"  # Valid for 1 hour
        return hmac.new(secret.encode(), data.encode(), hashlib.sha256).hexdigest()

    async def _validate_input(self, config: SecurityConfig, request_data: Dict[str, Any]) -> None:
        """Validate input data."""
        if not config.validate_input:
            return

        # Use validation rules if provided
        if config.validation_rules:
            results = self.input_validator.validate_data(request_data, config.validation_rules)

            # Check for validation errors
            errors = []
            for field_name, result in results.items():
                if not result.is_valid:
                    errors.extend(result.errors)

            if errors:
                raise SecurityError(
                    f"Input validation failed: {'; '.join(errors)}",
                    error_code="INPUT_VALIDATION_FAILED",
                    context={"validation_errors": errors}
                )

        # Check for common attack patterns
        for key, value in request_data.items():
            if isinstance(value, str):
                if self.input_validator.detect_xss_attempt(value):
                    raise SecurityError(
                        f"XSS attempt detected in field '{key}'",
                        error_code="XSS_ATTEMPT_DETECTED",
                        context={"field": key}
                    )

                if self.input_validator.detect_sql_injection_attempt(value):
                    raise SecurityError(
                        f"SQL injection attempt detected in field '{key}'",
                        error_code="SQL_INJECTION_ATTEMPT_DETECTED",
                        context={"field": key}
                    )

    async def _run_custom_checks(self, config: SecurityConfig, context: SecurityContext, request_data: Dict[str, Any]) -> None:
        """Run custom security checks."""
        if not config.custom_checks:
            return

        for check_func in config.custom_checks:
            try:
                if asyncio.iscoroutinefunction(check_func):
                    await check_func(context, request_data)
                else:
                    check_func(context, request_data)
            except Exception as e:
                raise SecurityError(
                    f"Custom security check failed: {str(e)}",
                    error_code="CUSTOM_CHECK_FAILED",
                    context={"check_function": check_func.__name__}
                )

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())

    def add_api_key(self, api_key: str, user_info: Dict[str, Any]) -> None:
        """Add API key for authentication."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.api_keys[key_hash] = user_info
        self.logger.info(f"Added API key for user: {user_info.get('user_id')}")

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash in self.api_keys:
            del self.api_keys[key_hash]
            self.logger.info(f"Revoked API key: {key_hash[:8]}...")
            return True
        return False

    def block_ip(self, ip_address: str) -> None:
        """Block IP address."""
        try:
            # Validate IP address
            ipaddress.ip_address(ip_address)
            self.blocked_ips.add(ip_address)
            self.logger.warning(f"Blocked IP address: {ip_address}")
        except ValueError:
            self.logger.error(f"Invalid IP address: {ip_address}")

    def unblock_ip(self, ip_address: str) -> None:
        """Unblock IP address."""
        self.blocked_ips.discard(ip_address)
        self.logger.info(f"Unblocked IP address: {ip_address}")

    def block_user_agent(self, user_agent: str) -> None:
        """Block user agent."""
        self.blocked_user_agents.add(user_agent)
        self.logger.warning(f"Blocked user agent: {user_agent}")

    def unblock_user_agent(self, user_agent: str) -> None:
        """Unblock user agent."""
        self.blocked_user_agents.discard(user_agent)
        self.logger.info(f"Unblocked user agent: {user_agent}")

    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics."""
        return {
            "blocked_ips_count": len(self.blocked_ips),
            "blocked_user_agents_count": len(self.blocked_user_agents),
            "active_api_keys_count": len(self.api_keys),
            "endpoint_configs_count": len(self.endpoint_configs),
            "csrf_tokens_count": len(self.csrf_tokens)
        }
