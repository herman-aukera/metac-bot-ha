"""Unit tests for SecurityMiddleware."""

import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
import hashlib

from src.infrastructure.security.security_middleware import (
    SecurityMiddleware,
    SecurityConfig,
    SecurityLevel,
    SecurityContext
)
from src.infrastructure.security.input_validator import InputValidator, ValidationRule, ValidationType
from src.infrastructure.security.rate_limiter import RateLimiter
from src.infrastructure.security.audit_logger import AuditLogger
from src.domain.exceptions.infrastructure_exceptions import (
    SecurityError,
    AuthenticationError,
    AuthorizationError,
    RateLimitExceededError
)


class TestSecurityMiddleware:
    """Test SecurityMiddleware functionality."""

    @pytest.fixture
    def mock_input_validator(self):
        validator = Mock(spec=InputValidator)
        validator.validate_data = Mock(return_value={})
        validator.detect_xss_attempt = Mock(return_value=False)
        validator.detect_sql_injection_attempt = Mock(return_value=False)
        return validator

    @pytest.fixture
    def mock_rate_limiter(self):
        limiter = Mock(spec=RateLimiter)
        limiter.middleware_check = AsyncMock()
        return limiter

    @pytest.fixture
    def mock_audit_logger(self):
        logger = Mock(spec=AuditLogger)
        logger.log_security_event = AsyncMock()
        return logger

    @pytest.fixture
    def security_middleware(self, mock_input_validator, mock_rate_limiter, mock_audit_logger):
        return SecurityMiddleware(
            input_validator=mock_input_validator,
            rate_limiter=mock_rate_limiter,
            audit_logger=mock_audit_logger
        )

    def test_add_endpoint_config(self, security_middleware):
        """Test adding endpoint security configuration."""
        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            require_authentication=True,
            enable_rate_limiting=True
        )

        security_middleware.add_endpoint_config("/test/endpoint", config)

        assert "/test/endpoint" in security_middleware.endpoint_configs
        assert security_middleware.endpoint_configs["/test/endpoint"] == config

    def test_get_endpoint_config_exact_match(self, security_middleware):
        """Test getting endpoint config with exact match."""
        config = SecurityConfig(level=SecurityLevel.HIGH)
        security_middleware.add_endpoint_config("/exact/match", config)

        result = security_middleware.get_endpoint_config("/exact/match")

        assert result == config

    def test_get_endpoint_config_pattern_match(self, security_middleware):
        """Test getting endpoint config with pattern matching."""
        config = SecurityConfig(level=SecurityLevel.MEDIUM)
        security_middleware.add_endpoint_config("/api/*", config)

        result = security_middleware.get_endpoint_config("/api/users")

        assert result.level == SecurityLevel.MEDIUM

    def test_get_endpoint_config_default(self, security_middleware):
        """Test getting default endpoint config when no match found."""
        result = security_middleware.get_endpoint_config("/unknown/endpoint")

        assert result.level == SecurityLevel.MEDIUM
        assert result.require_authentication is True

    @pytest.mark.asyncio
    async def test_process_request_success(self, security_middleware, mock_audit_logger):
        """Test successful request processing."""
        # Setup
        endpoint = "/public/test"
        request_data = {"test": "data"}
        headers = {"User-Agent": "test-agent"}
        client_ip = "192.168.1.1"

        # Configure for public endpoint (no auth required)
        config = SecurityConfig(
            level=SecurityLevel.LOW,
            require_authentication=False,
            require_authorization=False,
            enable_rate_limiting=False,
            validate_input=False
        )
        security_middleware.add_endpoint_config("/public/*", config)

        # Execute
        context = await security_middleware.process_request(
            endpoint, request_data, headers, client_ip
        )

        # Verify
        assert isinstance(context, SecurityContext)
        assert context.client_ip == client_ip
        assert context.user_agent == "test-agent"
        assert context.security_level == SecurityLevel.LOW
        mock_audit_logger.log_security_event.assert_called()

    @pytest.mark.asyncio
    async def test_check_blocked_ip(self, security_middleware):
        """Test blocked IP check."""
        # Setup
        security_middleware.block_ip("192.168.1.100")

        endpoint = "/test"
        request_data = {}
        headers = {"User-Agent": "test-agent"}
        client_ip = "192.168.1.100"

        # Execute & Verify
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        assert exc_info.value.error_code == "IP_BLOCKED"

    @pytest.mark.asyncio
    async def test_check_blocked_user_agent(self, security_middleware):
        """Test blocked user agent check."""
        # Setup
        security_middleware.block_user_agent("malicious-bot")

        endpoint = "/test"
        request_data = {}
        headers = {"User-Agent": "malicious-bot"}
        client_ip = "192.168.1.1"

        # Execute & Verify
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        assert exc_info.value.error_code == "USER_AGENT_BLOCKED"

    @pytest.mark.asyncio
    async def test_check_https_requirement(self, security_middleware):
        """Test HTTPS requirement check."""
        # Setup
        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            require_https=True,
            require_authentication=False
        )
        security_middleware.add_endpoint_config("/secure/*", config)

        endpoint = "/secure/test"
        request_data = {}
        headers = {"User-Agent": "test-agent"}  # No HTTPS headers
        client_ip = "192.168.1.1"

        # Execute & Verify
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        assert exc_info.value.error_code == "HTTPS_REQUIRED"

    @pytest.mark.asyncio
    async def test_check_https_requirement_satisfied(self, security_middleware):
        """Test HTTPS requirement check when satisfied."""
        # Setup
        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            require_https=True,
            require_authentication=False,
            enable_rate_limiting=False,
            validate_input=False
        )
        security_middleware.add_endpoint_config("/secure/*", config)

        endpoint = "/secure/test"
        request_data = {}
        headers = {
            "User-Agent": "test-agent",
            "X-Forwarded-Proto": "https"
        }
        client_ip = "192.168.1.1"

        # Execute - should not raise exception
        context = await security_middleware.process_request(
            endpoint, request_data, headers, client_ip
        )

        assert context.security_level == SecurityLevel.HIGH

    @pytest.mark.asyncio
    async def test_check_request_size_limit(self, security_middleware):
        """Test request size limit check."""
        # Setup
        config = SecurityConfig(
            level=SecurityLevel.MEDIUM,
            max_request_size=100,  # Very small limit
            require_authentication=False
        )
        security_middleware.add_endpoint_config("/test", config)

        endpoint = "/test"
        request_data = {"large_data": "x" * 1000}  # Large data
        headers = {"User-Agent": "test-agent"}
        client_ip = "192.168.1.1"

        # Execute & Verify
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        assert exc_info.value.error_code == "REQUEST_TOO_LARGE"

    @pytest.mark.asyncio
    async def test_check_cors_violation(self, security_middleware):
        """Test CORS policy violation."""
        # Setup
        config = SecurityConfig(
            level=SecurityLevel.MEDIUM,
            allowed_origins=["https://allowed.com"],
            require_authentication=False
        )
        security_middleware.add_endpoint_config("/api/*", config)

        endpoint = "/api/test"
        request_data = {}
        headers = {
            "User-Agent": "test-agent",
            "Origin": "https://malicious.com"
        }
        client_ip = "192.168.1.1"

        # Execute & Verify
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        assert exc_info.value.error_code == "CORS_VIOLATION"

    @pytest.mark.asyncio
    async def test_check_rate_limiting(self, security_middleware, mock_rate_limiter):
        """Test rate limiting check."""
        # Setup
        config = SecurityConfig(
            level=SecurityLevel.MEDIUM,
            enable_rate_limiting=True,
            rate_limit_rule="test_rule",
            require_authentication=False,
            validate_input=False
        )
        security_middleware.add_endpoint_config("/api/*", config)

        mock_rate_limiter.middleware_check.side_effect = RateLimitExceededError(
            "Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            context={"retry_after": 60}
        )

        endpoint = "/api/test"
        request_data = {}
        headers = {"User-Agent": "test-agent"}
        client_ip = "192.168.1.1"

        # Execute & Verify
        with pytest.raises(RateLimitExceededError):
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        mock_rate_limiter.middleware_check.assert_called_once_with("test_rule", client_ip)

    @pytest.mark.asyncio
    async def test_check_authentication_missing_key(self, security_middleware):
        """Test authentication check with missing API key."""
        # Setup
        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            require_authentication=True
        )
        security_middleware.add_endpoint_config("/secure/*", config)

        endpoint = "/secure/test"
        request_data = {}
        headers = {"User-Agent": "test-agent"}  # No API key
        client_ip = "192.168.1.1"

        # Execute & Verify
        with pytest.raises(AuthenticationError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        assert exc_info.value.error_code == "AUTHENTICATION_REQUIRED"

    @pytest.mark.asyncio
    async def test_check_authentication_invalid_key(self, security_middleware):
        """Test authentication check with invalid API key."""
        # Setup
        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            require_authentication=True
        )
        security_middleware.add_endpoint_config("/secure/*", config)

        endpoint = "/secure/test"
        request_data = {}
        headers = {
            "User-Agent": "test-agent",
            "X-API-Key": "invalid-key"
        }
        client_ip = "192.168.1.1"

        # Execute & Verify
        with pytest.raises(AuthenticationError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        assert exc_info.value.error_code == "INVALID_CREDENTIALS"

    @pytest.mark.asyncio
    async def test_check_authentication_valid_key(self, security_middleware):
        """Test authentication check with valid API key."""
        # Setup
        api_key = "valid-api-key"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        security_middleware.api_keys[key_hash] = {
            "user_id": "test_user",
            "roles": ["user"],
            "active": True
        }

        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            require_authentication=True,
            require_authorization=False,
            enable_rate_limiting=False,
            validate_input=False
        )
        security_middleware.add_endpoint_config("/secure/*", config)

        endpoint = "/secure/test"
        request_data = {}
        headers = {
            "User-Agent": "test-agent",
            "X-API-Key": api_key
        }
        client_ip = "192.168.1.1"

        # Execute
        context = await security_middleware.process_request(
            endpoint, request_data, headers, client_ip
        )

        # Verify
        assert context.authenticated_user == "test_user"
        assert context.user_roles == ["user"]

    @pytest.mark.asyncio
    async def test_check_authorization_insufficient_permissions(self, security_middleware):
        """Test authorization check with insufficient permissions."""
        # Setup
        api_key = "valid-api-key"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        security_middleware.api_keys[key_hash] = {
            "user_id": "test_user",
            "roles": ["user"],  # Only user role
            "active": True
        }

        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            require_authentication=True,
            require_authorization=True,
            enable_rate_limiting=False,
            validate_input=False
        )
        security_middleware.add_endpoint_config("/admin/*", config)

        endpoint = "/admin/test"  # Requires admin role
        request_data = {}
        headers = {
            "User-Agent": "test-agent",
            "X-API-Key": api_key
        }
        client_ip = "192.168.1.1"

        # Execute & Verify
        with pytest.raises(AuthorizationError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        assert exc_info.value.error_code == "INSUFFICIENT_PERMISSIONS"

    @pytest.mark.asyncio
    async def test_check_csrf_protection_missing_token(self, security_middleware):
        """Test CSRF protection with missing token."""
        # Setup
        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            enable_csrf_protection=True,
            require_authentication=False
        )
        security_middleware.add_endpoint_config("/api/*", config)

        endpoint = "/api/test"
        request_data = {}
        headers = {"User-Agent": "test-agent"}  # No CSRF token
        client_ip = "192.168.1.1"
        method = "POST"

        # Execute & Verify
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip, method
            )

        assert exc_info.value.error_code == "CSRF_TOKEN_REQUIRED"

    @pytest.mark.asyncio
    async def test_check_csrf_protection_invalid_token(self, security_middleware):
        """Test CSRF protection with invalid token."""
        # Setup
        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            enable_csrf_protection=True,
            require_authentication=False
        )
        security_middleware.add_endpoint_config("/api/*", config)

        endpoint = "/api/test"
        request_data = {}
        headers = {
            "User-Agent": "test-agent",
            "X-CSRF-Token": "invalid-token"
        }
        client_ip = "192.168.1.1"
        method = "POST"

        # Execute & Verify
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip, method
            )

        assert exc_info.value.error_code == "INVALID_CSRF_TOKEN"

    @pytest.mark.asyncio
    async def test_check_csrf_protection_get_method_skipped(self, security_middleware):
        """Test CSRF protection is skipped for GET requests."""
        # Setup
        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            enable_csrf_protection=True,
            require_authentication=False,
            enable_rate_limiting=False,
            validate_input=False
        )
        security_middleware.add_endpoint_config("/api/*", config)

        endpoint = "/api/test"
        request_data = {}
        headers = {"User-Agent": "test-agent"}  # No CSRF token
        client_ip = "192.168.1.1"
        method = "GET"

        # Execute - should not raise exception
        context = await security_middleware.process_request(
            endpoint, request_data, headers, client_ip, method
        )

        assert context.security_level == SecurityLevel.HIGH

    @pytest.mark.asyncio
    async def test_validate_input_xss_attempt(self, security_middleware, mock_input_validator):
        """Test input validation with XSS attempt."""
        # Setup
        mock_input_validator.detect_xss_attempt.return_value = True

        config = SecurityConfig(
            level=SecurityLevel.MEDIUM,
            validate_input=True,
            require_authentication=False
        )
        security_middleware.add_endpoint_config("/api/*", config)

        endpoint = "/api/test"
        request_data = {"content": "<script>alert('xss')</script>"}
        headers = {"User-Agent": "test-agent"}
        client_ip = "192.168.1.1"

        # Execute & Verify
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        assert exc_info.value.error_code == "XSS_ATTEMPT_DETECTED"

    @pytest.mark.asyncio
    async def test_validate_input_sql_injection_attempt(self, security_middleware, mock_input_validator):
        """Test input validation with SQL injection attempt."""
        # Setup
        mock_input_validator.detect_sql_injection_attempt.return_value = True

        config = SecurityConfig(
            level=SecurityLevel.MEDIUM,
            validate_input=True,
            require_authentication=False
        )
        security_middleware.add_endpoint_config("/api/*", config)

        endpoint = "/api/test"
        request_data = {"query": "'; DROP TABLE users; --"}
        headers = {"User-Agent": "test-agent"}
        client_ip = "192.168.1.1"

        # Execute & Verify
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        assert exc_info.value.error_code == "SQL_INJECTION_ATTEMPT_DETECTED"

    @pytest.mark.asyncio
    async def test_custom_security_checks(self, security_middleware):
        """Test custom security checks."""
        # Setup custom check function
        def custom_check(context, request_data):
            if request_data.get("forbidden_field"):
                raise SecurityError("Custom check failed", error_code="CUSTOM_CHECK_FAILED")

        config = SecurityConfig(
            level=SecurityLevel.MEDIUM,
            custom_checks=[custom_check],
            require_authentication=False,
            enable_rate_limiting=False,
            validate_input=False
        )
        security_middleware.add_endpoint_config("/api/*", config)

        endpoint = "/api/test"
        request_data = {"forbidden_field": "forbidden_value"}
        headers = {"User-Agent": "test-agent"}
        client_ip = "192.168.1.1"

        # Execute & Verify
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        assert exc_info.value.error_code == "CUSTOM_CHECK_FAILED"

    def test_add_api_key(self, security_middleware):
        """Test adding API key."""
        api_key = "test-api-key"
        user_info = {"user_id": "test_user", "roles": ["user"]}

        security_middleware.add_api_key(api_key, user_info)

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        assert key_hash in security_middleware.api_keys
        assert security_middleware.api_keys[key_hash] == user_info

    def test_revoke_api_key(self, security_middleware):
        """Test revoking API key."""
        api_key = "test-api-key"
        user_info = {"user_id": "test_user", "roles": ["user"]}

        security_middleware.add_api_key(api_key, user_info)
        result = security_middleware.revoke_api_key(api_key)

        assert result is True
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        assert key_hash not in security_middleware.api_keys

    def test_revoke_nonexistent_api_key(self, security_middleware):
        """Test revoking non-existent API key."""
        result = security_middleware.revoke_api_key("nonexistent-key")
        assert result is False

    def test_block_and_unblock_ip(self, security_middleware):
        """Test blocking and unblocking IP addresses."""
        ip_address = "192.168.1.100"

        # Block IP
        security_middleware.block_ip(ip_address)
        assert ip_address in security_middleware.blocked_ips

        # Unblock IP
        security_middleware.unblock_ip(ip_address)
        assert ip_address not in security_middleware.blocked_ips

    def test_block_invalid_ip(self, security_middleware):
        """Test blocking invalid IP address."""
        # Should not raise exception, but should log error
        security_middleware.block_ip("invalid-ip")
        assert "invalid-ip" not in security_middleware.blocked_ips

    def test_block_and_unblock_user_agent(self, security_middleware):
        """Test blocking and unblocking user agents."""
        user_agent = "malicious-bot"

        # Block user agent
        security_middleware.block_user_agent(user_agent)
        assert user_agent in security_middleware.blocked_user_agents

        # Unblock user agent
        security_middleware.unblock_user_agent(user_agent)
        assert user_agent not in security_middleware.blocked_user_agents

    def test_generate_csrf_token(self, security_middleware):
        """Test CSRF token generation."""
        client_ip = "192.168.1.1"
        token = security_middleware._generate_csrf_token(client_ip)

        assert isinstance(token, str)
        assert len(token) > 0

        # Same IP should generate same token (within same hour)
        token2 = security_middleware._generate_csrf_token(client_ip)
        assert token == token2

    def test_validate_csrf_token(self, security_middleware):
        """Test CSRF token validation."""
        client_ip = "192.168.1.1"
        valid_token = security_middleware._generate_csrf_token(client_ip)

        # Valid token should pass
        assert security_middleware._validate_csrf_token(valid_token, client_ip)

        # Invalid token should fail
        assert not security_middleware._validate_csrf_token("invalid-token", client_ip)

    def test_get_required_roles(self, security_middleware):
        """Test getting required roles for endpoints."""
        # Test admin endpoint
        roles = security_middleware._get_required_roles("/admin/users")
        assert "admin" in roles

        # Test API endpoint
        roles = security_middleware._get_required_roles("/api/forecast/submit")
        assert "forecaster" in roles or "admin" in roles

        # Test unknown endpoint
        roles = security_middleware._get_required_roles("/unknown/endpoint")
        assert roles == []

    @pytest.mark.asyncio
    async def test_get_security_metrics(self, security_middleware):
        """Test getting security metrics."""
        # Add some test data
        security_middleware.block_ip("192.168.1.100")
        security_middleware.block_user_agent("bot")
        security_middleware.add_api_key("test-key", {"user_id": "test"})

        metrics = await security_middleware.get_security_metrics()

        assert metrics["blocked_ips_count"] == 1
        assert metrics["blocked_user_agents_count"] == 1
        assert metrics["active_api_keys_count"] == 1
        assert "endpoint_configs_count" in metrics

    def test_default_security_configurations(self, security_middleware):
        """Test that default security configurations are set up."""
        # Check that default configs exist
        auth_config = security_middleware.get_endpoint_config("/auth/login")
        assert auth_config.level == SecurityLevel.HIGH
        assert auth_config.require_authentication is False  # Login endpoint

        admin_config = security_middleware.get_endpoint_config("/admin/test")
        assert admin_config.level == SecurityLevel.CRITICAL
        assert admin_config.require_authentication is True
        assert admin_config.require_authorization is True

        api_config = security_middleware.get_endpoint_config("/api/test")
        assert api_config.level == SecurityLevel.MEDIUM
        assert api_config.require_authentication is True

        public_config = security_middleware.get_endpoint_config("/public/test")
        assert public_config.level == SecurityLevel.LOW
        assert public_config.require_authentication is False


@pytest.mark.asyncio
async def test_security_middleware_integration():
    """Integration test for security middleware with all components."""
    # Setup
    input_validator = InputValidator()
    rate_limiter = RateLimiter()
    audit_logger = Mock(spec=AuditLogger)
    audit_logger.log_security_event = AsyncMock()

    middleware = SecurityMiddleware(
        input_validator=input_validator,
        rate_limiter=rate_limiter,
        audit_logger=audit_logger
    )

    # Add API key
    api_key = "integration-test-key"
    middleware.add_api_key(api_key, {
        "user_id": "integration_user",
        "roles": ["user"],
        "active": True
    })

    # Test successful request
    endpoint = "/api/test"
    request_data = {"message": "Hello, world!"}
    headers = {
        "User-Agent": "test-client",
        "X-API-Key": api_key,
        "X-Forwarded-Proto": "https"
    }
    client_ip = "192.168.1.1"

    context = await middleware.process_request(
        endpoint, request_data, headers, client_ip
    )

    # Verify
    assert context.authenticated_user == "integration_user"
    assert context.user_roles == ["user"]
    assert context.security_level == SecurityLevel.MEDIUM
    assert audit_logger.log_security_event.call_count >= 2  # Start and success events
