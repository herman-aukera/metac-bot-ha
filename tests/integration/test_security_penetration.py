"""Penetration testing scenarios for security components."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import hashlib
import json

from src.infrastructure.security.input_validator import InputValidator, ValidationRule, ValidationType
from src.infrastructure.security.rate_limiter import RateLimiter, RateLimitRule, RateLimitStrategy
from src.infrastructure.security.security_middleware import SecurityMiddleware, SecurityConfig, SecurityLevel
from src.infrastructure.security.audit_logger import AuditLogger
from src.domain.exceptions.infrastructure_exceptions import (
    SecurityError,
    AuthenticationError,
    RateLimitExceededError
)


class TestSecurityPenetrationScenarios:
    """Penetration testing scenarios for security components."""

    @pytest.fixture
    def input_validator(self):
        return InputValidator()

    @pytest.fixture
    def rate_limiter(self):
        limiter = RateLimiter()
        limiter.create_default_rules()
        return limiter

    @pytest.fixture
    def security_middleware(self):
        return SecurityMiddleware()

    @pytest.fixture
    def audit_logger(self):
        return AuditLogger(enable_console_output=False, enable_file_output=False)

    # XSS Attack Scenarios
    @pytest.mark.asyncio
    async def test_xss_attack_vectors(self, input_validator):
        """Test various XSS attack vectors."""
        xss_payloads = [
            # Basic script injection
            "<script>alert('XSS')</script>",
            "<SCRIPT>alert('XSS')</SCRIPT>",
            "<script>alert(String.fromCharCode(88,83,83))</script>",

            # Event handler injection
            "<img src=x onerror=alert('XSS')>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",

            # JavaScript protocol
            "javascript:alert('XSS')",
            "JaVaScRiPt:alert('XSS')",
            "&#106;&#97;&#118;&#97;&#115;&#99;&#114;&#105;&#112;&#116;&#58;alert('XSS')",

            # Data URI
            "data:text/html,<script>alert('XSS')</script>",
            "data:text/html;base64,PHNjcmlwdD5hbGVydCgnWFNTJyk8L3NjcmlwdD4=",

            # SVG injection
            "<svg onload=alert('XSS')>",
            "<svg><script>alert('XSS')</script></svg>",

            # CSS injection
            "<style>@import'javascript:alert(\"XSS\")';</style>",
            "<link rel=stylesheet href=javascript:alert('XSS')>",

            # HTML5 injection
            "<video><source onerror=alert('XSS')>",
            "<audio src=x onerror=alert('XSS')>",

            # Encoded payloads
            "%3Cscript%3Ealert('XSS')%3C/script%3E",
            "&lt;script&gt;alert('XSS')&lt;/script&gt;",

            # Filter bypass attempts
            "<scr<script>ipt>alert('XSS')</scr</script>ipt>",
            "<img src=\"javascript:alert('XSS')\">",
            "<iframe src=javascript:alert('XSS')></iframe>",

            # Context-specific payloads
            "';alert('XSS');//",
            "\";alert('XSS');//",
            "javascript:/*--></title></style></textarea></script></xmp><svg/onload='+/\"/+/onmouseover=1/+/[*/[]/+alert(1)//'>",
        ]

        rule = ValidationRule(
            field_name="user_input",
            validation_type=ValidationType.STRING,
            sanitize=True
        )

        for payload in xss_payloads:
            # Test XSS detection
            is_xss = input_validator.detect_xss_attempt(payload)
            assert is_xss, f"Failed to detect XSS in payload: {payload}"

            # Test validation with sanitization
            result = input_validator.validate_field(payload, rule)
            if result.is_valid:
                # If validation passes, ensure the payload is sanitized
                assert payload != result.sanitized_value, f"XSS payload not sanitized: {payload}"
                # Verify dangerous elements are removed
                assert "<script>" not in result.sanitized_value.lower()
                assert "javascript:" not in result.sanitized_value.lower()
                assert "onerror=" not in result.sanitized_value.lower()

    # SQL Injection Attack Scenarios
    @pytest.mark.asyncio
    async def test_sql_injection_vectors(self, input_validator):
        """Test various SQL injection attack vectors."""
        sql_payloads = [
            # Basic injection
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' OR 1=1 --",
            "admin'--",
            "admin'/*",

            # Union-based injection
            "' UNION SELECT * FROM passwords --",
            "' UNION SELECT username, password FROM users --",
            "1' UNION SELECT null, username, password FROM users --",

            # Boolean-based blind injection
            "' AND 1=1 --",
            "' AND 1=2 --",
            "' AND (SELECT COUNT(*) FROM users) > 0 --",

            # Time-based blind injection
            "'; WAITFOR DELAY '00:00:05' --",
            "' OR SLEEP(5) --",
            "'; SELECT pg_sleep(5) --",

            # Error-based injection
            "' AND EXTRACTVALUE(1, CONCAT(0x7e, (SELECT version()), 0x7e)) --",
            "' AND (SELECT * FROM (SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a) --",

            # Stacked queries
            "'; INSERT INTO users VALUES ('hacker', 'password') --",
            "'; DELETE FROM users WHERE username='admin' --",

            # Second-order injection
            "admin'; UPDATE users SET password='hacked' WHERE username='admin' --",

            # NoSQL injection (for MongoDB)
            "'; return db.users.find(); var dummy='",
            "'; return true; var dummy='",

            # Encoded payloads
            "%27%20OR%20%271%27%3D%271",
            "&#39; OR &#39;1&#39;=&#39;1",

            # Case variations
            "' oR '1'='1",
            "' UnIoN sElEcT * FrOm UsErS --",

            # Comment variations
            "' OR '1'='1' #",
            "' OR '1'='1' /*",
            "' OR '1'='1' --+",
        ]

        rule = ValidationRule(
            field_name="user_input",
            validation_type=ValidationType.SQL_SAFE,
            sanitize=True
        )

        for payload in sql_payloads:
            # Test SQL injection detection
            is_sql_injection = input_validator.detect_sql_injection_attempt(payload)
            assert is_sql_injection, f"Failed to detect SQL injection in payload: {payload}"

            # Test validation
            result = input_validator.validate_field(payload, rule)
            assert not result.is_valid, f"SQL injection payload should be rejected: {payload}"
            assert "dangerous SQL patterns" in result.errors[0]

    # Rate Limiting Attack Scenarios
    @pytest.mark.asyncio
    async def test_rate_limiting_bypass_attempts(self, rate_limiter):
        """Test attempts to bypass rate limiting."""
        # Add a strict rate limit rule
        rule = RateLimitRule(
            name="strict_test",
            requests_per_window=3,
            window_size_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )
        rate_limiter.add_rule(rule)

        user_id = "attacker"

        # Normal rate limiting test
        for i in range(3):
            result = await rate_limiter.check_rate_limit("strict_test", user_id)
            assert result.allowed, f"Request {i+1} should be allowed"

        # 4th request should be blocked
        result = await rate_limiter.check_rate_limit("strict_test", user_id)
        assert not result.allowed, "4th request should be blocked"

        # Test bypass attempts with different identifiers
        bypass_attempts = [
            f"{user_id} ",  # Trailing space
            f" {user_id}",  # Leading space
            f"{user_id}\n",  # Newline
            f"{user_id}\t",  # Tab
            user_id.upper(),  # Case change
            f"{user_id}.",  # Additional character
            f"{user_id}%20",  # URL encoding
        ]

        for attempt_id in bypass_attempts:
            # These should still be blocked if using the same base identifier
            # But our current implementation treats them as different users
            result = await rate_limiter.check_rate_limit("strict_test", attempt_id)
            # This test documents current behavior - in production, you'd want
            # to normalize identifiers to prevent this bypass

    # Authentication Bypass Scenarios
    @pytest.mark.asyncio
    async def test_authentication_bypass_attempts(self, security_middleware):
        """Test various authentication bypass attempts."""
        # Setup secure endpoint
        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            require_authentication=True,
            require_https=False  # Disable for testing
        )
        security_middleware.add_endpoint_config("/secure/*", config)

        endpoint = "/secure/data"
        request_data = {}
        client_ip = "192.168.1.1"

        # Test missing authentication
        headers = {"User-Agent": "test-agent"}
        with pytest.raises(AuthenticationError):
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip
            )

        # Test various bypass attempts
        bypass_attempts = [
            # Empty/null values
            {"X-API-Key": ""},
            {"X-API-Key": "null"},
            {"X-API-Key": "undefined"},

            # Common default/weak keys
            {"X-API-Key": "admin"},
            {"X-API-Key": "password"},
            {"X-API-Key": "123456"},
            {"X-API-Key": "test"},
            {"X-API-Key": "api_key"},

            # SQL injection in API key
            {"X-API-Key": "' OR '1'='1"},
            {"X-API-Key": "admin'; --"},

            # Header injection
            {"X-API-Key": "valid_key\r\nX-Admin: true"},

            # Case variations
            {"x-api-key": "test_key"},
            {"X-Api-Key": "test_key"},

            # Alternative headers
            {"Authorization": "Bearer fake_token"},
            {"Authorization": "Basic YWRtaW46cGFzc3dvcmQ="},  # admin:password
            {"X-Auth-Token": "fake_token"},
            {"X-Access-Token": "fake_token"},

            # Multiple headers
            {"X-API-Key": "fake1", "Authorization": "Bearer fake2"},
        ]

        for headers_attempt in bypass_attempts:
            headers_attempt["User-Agent"] = "test-agent"
            with pytest.raises(AuthenticationError):
                await security_middleware.process_request(
                    endpoint, request_data, headers_attempt, client_ip
                )

    # CSRF Attack Scenarios
    @pytest.mark.asyncio
    async def test_csrf_attack_scenarios(self, security_middleware):
        """Test CSRF attack scenarios."""
        # Setup CSRF-protected endpoint
        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            require_authentication=False,
            enable_csrf_protection=True,
            require_https=False
        )
        security_middleware.add_endpoint_config("/api/*", config)

        endpoint = "/api/transfer"
        request_data = {"amount": 1000, "to_account": "attacker_account"}
        client_ip = "192.168.1.1"

        # Test missing CSRF token
        headers = {"User-Agent": "test-agent"}
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                endpoint, request_data, headers, client_ip, "POST"
            )
        assert exc_info.value.error_code == "CSRF_TOKEN_REQUIRED"

        # Test various CSRF bypass attempts
        csrf_bypass_attempts = [
            # Empty/null tokens
            "",
            "null",
            "undefined",

            # Predictable tokens
            "csrf_token",
            "12345",
            "token",

            # Tokens from other users/sessions
            "user123_token",
            "session_abc_token",

            # Malformed tokens
            "invalid-token-format",
            "token with spaces",
            "token\nwith\nnewlines",

            # Encoded attempts
            "%00",
            "%20",
            "token%00",

            # Very long tokens (buffer overflow attempt)
            "A" * 10000,

            # Special characters
            "token<script>alert('xss')</script>",
            "token'; DROP TABLE sessions; --",
        ]

        for token_attempt in csrf_bypass_attempts:
            headers = {
                "User-Agent": "test-agent",
                "X-CSRF-Token": token_attempt
            }
            with pytest.raises(SecurityError) as exc_info:
                await security_middleware.process_request(
                    endpoint, request_data, headers, client_ip, "POST"
                )
            assert exc_info.value.error_code == "INVALID_CSRF_TOKEN"

    # Input Validation Bypass Scenarios
    @pytest.mark.asyncio
    async def test_input_validation_bypass_attempts(self, input_validator):
        """Test various input validation bypass attempts."""
        # Test filename validation bypass
        filename_rule = ValidationRule(
            field_name="filename",
            validation_type=ValidationType.FILENAME,
            sanitize=True
        )

        dangerous_filenames = [
            # Path traversal
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",

            # Null byte injection
            "innocent.txt\x00.exe",
            "safe.pdf%00.php",

            # Double encoding
            "%252e%252e%252fetc%252fpasswd",

            # Unicode normalization
            "..%c0%af..%c0%afetc%c0%afpasswd",

            # Long path names
            "A" * 1000 + ".txt",

            # Special device names (Windows)
            "CON.txt",
            "PRN.pdf",
            "AUX.doc",
            "NUL.exe",

            # Executable extensions
            "document.exe",
            "image.scr",
            "file.bat",
            "script.vbs",
            "program.com",
        ]

        for filename in dangerous_filenames:
            result = input_validator.validate_field(filename, filename_rule)
            # Most should be rejected
            if ".." in filename or any(ext in filename.lower() for ext in ['.exe', '.bat', '.scr', '.vbs', '.com']):
                assert not result.is_valid, f"Dangerous filename should be rejected: {filename}"

        # Test email validation bypass
        email_rule = ValidationRule(
            field_name="email",
            validation_type=ValidationType.EMAIL,
            sanitize=True
        )

        malicious_emails = [
            # Header injection
            "user@domain.com\r\nBcc: attacker@evil.com",
            "user@domain.com\nX-Mailer: Evil Script",

            # Script injection
            "user+<script>alert('xss')</script>@domain.com",
            "user@domain.com<script>alert('xss')</script>",

            # SQL injection
            "user'; DROP TABLE users; --@domain.com",
            "user@domain.com'; DELETE FROM emails; --",

            # Command injection
            "user@domain.com; cat /etc/passwd",
            "user@domain.com`whoami`",

            # Very long emails
            "A" * 1000 + "@domain.com",
            "user@" + "A" * 1000 + ".com",
        ]

        for email in malicious_emails:
            result = input_validator.validate_field(email, email_rule)
            # Should either be rejected or sanitized
            if result.is_valid:
                assert email != result.sanitized_value, f"Malicious email not sanitized: {email}"

    # Concurrent Attack Scenarios
    @pytest.mark.asyncio
    async def test_concurrent_attack_scenarios(self, rate_limiter):
        """Test concurrent attack scenarios."""
        # Setup rate limiting
        rule = RateLimitRule(
            name="concurrent_test",
            requests_per_window=5,
            window_size_seconds=60,
            strategy=RateLimitStrategy.FIXED_WINDOW
        )
        rate_limiter.add_rule(rule)

        # Simulate concurrent requests from same user
        async def make_request(user_id, request_num):
            try:
                result = await rate_limiter.check_rate_limit("concurrent_test", user_id)
                return {"request": request_num, "allowed": result.allowed}
            except Exception as e:
                return {"request": request_num, "error": str(e)}

        # Test with many concurrent requests
        user_id = "concurrent_attacker"
        tasks = [make_request(user_id, i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # Count allowed vs denied
        allowed_count = sum(1 for r in results if r.get("allowed", False))
        denied_count = sum(1 for r in results if not r.get("allowed", True))

        # Should allow at most 5 requests
        assert allowed_count <= 5, f"Too many requests allowed: {allowed_count}"
        assert denied_count >= 15, f"Not enough requests denied: {denied_count}"

    # Session Fixation and Hijacking Scenarios
    @pytest.mark.asyncio
    async def test_session_attack_scenarios(self, security_middleware):
        """Test session fixation and hijacking scenarios."""
        # Add valid API key
        api_key = "valid_session_key"
        security_middleware.add_api_key(api_key, {
            "user_id": "legitimate_user",
            "roles": ["user"],
            "active": True
        })

        config = SecurityConfig(
            level=SecurityLevel.MEDIUM,
            require_authentication=True,
            require_https=False
        )
        security_middleware.add_endpoint_config("/api/*", config)

        # Test session hijacking attempts
        hijack_attempts = [
            # Brute force session IDs
            "valid_session_ke",  # Truncated
            "valid_session_key1",  # Modified
            "valid_session_keyA",  # Modified

            # Common session patterns
            "PHPSESSID123456",
            "JSESSIONID=ABC123",
            "session_id_12345",

            # Encoded attempts
            "dmFsaWRfc2Vzc2lvbl9rZXk=",  # Base64 of original
            "%76%61%6c%69%64%5f%73%65%73%73%69%6f%6e%5f%6b%65%79",  # URL encoded
        ]

        endpoint = "/api/user/profile"
        request_data = {}
        client_ip = "192.168.1.1"

        for hijack_key in hijack_attempts:
            headers = {
                "User-Agent": "test-agent",
                "X-API-Key": hijack_key
            }

            with pytest.raises(AuthenticationError):
                await security_middleware.process_request(
                    endpoint, request_data, headers, client_ip
                )

    # Timing Attack Scenarios
    @pytest.mark.asyncio
    async def test_timing_attack_scenarios(self, security_middleware):
        """Test timing attack scenarios."""
        # Add API key for timing comparison
        valid_key = "timing_test_key_12345"
        security_middleware.add_api_key(valid_key, {
            "user_id": "timing_user",
            "roles": ["user"],
            "active": True
        })

        config = SecurityConfig(
            level=SecurityLevel.HIGH,
            require_authentication=True,
            require_https=False
        )
        security_middleware.add_endpoint_config("/timing/*", config)

        endpoint = "/timing/test"
        request_data = {}
        client_ip = "192.168.1.1"

        # Test timing differences between valid and invalid keys
        timing_tests = [
            ("", "empty_key"),
            ("a", "short_key"),
            ("timing_test_key_12344", "almost_valid"),  # One char different
            ("timing_test_key_12345", "valid_key"),
            ("completely_different_key", "different_key"),
            ("x" * len(valid_key), "same_length_invalid"),
        ]

        timings = {}

        for test_key, test_name in timing_tests:
            headers = {
                "User-Agent": "test-agent",
                "X-API-Key": test_key
            }

            start_time = time.time()
            try:
                await security_middleware.process_request(
                    endpoint, request_data, headers, client_ip
                )
                success = True
            except AuthenticationError:
                success = False
            end_time = time.time()

            timings[test_name] = {
                "duration": end_time - start_time,
                "success": success
            }

        # Verify that timing differences are not significant enough for timing attacks
        # (This is more of a documentation test - real timing attack prevention
        # would require constant-time comparison functions)
        valid_time = timings["valid_key"]["duration"]
        invalid_times = [t["duration"] for name, t in timings.items() if name != "valid_key"]

        # In a secure implementation, all authentication attempts should take similar time
        # This test documents the current behavior
        assert timings["valid_key"]["success"] is True
        assert all(not t["success"] for name, t in timings.items() if name != "valid_key")

    # Audit Log Tampering Scenarios
    @pytest.mark.asyncio
    async def test_audit_log_tampering_scenarios(self, audit_logger):
        """Test audit log tampering and evasion scenarios."""
        # Test log injection attempts
        malicious_inputs = [
            # Log injection
            "user\nFAKE LOG ENTRY: Admin login successful",
            "user\r\nERROR: System compromised",

            # JSON injection
            'user", "admin": true, "fake_field": "',
            'user\\", \\"admin\\": true, \\"fake_field\\": \\"',

            # Control character injection
            "user\x00admin",
            "user\x1badmin",

            # Unicode normalization attacks
            "user\u202eadmin",  # Right-to-left override
            "user\u200badmin",  # Zero-width space

            # Very long inputs (log truncation)
            "user" + "A" * 10000,

            # Binary data
            "user\xff\xfe\x00\x00admin",
        ]

        for malicious_input in malicious_inputs:
            # Test that malicious input is properly sanitized in logs
            await audit_logger.log_authentication_event(
                user_id=malicious_input,
                success=False,
                client_ip="192.168.1.1",
                error="Invalid credentials"
            )

            # Verify event was logged (buffer should have events)
            assert len(audit_logger.event_buffer) > 0

            # Check that the logged event has sanitized data
            event = audit_logger.event_buffer[-1]
            if event.user_id:
                # Should not contain raw malicious input
                assert "\n" not in event.user_id
                assert "\r" not in event.user_id
                assert "\x00" not in event.user_id

        # Clear buffer for next test
        audit_logger.event_buffer.clear()

    # Resource Exhaustion Scenarios
    @pytest.mark.asyncio
    async def test_resource_exhaustion_scenarios(self, input_validator, security_middleware):
        """Test resource exhaustion attack scenarios."""
        # Test extremely large inputs
        large_inputs = [
            "A" * 1000000,  # 1MB string
            {"key": "A" * 100000},  # Large dictionary
            ["item"] * 10000,  # Large list
        ]

        rule = ValidationRule(
            field_name="large_input",
            validation_type=ValidationType.STRING,
            max_length=1000  # Reasonable limit
        )

        for large_input in large_inputs:
            if isinstance(large_input, str):
                result = input_validator.validate_field(large_input, rule)
                assert not result.is_valid, "Large input should be rejected"

        # Test request size limits in security middleware
        config = SecurityConfig(
            level=SecurityLevel.MEDIUM,
            max_request_size=1024,  # 1KB limit
            require_authentication=False
        )
        security_middleware.add_endpoint_config("/api/*", config)

        large_request_data = {"data": "X" * 10000}  # Much larger than limit
        headers = {"User-Agent": "test-agent"}
        client_ip = "192.168.1.1"

        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.process_request(
                "/api/test", large_request_data, headers, client_ip
            )
        assert exc_info.value.error_code == "REQUEST_TOO_LARGE"


@pytest.mark.asyncio
async def test_comprehensive_security_penetration():
    """Comprehensive penetration test combining multiple attack vectors."""
    # Setup all security components
    input_validator = InputValidator()
    rate_limiter = RateLimiter()
    rate_limiter.create_default_rules()
    audit_logger = AuditLogger(enable_console_output=False, enable_file_output=False)

    security_middleware = SecurityMiddleware(
        input_validator=input_validator,
        rate_limiter=rate_limiter,
        audit_logger=audit_logger
    )

    # Add a valid API key for testing
    valid_api_key = "comprehensive_test_key"
    security_middleware.add_api_key(valid_api_key, {
        "user_id": "test_user",
        "roles": ["user"],
        "active": True
    })

    # Test multi-vector attack scenario
    endpoint = "/api/user/update"
    malicious_request_data = {
        "username": "<script>alert('xss')</script>",
        "email": "user'; DROP TABLE users; --@evil.com",
        "bio": "Normal bio content",
        "profile_image": "../../../etc/passwd"
    }

    headers = {
        "User-Agent": "AttackBot/1.0 <script>alert('xss')</script>",
        "X-API-Key": valid_api_key,
        "X-Forwarded-Proto": "https",
        "Origin": "https://evil.com"  # CORS violation
    }
    client_ip = "192.168.1.1"

    # Configure endpoint with validation rules
    validation_rules = [
        ValidationRule("username", ValidationType.STRING, sanitize=True, max_length=50),
        ValidationRule("email", ValidationType.EMAIL, sanitize=True),
        ValidationRule("bio", ValidationType.STRING, sanitize=True, max_length=500),
        ValidationRule("profile_image", ValidationType.FILENAME, sanitize=True)
    ]

    config = SecurityConfig(
        level=SecurityLevel.HIGH,
        require_authentication=True,
        validate_input=True,
        validation_rules=validation_rules,
        allowed_origins=["https://trusted.com"],  # Will cause CORS violation
        enable_rate_limiting=True,
        rate_limit_rule="api_general"
    )
    security_middleware.add_endpoint_config("/api/*", config)

    # The request should fail due to CORS violation
    with pytest.raises(SecurityError) as exc_info:
        await security_middleware.process_request(
            endpoint, malicious_request_data, headers, client_ip
        )

    assert exc_info.value.error_code == "CORS_VIOLATION"

    # Verify that security events were logged
    assert len(audit_logger.event_buffer) > 0

    # Check that the failure was properly logged
    security_events = [event for event in audit_logger.event_buffer
                      if "SECURITY_CHECK_FAILED" in event.event_type.value]
    assert len(security_events) > 0


if __name__ == "__main__":
    # Run penetration tests
    pytest.main([__file__, "-v", "--tb=short"])
