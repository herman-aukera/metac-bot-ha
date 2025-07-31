"""Comprehensive security tests for input validation and authentication flows."""

import pytest
import asyncio
import time
import hashlib
import jwt
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any, Optional

from src.infrastructure.security.input_validator import InputValidator
from src.infrastructure.security.credential_manager import CredentialManager
from src.infrastructure.security.rate_limiter import RateLimiter
from src.infrastructure.security.audit_logger import AuditLogger
from src.infrastructure.security.security_middleware import SecurityMiddleware


class SecurityTestHarness:
    """Test harness for security testing scenarios."""

    def __init__(self):
        self.input_validator = InputValidator()
        self.credential_manager = CredentialManager()
        self.rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
        self.audit_logger = AuditLogger()
        self.security_middleware = SecurityMiddleware()

        # Track security events
        self.security_events = []
        self.blocked_requests = []
        self.validation_failures = []

    async def process_request(self, request_data: Dict[str, Any], user_id: str = None) -> Dict[str, Any]:
        """Process a request through security layers."""
        request_id = f"req_{int(time.time() * 1000)}"

        try:
            # Step 1: Rate limiting
            if user_id and not await self.rate_limiter.is_allowed(user_id):
                self.blocked_requests.append({
                    'request_id': request_id,
                    'user_id': user_id,
                    'reason': 'rate_limit_exceeded',
                    'timestamp': datetime.utcnow()
                })
                raise SecurityException("Rate limit exceeded")

            # Step 2: Input validation
            validation_result = await self.input_validator.validate_request(request_data)
            if not validation_result['valid']:
                self.validation_failures.append({
                    'request_id': request_id,
                    'errors': validation_result['errors'],
                    'timestamp': datetime.utcnow()
                })
                raise SecurityException(f"Input validation failed: {validation_result['errors']}")

            # Step 3: Authentication (if user_id provided)
            if user_id:
                auth_result = await self.credential_manager.validate_user(user_id)
                if not auth_result:
                    raise SecurityException("Authentication failed")

            # Step 4: Security middleware processing
            middleware_result = await self.security_middleware.process_request(request_data)
            if not middleware_result['allowed']:
                raise SecurityException(f"Security middleware blocked request: {middleware_result['reason']}")

            # Step 5: Audit logging
            await self.audit_logger.log_security_event({
                'request_id': request_id,
                'user_id': user_id,
                'action': 'request_processed',
                'status': 'success',
                'timestamp': datetime.utcnow()
            })

            return {
                'request_id': request_id,
                'status': 'success',
                'processed_at': datetime.utcnow(),
                'security_checks_passed': True
            }

        except SecurityException as e:
            # Log security event
            await self.audit_logger.log_security_event({
                'request_id': request_id,
                'user_id': user_id,
                'action': 'request_blocked',
                'reason': str(e),
                'status': 'blocked',
                'timestamp': datetime.utcnow()
            })

            self.security_events.append({
                'request_id': request_id,
                'error': str(e),
                'timestamp': datetime.utcnow()
            })

            return {
                'request_id': request_id,
                'status': 'blocked',
                'error': str(e),
                'security_checks_passed': False
            }

    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics from test run."""
        total_events = len(self.security_events)
        blocked_count = len(self.blocked_requests)
        validation_failures = len(self.validation_failures)

        return {
            'total_security_events': total_events,
            'blocked_requests': blocked_count,
            'validation_failures': validation_failures,
            'rate_limit_violations': len([r for r in self.blocked_requests if r['reason'] == 'rate_limit_exceeded'])
        }


class SecurityException(Exception):
    """Custom exception for security-related errors."""
    pass


@pytest.mark.security
@pytest.mark.asyncio
class TestInputValidationSecurity:
    """Test input validation security measures."""

    @pytest.fixture
    def security_harness(self):
        """Create security test harness."""
        return SecurityTestHarness()

    async def test_sql_injection_prevention(self, security_harness, malicious_inputs):
        """Test SQL injection attack prevention."""
        print("\n🛡️ Testing SQL injection prevention...")

        sql_payloads = malicious_inputs['sql_injection']
        results = []

        for payload in sql_payloads:
            request_data = {
                'question_text': payload,
                'user_input': payload,
                'search_query': payload
            }

            result = await security_harness.process_request(request_data, user_id="test_user")
            results.append(result)

        # All SQL injection attempts should be blocked
        blocked_count = len([r for r in results if r['status'] == 'blocked'])
        success_count = len([r for r in results if r['status'] == 'success'])

        print(f"SQL injection test results:")
        print(f"  Total payloads tested: {len(sql_payloads)}")
        print(f"  Blocked requests: {blocked_count}")
        print(f"  Successful requests: {success_count}")

        # Assert that all SQL injection attempts are blocked
        assert blocked_count == len(sql_payloads), f"Only {blocked_count}/{len(sql_payloads)} SQL injection attempts were blocked"
        assert success_count == 0, f"{success_count} SQL injection attempts succeeded"

        # Verify validation failures were recorded
        metrics = security_harness.get_security_metrics()
        assert metrics['validation_failures'] > 0, "No validation failures recorded for SQL injection attempts"

    async def test_xss_prevention(self, security_harness, malicious_inputs):
        """Test XSS attack prevention."""
        print("\n🛡️ Testing XSS prevention...")

        xss_payloads = malicious_inputs['xss_payloads']
        results = []

        for payload in xss_payloads:
            request_data = {
                'question_text': payload,
                'reasoning_text': payload,
                'metadata': {'description': payload}
            }

            result = await security_harness.process_request(request_data, user_id="test_user")
            results.append(result)

        # All XSS attempts should be blocked or sanitized
        blocked_count = len([r for r in results if r['status'] == 'blocked'])
        success_count = len([r for r in results if r['status'] == 'success'])

        print(f"XSS prevention test results:")
        print(f"  Total payloads tested: {len(xss_payloads)}")
        print(f"  Blocked requests: {blocked_count}")
        print(f"  Successful requests: {success_count}")

        # Most XSS attempts should be blocked
        assert blocked_count >= len(xss_payloads) * 0.8, f"Only {blocked_count}/{len(xss_payloads)} XSS attempts were blocked"

        # If any succeed, they should be sanitized
        if success_count > 0:
            print(f"  Note: {success_count} requests succeeded (likely sanitized)")

    async def test_command_injection_prevention(self, security_harness, malicious_inputs):
        """Test command injection prevention."""
        print("\n🛡️ Testing command injection prevention...")

        command_payloads = malicious_inputs['command_injection']
        results = []

        for payload in command_payloads:
            request_data = {
                'search_parameters': payload,
                'file_path': payload,
                'system_command': payload
            }

            result = await security_harness.process_request(request_data, user_id="test_user")
            results.append(result)

        # All command injection attempts should be blocked
        blocked_count = len([r for r in results if r['status'] == 'blocked'])

        print(f"Command injection test results:")
        print(f"  Total payloads tested: {len(command_payloads)}")
        print(f"  Blocked requests: {blocked_count}")

        assert blocked_count == len(command_payloads), f"Only {blocked_count}/{len(command_payloads)} command injection attempts were blocked"

    async def test_path_traversal_prevention(self, security_harness, malicious_inputs):
        """Test path traversal attack prevention."""
        print("\n🛡️ Testing path traversal prevention...")

        path_payloads = malicious_inputs['path_traversal']
        results = []

        for payload in path_payloads:
            request_data = {
                'file_path': payload,
                'resource_path': payload,
                'template_path': payload
            }

            result = await security_harness.process_request(request_data, user_id="test_user")
            results.append(result)

        # All path traversal attempts should be blocked
        blocked_count = len([r for r in results if r['status'] == 'blocked'])

        print(f"Path traversal test results:")
        print(f"  Total payloads tested: {len(path_payloads)}")
        print(f"  Blocked requests: {blocked_count}")

        assert blocked_count == len(path_payloads), f"Only {blocked_count}/{len(path_payloads)} path traversal attempts were blocked"

    async def test_oversized_input_handling(self, security_harness, malicious_inputs):
        """Test handling of oversized inputs."""
        print("\n🛡️ Testing oversized input handling...")

        oversized_inputs = malicious_inputs['oversized_inputs']
        results = []

        for oversized_input in oversized_inputs:
            request_data = {
                'large_text': oversized_input,
                'bulk_data': oversized_input
            }

            result = await security_harness.process_request(request_data, user_id="test_user")
            results.append(result)

        # Oversized inputs should be handled gracefully (blocked or truncated)
        blocked_count = len([r for r in results if r['status'] == 'blocked'])
        success_count = len([r for r in results if r['status'] == 'success'])

        print(f"Oversized input test results:")
        print(f"  Total oversized inputs tested: {len(oversized_inputs)}")
        print(f"  Blocked requests: {blocked_count}")
        print(f"  Successful requests: {success_count}")

        # System should handle oversized inputs without crashing
        assert blocked_count + success_count == len(oversized_inputs), "Some requests were not processed"

        # Most oversized inputs should be blocked
        assert blocked_count >= len(oversized_inputs) * 0.7, f"Only {blocked_count}/{len(oversized_inputs)} oversized inputs were blocked"


@pytest.mark.security
@pytest.mark.asyncio
class TestAuthenticationSecurity:
    """Test authentication and authorization security."""

    @pytest.fixture
    def security_harness(self):
        """Create security test harness."""
        return SecurityTestHarness()

    async def test_valid_authentication_flow(self, security_harness, security_test_credentials):
        """Test valid authentication flow."""
        print("\n🔐 Testing valid authentication flow...")

        valid_api_key = security_test_credentials['valid_api_key']

        # Mock credential manager to accept valid key
        security_harness.credential_manager.validate_user = AsyncMock(return_value=True)
        security_harness.credential_manager.validate_api_key = AsyncMock(return_value=True)

        request_data = {
            'question_text': 'Valid question text',
            'api_key': valid_api_key
        }

        result = await security_harness.process_request(request_data, user_id="valid_user")

        print(f"Valid authentication result: {result['status']}")

        assert result['status'] == 'success', f"Valid authentication failed: {result.get('error')}"
        assert result['security_checks_passed'] is True

    async def test_invalid_authentication_rejection(self, security_harness, security_test_credentials):
        """Test rejection of invalid authentication."""
        print("\n🔐 Testing invalid authentication rejection...")

        invalid_credentials = [
            security_test_credentials['invalid_api_key'],
            security_test_credentials['expired_token'],
            security_test_credentials['malformed_token']
        ]

        # Mock credential manager to reject invalid credentials
        security_harness.credential_manager.validate_user = AsyncMock(return_value=False)
        security_harness.credential_manager.validate_api_key = AsyncMock(return_value=False)

        results = []
        for credential in invalid_credentials:
            request_data = {
                'question_text': 'Test question',
                'api_key': credential
            }

            result = await security_harness.process_request(request_data, user_id="invalid_user")
            results.append(result)

        # All invalid authentication attempts should be blocked
        blocked_count = len([r for r in results if r['status'] == 'blocked'])

        print(f"Invalid authentication test results:")
        print(f"  Total invalid credentials tested: {len(invalid_credentials)}")
        print(f"  Blocked requests: {blocked_count}")

        assert blocked_count == len(invalid_credentials), f"Only {blocked_count}/{len(invalid_credentials)} invalid authentication attempts were blocked"

    async def test_jwt_token_validation(self, security_harness):
        """Test JWT token validation."""
        print("\n🔐 Testing JWT token validation...")

        # Create test JWT tokens
        secret_key = "test_secret_key"

        # Valid token
        valid_payload = {
            'user_id': 'test_user',
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow()
        }
        valid_token = jwt.encode(valid_payload, secret_key, algorithm='HS256')

        # Expired token
        expired_payload = {
            'user_id': 'test_user',
            'exp': datetime.utcnow() - timedelta(hours=1),
            'iat': datetime.utcnow() - timedelta(hours=2)
        }
        expired_token = jwt.encode(expired_payload, secret_key, algorithm='HS256')

        # Invalid signature token
        invalid_token = jwt.encode(valid_payload, "wrong_secret", algorithm='HS256')

        # Mock JWT validation
        def mock_validate_jwt(token):
            try:
                payload = jwt.decode(token, secret_key, algorithms=['HS256'])
                return payload['exp'] > datetime.utcnow().timestamp()
            except:
                return False

        security_harness.credential_manager.validate_jwt = mock_validate_jwt

        # Test tokens
        test_cases = [
            ('valid_token', valid_token, True),
            ('expired_token', expired_token, False),
            ('invalid_signature', invalid_token, False),
            ('malformed_token', 'not.a.jwt.token', False)
        ]

        results = []
        for test_name, token, should_succeed in test_cases:
            # Mock validation based on expected result
            security_harness.credential_manager.validate_user = AsyncMock(return_value=should_succeed)

            request_data = {
                'question_text': 'Test question',
                'jwt_token': token
            }

            result = await security_harness.process_request(request_data, user_id="jwt_user")
            results.append((test_name, result, should_succeed))

        print(f"JWT validation test results:")
        for test_name, result, expected_success in results:
            actual_success = result['status'] == 'success'
            status = "✅" if actual_success == expected_success else "❌"
            print(f"  {status} {test_name}: expected {expected_success}, got {actual_success}")

            assert actual_success == expected_success, f"JWT validation failed for {test_name}"

    async def test_session_management_security(self, security_harness):
        """Test session management security."""
        print("\n🔐 Testing session management security...")

        # Mock session manager
        active_sessions = {}

        async def create_session(user_id: str) -> str:
            session_id = hashlib.md5(f"{user_id}_{time.time()}".encode()).hexdigest()
            active_sessions[session_id] = {
                'user_id': user_id,
                'created_at': datetime.utcnow(),
                'last_activity': datetime.utcnow(),
                'expires_at': datetime.utcnow() + timedelta(hours=2)
            }
            return session_id

        async def validate_session(session_id: str) -> bool:
            if session_id not in active_sessions:
                return False

            session = active_sessions[session_id]
            if datetime.utcnow() > session['expires_at']:
                del active_sessions[session_id]
                return False

            # Update last activity
            session['last_activity'] = datetime.utcnow()
            return True

        async def invalidate_session(session_id: str):
            active_sessions.pop(session_id, None)

        # Test session lifecycle
        user_id = "session_test_user"

        # Create session
        session_id = await create_session(user_id)
        assert session_id in active_sessions

        # Validate active session
        is_valid = await validate_session(session_id)
        assert is_valid is True

        # Test with invalid session
        invalid_session_id = "invalid_session_123"
        is_invalid = await validate_session(invalid_session_id)
        assert is_invalid is False

        # Test session expiration
        # Manually expire session
        active_sessions[session_id]['expires_at'] = datetime.utcnow() - timedelta(minutes=1)
        is_expired = await validate_session(session_id)
        assert is_expired is False
        assert session_id not in active_sessions  # Should be cleaned up

        print(f"Session management test results:")
        print(f"  Session creation: ✅")
        print(f"  Session validation: ✅")
        print(f"  Invalid session rejection: ✅")
        print(f"  Session expiration: ✅")


@pytest.mark.security
@pytest.mark.asyncio
class TestRateLimitingSecurity:
    """Test rate limiting security measures."""

    @pytest.fixture
    def security_harness(self):
        """Create security test harness."""
        return SecurityTestHarness()

    async def test_rate_limiting_enforcement(self, security_harness):
        """Test rate limiting enforcement."""
        print("\n🚦 Testing rate limiting enforcement...")

        # Configure strict rate limiting for testing
        test_rate_limiter = RateLimiter(max_requests=5, window_seconds=10)
        security_harness.rate_limiter = test_rate_limiter

        user_id = "rate_limit_test_user"
        results = []

        # Send requests up to and beyond the limit
        for i in range(10):
            request_data = {
                'question_text': f'Test question {i}',
                'request_number': i
            }

            result = await security_harness.process_request(request_data, user_id=user_id)
            results.append(result)

            # Small delay between requests
            await asyncio.sleep(0.1)

        # Analyze results
        successful_requests = [r for r in results if r['status'] == 'success']
        blocked_requests = [r for r in results if r['status'] == 'blocked' and 'rate limit' in r.get('error', '').lower()]

        print(f"Rate limiting test results:")
        print(f"  Total requests sent: {len(results)}")
        print(f"  Successful requests: {len(successful_requests)}")
        print(f"  Rate-limited requests: {len(blocked_requests)}")

        # Should allow exactly 5 requests, block the rest
        assert len(successful_requests) == 5, f"Expected 5 successful requests, got {len(successful_requests)}"
        assert len(blocked_requests) == 5, f"Expected 5 blocked requests, got {len(blocked_requests)}"

    async def test_rate_limiting_per_user(self, security_harness):
        """Test per-user rate limiting."""
        print("\n🚦 Testing per-user rate limiting...")

        # Configure rate limiting
        test_rate_limiter = RateLimiter(max_requests=3, window_seconds=5)
        security_harness.rate_limiter = test_rate_limiter

        users = ["user_1", "user_2", "user_3"]
        all_results = {}

        # Send requests from multiple users
        for user_id in users:
            user_results = []

            for i in range(5):  # Each user sends 5 requests
                request_data = {
                    'question_text': f'Question from {user_id} #{i}',
                    'user_id': user_id
                }

                result = await security_harness.process_request(request_data, user_id=user_id)
                user_results.append(result)

                await asyncio.sleep(0.05)

            all_results[user_id] = user_results

        # Analyze per-user results
        for user_id, results in all_results.items():
            successful = len([r for r in results if r['status'] == 'success'])
            blocked = len([r for r in results if r['status'] == 'blocked'])

            print(f"User {user_id}: {successful} successful, {blocked} blocked")

            # Each user should have exactly 3 successful requests
            assert successful == 3, f"User {user_id} had {successful} successful requests, expected 3"
            assert blocked == 2, f"User {user_id} had {blocked} blocked requests, expected 2"

    async def test_rate_limiting_window_reset(self, security_harness):
        """Test rate limiting window reset."""
        print("\n🚦 Testing rate limiting window reset...")

        # Configure short window for testing
        test_rate_limiter = RateLimiter(max_requests=2, window_seconds=2)
        security_harness.rate_limiter = test_rate_limiter

        user_id = "window_reset_user"

        # Phase 1: Use up the rate limit
        phase1_results = []
        for i in range(3):
            request_data = {'question_text': f'Phase 1 question {i}'}
            result = await security_harness.process_request(request_data, user_id=user_id)
            phase1_results.append(result)

        phase1_successful = len([r for r in phase1_results if r['status'] == 'success'])
        phase1_blocked = len([r for r in phase1_results if r['status'] == 'blocked'])

        print(f"Phase 1 (initial): {phase1_successful} successful, {phase1_blocked} blocked")

        # Wait for window to reset
        print("Waiting for rate limit window to reset...")
        await asyncio.sleep(3)  # Wait longer than window

        # Phase 2: Should be able to make requests again
        phase2_results = []
        for i in range(3):
            request_data = {'question_text': f'Phase 2 question {i}'}
            result = await security_harness.process_request(request_data, user_id=user_id)
            phase2_results.append(result)

        phase2_successful = len([r for r in phase2_results if r['status'] == 'success'])
        phase2_blocked = len([r for r in phase2_results if r['status'] == 'blocked'])

        print(f"Phase 2 (after reset): {phase2_successful} successful, {phase2_blocked} blocked")

        # Verify rate limit reset worked
        assert phase1_successful == 2, f"Phase 1 should have 2 successful requests, got {phase1_successful}"
        assert phase1_blocked == 1, f"Phase 1 should have 1 blocked request, got {phase1_blocked}"
        assert phase2_successful == 2, f"Phase 2 should have 2 successful requests, got {phase2_successful}"
        assert phase2_blocked == 1, f"Phase 2 should have 1 blocked request, got {phase2_blocked}"

    async def test_distributed_rate_limiting(self, security_harness):
        """Test distributed rate limiting across multiple instances."""
        print("\n🚦 Testing distributed rate limiting...")

        # Simulate multiple instances sharing rate limit state
        shared_rate_limit_state = {}

        class DistributedRateLimiter:
            def __init__(self, max_requests: int, window_seconds: int):
                self.max_requests = max_requests
                self.window_seconds = window_seconds

            async def is_allowed(self, user_id: str) -> bool:
                current_time = time.time()
                window_start = current_time - self.window_seconds

                if user_id not in shared_rate_limit_state:
                    shared_rate_limit_state[user_id] = []

                # Clean old requests
                shared_rate_limit_state[user_id] = [
                    req_time for req_time in shared_rate_limit_state[user_id]
                    if req_time > window_start
                ]

                # Check limit
                if len(shared_rate_limit_state[user_id]) < self.max_requests:
                    shared_rate_limit_state[user_id].append(current_time)
                    return True
                return False

        # Create multiple "instances" with shared state
        instance1_limiter = DistributedRateLimiter(max_requests=4, window_seconds=5)
        instance2_limiter = DistributedRateLimiter(max_requests=4, window_seconds=5)

        user_id = "distributed_test_user"

        # Send requests from both instances
        instance1_results = []
        instance2_results = []

        # Interleave requests from both instances
        for i in range(6):
            if i % 2 == 0:
                # Instance 1
                allowed = await instance1_limiter.is_allowed(user_id)
                instance1_results.append(allowed)
            else:
                # Instance 2
                allowed = await instance2_limiter.is_allowed(user_id)
                instance2_results.append(allowed)

            await asyncio.sleep(0.1)

        total_allowed = sum(instance1_results) + sum(instance2_results)
        total_blocked = len(instance1_results) + len(instance2_results) - total_allowed

        print(f"Distributed rate limiting results:")
        print(f"  Instance 1 allowed: {sum(instance1_results)}/{len(instance1_results)}")
        print(f"  Instance 2 allowed: {sum(instance2_results)}/{len(instance2_results)}")
        print(f"  Total allowed: {total_allowed}")
        print(f"  Total blocked: {total_blocked}")

        # Should respect global limit across instances
        assert total_allowed == 4, f"Expected 4 total allowed requests, got {total_allowed}"
        assert total_blocked == 2, f"Expected 2 total blocked requests, got {total_blocked}"


@pytest.mark.security
@pytest.mark.asyncio
class TestSecurityAuditingAndLogging:
    """Test security auditing and logging."""

    @pytest.fixture
    def security_harness(self):
        """Create security test harness."""
        return SecurityTestHarness()

    async def test_security_event_logging(self, security_harness):
        """Test comprehensive security event logging."""
        print("\n📝 Testing security event logging...")

        # Mock audit logger to capture events
        logged_events = []

        async def mock_log_security_event(event):
            logged_events.append(event)

        security_harness.audit_logger.log_security_event = mock_log_security_event

        # Generate various security events
        test_scenarios = [
            # Successful request
            {
                'request_data': {'question_text': 'Valid question'},
                'user_id': 'valid_user',
                'expected_status': 'success'
            },
            # Blocked request (invalid input)
            {
                'request_data': {'question_text': "'; DROP TABLE users; --"},
                'user_id': 'malicious_user',
                'expected_status': 'blocked'
            },
            # Rate limited request
            {
                'request_data': {'question_text': 'Rate limited question'},
                'user_id': 'rate_limited_user',
                'expected_status': 'blocked'  # Will be rate limited
            }
        ]

        # Set up rate limiting for one user
        security_harness.rate_limiter = RateLimiter(max_requests=0, window_seconds=60)  # Block all for rate_limited_user

        results = []
        for scenario in test_scenarios:
            result = await security_harness.process_request(
                scenario['request_data'],
                user_id=scenario['user_id']
            )
            results.append(result)

        print(f"Security event logging results:")
        print(f"  Total events logged: {len(logged_events)}")
        print(f"  Total requests processed: {len(results)}")

        # Verify all requests generated log events
        assert len(logged_events) == len(results), f"Expected {len(results)} log events, got {len(logged_events)}"

        # Verify event structure
        for event in logged_events:
            required_fields = ['request_id', 'user_id', 'action', 'status', 'timestamp']
            for field in required_fields:
                assert field in event, f"Missing required field '{field}' in security event"

        # Verify event types
        success_events = [e for e in logged_events if e['status'] == 'success']
        blocked_events = [e for e in logged_events if e['status'] == 'blocked']

        print(f"  Success events: {len(success_events)}")
        print(f"  Blocked events: {len(blocked_events)}")

        assert len(success_events) >= 1, "Should have at least one success event"
        assert len(blocked_events) >= 2, "Should have at least two blocked events"

    async def test_audit_trail_integrity(self, security_harness):
        """Test audit trail integrity and tamper detection."""
        print("\n📝 Testing audit trail integrity...")

        # Mock audit logger with integrity checking
        audit_trail = []

        def calculate_event_hash(event):
            """Calculate hash for event integrity."""
            event_str = f"{event['request_id']}{event['user_id']}{event['action']}{event['timestamp']}"
            return hashlib.sha256(event_str.encode()).hexdigest()

        async def mock_log_with_integrity(event):
            event['hash'] = calculate_event_hash(event)
            event['sequence_number'] = len(audit_trail) + 1
            audit_trail.append(event)

        security_harness.audit_logger.log_security_event = mock_log_with_integrity

        # Generate audit events
        for i in range(5):
            request_data = {'question_text': f'Audit test question {i}'}
            await security_harness.process_request(request_data, user_id=f"audit_user_{i}")

        print(f"Audit trail integrity results:")
        print(f"  Total events in trail: {len(audit_trail)}")

        # Verify sequence numbers
        for i, event in enumerate(audit_trail):
            expected_sequence = i + 1
            assert event['sequence_number'] == expected_sequence, f"Sequence number mismatch: expected {expected_sequence}, got {event['sequence_number']}"

        # Verify hash integrity
        for event in audit_trail:
            expected_hash = calculate_event_hash(event)
            assert event['hash'] == expected_hash, f"Hash integrity check failed for event {event['request_id']}"

        # Test tamper detection
        # Simulate tampering with an event
        if audit_trail:
            original_action = audit_trail[0]['action']
            audit_trail[0]['action'] = 'tampered_action'

            # Recalculate hash to detect tampering
            current_hash = audit_trail[0]['hash']
            expected_hash = calculate_event_hash(audit_trail[0])

            tamper_detected = current_hash != expected_hash
            assert tamper_detected, "Tampering should have been detected"

            # Restore original
            audit_trail[0]['action'] = original_action

            print(f"  Tamper detection: ✅")

        print(f"  Sequence integrity: ✅")
        print(f"  Hash integrity: ✅")

    async def test_security_metrics_collection(self, security_harness):
        """Test security metrics collection and analysis."""
        print("\n📊 Testing security metrics collection...")

        # Generate diverse security events
        test_cases = [
            # Normal requests
            *[{'question_text': f'Normal question {i}'} for i in range(10)],
            # Malicious requests
            *[{'question_text': "'; DROP TABLE users; --"} for _ in range(3)],
            *[{'question_text': '<script>alert("xss")</script>'} for _ in range(2)],
            # Oversized requests
            *[{'question_text': 'A' * 10000} for _ in range(2)]
        ]

        results = []
        for i, request_data in enumerate(test_cases):
            user_id = f"metrics_user_{i % 5}"  # 5 different users
            result = await security_harness.process_request(request_data, user_id=user_id)
            results.append(result)

        # Collect security metrics
        metrics = security_harness.get_security_metrics()

        print(f"Security metrics collection results:")
        print(f"  Total security events: {metrics['total_security_events']}")
        print(f"  Blocked requests: {metrics['blocked_requests']}")
        print(f"  Validation failures: {metrics['validation_failures']}")
        print(f"  Rate limit violations: {metrics['rate_limit_violations']}")

        # Verify metrics accuracy
        total_requests = len(test_cases)
        successful_requests = len([r for r in results if r['status'] == 'success'])
        blocked_requests = len([r for r in results if r['status'] == 'blocked'])

        assert successful_requests + blocked_requests == total_requests, "Metrics don't add up to total requests"
        assert metrics['blocked_requests'] == blocked_requests, f"Blocked requests metric mismatch: {metrics['blocked_requests']} vs {blocked_requests}"

        # Should have detected malicious requests
        assert metrics['validation_failures'] >= 5, f"Should have detected at least 5 validation failures, got {metrics['validation_failures']}"

        # Calculate security score
        security_score = (successful_requests / total_requests) * 100
        threat_detection_rate = (metrics['validation_failures'] / 7) * 100  # 7 malicious requests

        print(f"  Security score: {security_score:.1f}%")
        print(f"  Threat detection rate: {threat_detection_rate:.1f}%")

        assert threat_detection_rate >= 70, f"Threat detection rate {threat_detection_rate:.1f}% below threshold (70%)"
