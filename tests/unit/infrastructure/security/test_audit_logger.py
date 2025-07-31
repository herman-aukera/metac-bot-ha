"""Unit tests for AuditLogger."""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from src.infrastructure.security.audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSeverity
)
from src.infrastructure.logging.structured_logger import StructuredLogger
from src.infrastructure.logging.correlation_context import CorrelationContext


class TestAuditEvent:
    """Test AuditEvent data structure."""

    def test_audit_event_creation(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_id="test-event-123",
            event_type=AuditEventType.CREDENTIAL_ACCESS,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.INFO,
            user_id="test_user",
            client_ip="192.168.1.1",
            action="ACCESS",
            success=True
        )

        assert event.event_id == "test-event-123"
        assert event.event_type == AuditEventType.CREDENTIAL_ACCESS
        assert event.severity == AuditSeverity.INFO
        assert event.user_id == "test_user"
        assert event.success is True

    def test_audit_event_to_dict(self):
        """Test converting audit event to dictionary."""
        timestamp = datetime.now(timezone.utc)
        event = AuditEvent(
            event_id="test-event-123",
            event_type=AuditEventType.AUTHENTICATION_SUCCESS,
            timestamp=timestamp,
            severity=AuditSeverity.INFO,
            metadata={"test": "data"}
        )

        event_dict = event.to_dict()

        assert event_dict["event_id"] == "test-event-123"
        assert event_dict["event_type"] == "authentication_success"
        assert event_dict["severity"] == "info"
        assert event_dict["timestamp"] == timestamp.isoformat()
        assert event_dict["metadata"] == {"test": "data"}


class TestAuditLogger:
    """Test AuditLogger functionality."""

    @pytest.fixture
    def mock_structured_logger(self):
        return Mock(spec=StructuredLogger)

    @pytest.fixture
    def mock_correlation_context(self):
        context = Mock(spec=CorrelationContext)
        context.get_correlation_id.return_value = "test-correlation-123"
        return context

    @pytest.fixture
    def temp_audit_file(self):
        """Create temporary audit log file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def audit_logger(self, mock_structured_logger, mock_correlation_context, temp_audit_file):
        return AuditLogger(
            structured_logger=mock_structured_logger,
            correlation_context=mock_correlation_context,
            audit_file_path=temp_audit_file,
            enable_console_output=False,  # Disable for testing
            enable_remote_logging=False
        )

    def test_mask_sensitive_data(self, audit_logger):
        """Test masking of sensitive data."""
        data = {
            "username": "testuser",
            "password": "secret123",
            "api_key": "sk-1234567890abcdef",
            "token": "bearer_token_xyz",
            "public_info": "visible",
            "nested": {
                "secret": "hidden_value",
                "safe_data": "also_visible"
            },
            "list_data": [
                {"password": "list_secret", "name": "item1"},
                "plain_string"
            ]
        }

        masked = audit_logger._mask_sensitive_data(data)

        assert masked["username"] == "testuser"
        assert masked["password"] == "***MASKED***"
        assert masked["api_key"] == "sk-1***cdef"
        assert masked["token"] == "bear***_xyz"
        assert masked["public_info"] == "visible"
        assert masked["nested"]["secret"] == "***MASKED***"
        assert masked["nested"]["safe_data"] == "also_visible"
        assert masked["list_data"][0]["password"] == "***MASKED***"
        assert masked["list_data"][0]["name"] == "item1"
        assert masked["list_data"][1] == "plain_string"

    @pytest.mark.asyncio
    async def test_log_audit_event(self, audit_logger, mock_correlation_context):
        """Test logging an audit event."""
        event = AuditEvent(
            event_id="test-event-123",
            event_type=AuditEventType.CREDENTIAL_ACCESS,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.INFO,
            user_id="test_user",
            action="ACCESS",
            success=True
        )

        await audit_logger.log_audit_event(event)

        # Event should be added to buffer
        assert len(audit_logger.event_buffer) == 1
        assert audit_logger.event_buffer[0].correlation_id == "test-correlation-123"

    @pytest.mark.asyncio
    async def test_log_audit_event_critical_immediate_flush(self, audit_logger):
        """Test that critical events trigger immediate flush."""
        event = AuditEvent(
            event_id="critical-event-123",
            event_type=AuditEventType.SECURITY_CHECK_FAILED,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.CRITICAL,
            error_message="Critical security failure"
        )

        with patch.object(audit_logger, '_flush_events', new_callable=AsyncMock) as mock_flush:
            await audit_logger.log_audit_event(event)
            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_events_to_file(self, audit_logger, temp_audit_file):
        """Test flushing events to file."""
        # Add events to buffer
        event1 = AuditEvent(
            event_id="event-1",
            event_type=AuditEventType.CREDENTIAL_ACCESS,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.INFO
        )
        event2 = AuditEvent(
            event_id="event-2",
            event_type=AuditEventType.AUTHENTICATION_SUCCESS,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.INFO
        )

        audit_logger.event_buffer = [event1, event2]

        # Flush events
        await audit_logger._flush_events()

        # Verify file contents
        with open(temp_audit_file, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Parse and verify first event
        event1_data = json.loads(lines[0].strip())
        assert event1_data["event_id"] == "event-1"
        assert event1_data["event_type"] == "credential_access"

        # Parse and verify second event
        event2_data = json.loads(lines[1].strip())
        assert event2_data["event_id"] == "event-2"
        assert event2_data["event_type"] == "authentication_success"

        # Buffer should be cleared
        assert len(audit_logger.event_buffer) == 0

    @pytest.mark.asyncio
    async def test_log_credential_access(self, audit_logger):
        """Test logging credential access event."""
        await audit_logger.log_credential_access(
            service="openai",
            credential_type="api_key",
            action="ACCESS",
            success=True,
            user_id="test_user",
            source="vault"
        )

        assert len(audit_logger.event_buffer) == 1
        event = audit_logger.event_buffer[0]
        assert event.event_type == AuditEventType.CREDENTIAL_ACCESS
        assert event.user_id == "test_user"
        assert event.resource == "openai:api_key"
        assert event.action == "ACCESS"
        assert event.success is True
        assert event.metadata["service"] == "openai"
        assert event.metadata["source"] == "vault"

    @pytest.mark.asyncio
    async def test_log_authentication_event_success(self, audit_logger):
        """Test logging successful authentication event."""
        await audit_logger.log_authentication_event(
            user_id="test_user",
            success=True,
            client_ip="192.168.1.1",
            user_agent="test-agent",
            endpoint="/auth/login"
        )

        assert len(audit_logger.event_buffer) == 1
        event = audit_logger.event_buffer[0]
        assert event.event_type == AuditEventType.AUTHENTICATION_SUCCESS
        assert event.user_id == "test_user"
        assert event.client_ip == "192.168.1.1"
        assert event.user_agent == "test-agent"
        assert event.endpoint == "/auth/login"
        assert event.success is True
        assert event.severity == AuditSeverity.INFO

    @pytest.mark.asyncio
    async def test_log_authentication_event_failure(self, audit_logger):
        """Test logging failed authentication event."""
        await audit_logger.log_authentication_event(
            user_id="test_user",
            success=False,
            client_ip="192.168.1.1",
            error="Invalid credentials"
        )

        assert len(audit_logger.event_buffer) == 1
        event = audit_logger.event_buffer[0]
        assert event.event_type == AuditEventType.AUTHENTICATION_FAILURE
        assert event.success is False
        assert event.error_message == "Invalid credentials"
        assert event.severity == AuditSeverity.WARNING

    @pytest.mark.asyncio
    async def test_log_authorization_event(self, audit_logger):
        """Test logging authorization event."""
        await audit_logger.log_authorization_event(
            user_id="test_user",
            resource="/admin/users",
            action="READ",
            success=True,
            client_ip="192.168.1.1",
            endpoint="/admin/users"
        )

        assert len(audit_logger.event_buffer) == 1
        event = audit_logger.event_buffer[0]
        assert event.event_type == AuditEventType.AUTHORIZATION_SUCCESS
        assert event.user_id == "test_user"
        assert event.resource == "/admin/users"
        assert event.action == "READ"
        assert event.success is True

    @pytest.mark.asyncio
    async def test_log_security_event(self, audit_logger):
        """Test logging general security event."""
        await audit_logger.log_security_event(
            event_type="security_check_passed",
            endpoint="/api/test",
            client_ip="192.168.1.1",
            user="test_user",
            security_level="high",
            processing_time=0.123
        )

        assert len(audit_logger.event_buffer) == 1
        event = audit_logger.event_buffer[0]
        assert event.event_type == AuditEventType.SECURITY_CHECK_PASSED
        assert event.endpoint == "/api/test"
        assert event.client_ip == "192.168.1.1"
        assert event.user_id == "test_user"
        assert event.metadata["security_level"] == "high"
        assert event.metadata["processing_time_ms"] == 123.0

    @pytest.mark.asyncio
    async def test_log_rate_limit_event(self, audit_logger):
        """Test logging rate limit event."""
        await audit_logger.log_rate_limit_event(
            rule_name="api_general",
            identifier="user123",
            client_ip="192.168.1.1",
            endpoint="/api/test",
            retry_after=60
        )

        assert len(audit_logger.event_buffer) == 1
        event = audit_logger.event_buffer[0]
        assert event.event_type == AuditEventType.RATE_LIMIT_EXCEEDED
        assert event.client_ip == "192.168.1.1"
        assert event.endpoint == "/api/test"
        assert event.severity == AuditSeverity.WARNING
        assert event.metadata["rule_name"] == "api_general"
        assert event.metadata["retry_after"] == 60

    @pytest.mark.asyncio
    async def test_log_input_validation_event(self, audit_logger):
        """Test logging input validation event."""
        await audit_logger.log_input_validation_event(
            field_name="user_input",
            validation_errors=["Field too long", "Invalid characters"],
            client_ip="192.168.1.1",
            endpoint="/api/submit",
            attack_type="xss"
        )

        assert len(audit_logger.event_buffer) == 1
        event = audit_logger.event_buffer[0]
        assert event.event_type == AuditEventType.INPUT_VALIDATION_FAILED
        assert event.client_ip == "192.168.1.1"
        assert event.endpoint == "/api/submit"
        assert event.severity == AuditSeverity.ERROR  # Because attack_type is specified
        assert event.metadata["field_name"] == "user_input"
        assert event.metadata["validation_errors"] == ["Field too long", "Invalid characters"]
        assert event.metadata["attack_type"] == "xss"

    @pytest.mark.asyncio
    async def test_log_configuration_change(self, audit_logger):
        """Test logging configuration change event."""
        await audit_logger.log_configuration_change(
            component="security_middleware",
            change_type="UPDATE",
            old_value={"rate_limit": 100},
            new_value={"rate_limit": 200},
            user_id="admin_user",
            client_ip="192.168.1.1"
        )

        assert len(audit_logger.event_buffer) == 1
        event = audit_logger.event_buffer[0]
        assert event.event_type == AuditEventType.SECURITY_CONFIGURATION_CHANGED
        assert event.user_id == "admin_user"
        assert event.resource == "security_middleware"
        assert event.action == "UPDATE"
        assert event.metadata["component"] == "security_middleware"
        assert event.metadata["old_value"] == {"rate_limit": 100}
        assert event.metadata["new_value"] == {"rate_limit": 200}

    @pytest.mark.asyncio
    async def test_search_audit_logs(self, audit_logger, temp_audit_file):
        """Test searching audit logs."""
        # Create test events and flush to file
        events = [
            AuditEvent(
                event_id="event-1",
                event_type=AuditEventType.CREDENTIAL_ACCESS,
                timestamp=datetime.now(timezone.utc),
                severity=AuditSeverity.INFO,
                user_id="user1"
            ),
            AuditEvent(
                event_id="event-2",
                event_type=AuditEventType.AUTHENTICATION_SUCCESS,
                timestamp=datetime.now(timezone.utc),
                severity=AuditSeverity.INFO,
                user_id="user2"
            ),
            AuditEvent(
                event_id="event-3",
                event_type=AuditEventType.CREDENTIAL_ACCESS,
                timestamp=datetime.now(timezone.utc),
                severity=AuditSeverity.WARNING,
                user_id="user1"
            )
        ]

        audit_logger.event_buffer = events
        await audit_logger._flush_events()

        # Search by event type
        results = await audit_logger.search_audit_logs(
            event_types=[AuditEventType.CREDENTIAL_ACCESS]
        )
        assert len(results) == 2
        assert all(r["event_type"] == "credential_access" for r in results)

        # Search by user
        results = await audit_logger.search_audit_logs(user_id="user1")
        assert len(results) == 2
        assert all(r["user_id"] == "user1" for r in results)

        # Search by severity
        results = await audit_logger.search_audit_logs(severity=AuditSeverity.WARNING)
        assert len(results) == 1
        assert results[0]["severity"] == "warning"

    @pytest.mark.asyncio
    async def test_get_audit_statistics(self, audit_logger, temp_audit_file):
        """Test getting audit statistics."""
        # Create test events and flush to file
        events = [
            AuditEvent(
                event_id="event-1",
                event_type=AuditEventType.CREDENTIAL_ACCESS,
                timestamp=datetime.now(timezone.utc),
                severity=AuditSeverity.INFO,
                user_id="user1",
                client_ip="192.168.1.1",
                endpoint="/api/test",
                success=True
            ),
            AuditEvent(
                event_id="event-2",
                event_type=AuditEventType.AUTHENTICATION_FAILURE,
                timestamp=datetime.now(timezone.utc),
                severity=AuditSeverity.WARNING,
                user_id="user2",
                client_ip="192.168.1.2",
                endpoint="/auth/login",
                success=False,
                error_message="Invalid password"
            ),
            AuditEvent(
                event_id="event-3",
                event_type=AuditEventType.CREDENTIAL_ACCESS,
                timestamp=datetime.now(timezone.utc),
                severity=AuditSeverity.INFO,
                user_id="user1",
                success=True
            )
        ]

        audit_logger.event_buffer = events
        await audit_logger._flush_events()

        # Get statistics
        stats = await audit_logger.get_audit_statistics()

        assert stats["total_events"] == 3
        assert stats["events_by_type"]["credential_access"] == 2
        assert stats["events_by_type"]["authentication_failure"] == 1
        assert stats["events_by_severity"]["info"] == 2
        assert stats["events_by_severity"]["warning"] == 1
        assert stats["unique_users"] == 2
        assert stats["unique_ips"] == 2
        assert stats["success_rate"] == 2/3  # 2 successful out of 3 total
        assert stats["top_endpoints"]["/api/test"] == 1
        assert stats["top_endpoints"]["/auth/login"] == 1
        assert stats["top_errors"]["Invalid password"] == 1

    @pytest.mark.asyncio
    async def test_buffer_size_flush(self, audit_logger):
        """Test that buffer flushes when it reaches maximum size."""
        # Set small buffer size for testing
        audit_logger.buffer_size = 2

        with patch.object(audit_logger, '_flush_events', new_callable=AsyncMock) as mock_flush:
            # Add events to reach buffer size
            event1 = AuditEvent(
                event_id="event-1",
                event_type=AuditEventType.CREDENTIAL_ACCESS,
                timestamp=datetime.now(timezone.utc),
                severity=AuditSeverity.INFO
            )
            event2 = AuditEvent(
                event_id="event-2",
                event_type=AuditEventType.CREDENTIAL_ACCESS,
                timestamp=datetime.now(timezone.utc),
                severity=AuditSeverity.INFO
            )

            await audit_logger.log_audit_event(event1)
            mock_flush.assert_not_called()

            await audit_logger.log_audit_event(event2)
            mock_flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_periodic_flush_task(self, audit_logger):
        """Test periodic flush task."""
        # Set short flush interval for testing
        audit_logger.flush_interval = 0.1

        # Add event to buffer
        event = AuditEvent(
            event_id="periodic-test",
            event_type=AuditEventType.CREDENTIAL_ACCESS,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.INFO
        )
        audit_logger.event_buffer.append(event)

        # Wait for periodic flush
        await asyncio.sleep(0.2)

        # Buffer should be empty after flush
        assert len(audit_logger.event_buffer) == 0

    @pytest.mark.asyncio
    async def test_cleanup(self, audit_logger):
        """Test cleanup of audit logger resources."""
        # Add event to buffer
        event = AuditEvent(
            event_id="cleanup-test",
            event_type=AuditEventType.CREDENTIAL_ACCESS,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.INFO
        )
        audit_logger.event_buffer.append(event)

        # Cleanup should flush remaining events
        await audit_logger.cleanup()

        # Buffer should be empty
        assert len(audit_logger.event_buffer) == 0

    @pytest.mark.asyncio
    async def test_remote_logging(self, mock_structured_logger, mock_correlation_context):
        """Test remote logging functionality."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            audit_logger = AuditLogger(
                structured_logger=mock_structured_logger,
                correlation_context=mock_correlation_context,
                enable_console_output=False,
                enable_file_output=False,
                enable_remote_logging=True,
                remote_endpoint="https://logs.example.com/audit"
            )

            # Add event and flush
            event = AuditEvent(
                event_id="remote-test",
                event_type=AuditEventType.CREDENTIAL_ACCESS,
                timestamp=datetime.now(timezone.utc),
                severity=AuditSeverity.INFO
            )
            audit_logger.event_buffer.append(event)

            await audit_logger._flush_events()

            # Verify remote call was made
            mock_session.return_value.__aenter__.return_value.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_in_flush(self, audit_logger, temp_audit_file):
        """Test error handling during flush operations."""
        # Add event to buffer
        event = AuditEvent(
            event_id="error-test",
            event_type=AuditEventType.CREDENTIAL_ACCESS,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.INFO
        )
        audit_logger.event_buffer.append(event)

        # Mock file write error
        with patch('builtins.open', side_effect=IOError("File write error")):
            # Should not raise exception, but should re-add events to buffer
            await audit_logger._flush_events()

            # Events should be back in buffer for retry
            assert len(audit_logger.event_buffer) == 1

    def test_generate_event_id(self, audit_logger):
        """Test event ID generation."""
        event_id = audit_logger._generate_event_id()

        assert isinstance(event_id, str)
        assert len(event_id) > 0

        # Should generate unique IDs
        event_id2 = audit_logger._generate_event_id()
        assert event_id != event_id2


@pytest.mark.asyncio
async def test_audit_logger_integration():
    """Integration test for audit logger with file output."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        temp_path = f.name

    try:
        # Create audit logger with file output
        audit_logger = AuditLogger(
            audit_file_path=temp_path,
            enable_console_output=False,
            enable_remote_logging=False
        )

        # Log various types of events
        await audit_logger.log_credential_access(
            service="test_service",
            credential_type="api_key",
            action="ACCESS",
            success=True,
            user_id="test_user"
        )

        await audit_logger.log_authentication_event(
            user_id="test_user",
            success=True,
            client_ip="192.168.1.1"
        )

        await audit_logger.log_security_event(
            event_type="security_check_passed",
            endpoint="/api/test",
            user="test_user"
        )

        # Force flush
        await audit_logger._flush_events()

        # Verify file contents
        with open(temp_path, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 3

        # Parse and verify events
        events = [json.loads(line.strip()) for line in lines]

        assert events[0]["event_type"] == "credential_access"
        assert events[1]["event_type"] == "authentication_success"
        assert events[2]["event_type"] == "security_check_passed"

        # Cleanup
        await audit_logger.cleanup()

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
