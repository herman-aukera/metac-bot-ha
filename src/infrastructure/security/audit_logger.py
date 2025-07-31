"""Audit logging for all security events and credential access."""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import os

from ..logging.structured_logger import StructuredLogger
from ..logging.correlation_context import CorrelationContext


class AuditEventType(Enum):
    """Types of audit events."""
    CREDENTIAL_ACCESS = "credential_access"
    CREDENTIAL_STORE = "credential_store"
    CREDENTIAL_ROTATE = "credential_rotate"
    CREDENTIAL_REVOKE = "credential_revoke"
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_SUCCESS = "authorization_success"
    AUTHORIZATION_FAILURE = "authorization_failure"
    SECURITY_CHECK_PASSED = "security_check_passed"
    SECURITY_CHECK_FAILED = "security_check_failed"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INPUT_VALIDATION_FAILED = "input_validation_failed"
    XSS_ATTEMPT_DETECTED = "xss_attempt_detected"
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    CSRF_TOKEN_VALIDATION = "csrf_token_validation"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    IP_BLOCKED = "ip_blocked"
    IP_UNBLOCKED = "ip_unblocked"
    USER_AGENT_BLOCKED = "user_agent_blocked"
    SECURITY_CONFIGURATION_CHANGED = "security_configuration_changed"
    REQUEST_START = "request_start"
    REQUEST_END = "request_end"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    severity: AuditSeverity
    user_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AuditLogger:
    """Audit logging for all security events and credential access."""

    def __init__(self,
                 structured_logger: Optional[StructuredLogger] = None,
                 correlation_context: Optional[CorrelationContext] = None,
                 audit_file_path: Optional[str] = None,
                 enable_console_output: bool = True,
                 enable_file_output: bool = True,
                 enable_remote_logging: bool = False,
                 remote_endpoint: Optional[str] = None):

        self.structured_logger = structured_logger or StructuredLogger()
        self.correlation_context = correlation_context or CorrelationContext()
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.audit_file_path = audit_file_path or "logs/audit.log"
        self.enable_console_output = enable_console_output
        self.enable_file_output = enable_file_output
        self.enable_remote_logging = enable_remote_logging
        self.remote_endpoint = remote_endpoint

        # Ensure audit log directory exists
        if self.enable_file_output:
            os.makedirs(os.path.dirname(self.audit_file_path), exist_ok=True)

        # Event buffer for batch processing
        self.event_buffer: List[AuditEvent] = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds

        # Start background flush task
        self._flush_task = asyncio.create_task(self._periodic_flush())

        # Sensitive fields to mask in logs
        self.sensitive_fields = {
            'password', 'token', 'api_key', 'secret', 'credential',
            'auth_header', 'authorization', 'x-api-key', 'bearer'
        }

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return str(uuid.uuid4())

    def _mask_sensitive_data(self, data: Any) -> Any:
        """Mask sensitive data in audit logs."""
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                key_lower = key.lower()
                if any(sensitive in key_lower for sensitive in self.sensitive_fields):
                    if isinstance(value, str) and len(value) > 8:
                        masked[key] = f"{value[:4]}***{value[-4:]}"
                    else:
                        masked[key] = "***MASKED***"
                elif isinstance(value, (dict, list)):
                    masked[key] = self._mask_sensitive_data(value)
                else:
                    masked[key] = value
            return masked
        elif isinstance(data, list):
            return [self._mask_sensitive_data(item) for item in data]
        else:
            return data

    async def log_audit_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        try:
            # Add correlation ID if available
            if not event.correlation_id:
                event.correlation_id = self.correlation_context.get_correlation_id()

            # Mask sensitive data
            if event.metadata:
                event.metadata = self._mask_sensitive_data(event.metadata)

            # Add to buffer
            self.event_buffer.append(event)

            # Flush if buffer is full or event is critical
            if len(self.event_buffer) >= self.buffer_size or event.severity == AuditSeverity.CRITICAL:
                await self._flush_events()

        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")

    async def _flush_events(self) -> None:
        """Flush buffered events to all configured outputs."""
        if not self.event_buffer:
            return

        events_to_flush = self.event_buffer.copy()
        self.event_buffer.clear()

        try:
            # Console output
            if self.enable_console_output:
                await self._log_to_console(events_to_flush)

            # File output
            if self.enable_file_output:
                await self._log_to_file(events_to_flush)

            # Remote logging
            if self.enable_remote_logging and self.remote_endpoint:
                await self._log_to_remote(events_to_flush)

        except Exception as e:
            self.logger.error(f"Failed to flush audit events: {e}")
            # Re-add events to buffer for retry
            self.event_buffer.extend(events_to_flush)

    async def _log_to_console(self, events: List[AuditEvent]) -> None:
        """Log events to console."""
        for event in events:
            log_data = event.to_dict()

            if event.severity == AuditSeverity.CRITICAL:
                self.logger.critical(f"AUDIT: {json.dumps(log_data)}")
            elif event.severity == AuditSeverity.ERROR:
                self.logger.error(f"AUDIT: {json.dumps(log_data)}")
            elif event.severity == AuditSeverity.WARNING:
                self.logger.warning(f"AUDIT: {json.dumps(log_data)}")
            else:
                self.logger.info(f"AUDIT: {json.dumps(log_data)}")

    async def _log_to_file(self, events: List[AuditEvent]) -> None:
        """Log events to file."""
        try:
            with open(self.audit_file_path, 'a', encoding='utf-8') as f:
                for event in events:
                    log_line = json.dumps(event.to_dict()) + '\n'
                    f.write(log_line)
                f.flush()
        except Exception as e:
            self.logger.error(f"Failed to write audit log to file: {e}")

    async def _log_to_remote(self, events: List[AuditEvent]) -> None:
        """Log events to remote endpoint."""
        try:
            import aiohttp

            payload = {
                'events': [event.to_dict() for event in events],
                'source': 'tournament-optimization-system',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.remote_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"Remote audit logging failed: {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to send audit logs to remote endpoint: {e}")

    async def _periodic_flush(self) -> None:
        """Periodically flush buffered events."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_events()
            except asyncio.CancelledError:
                # Final flush before shutdown
                await self._flush_events()
                break
            except Exception as e:
                self.logger.error(f"Error in periodic flush: {e}")

    # Convenience methods for common audit events

    async def log_credential_access(self,
                                  service: str,
                                  credential_type: str,
                                  action: str,
                                  success: bool,
                                  user_id: Optional[str] = None,
                                  source: Optional[str] = None,
                                  error: Optional[str] = None) -> None:
        """Log credential access event."""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.CREDENTIAL_ACCESS,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            user_id=user_id,
            resource=f"{service}:{credential_type}",
            action=action,
            success=success,
            error_message=error,
            metadata={
                'service': service,
                'credential_type': credential_type,
                'source': source
            }
        )
        await self.log_audit_event(event)

    async def log_authentication_event(self,
                                     user_id: Optional[str],
                                     success: bool,
                                     client_ip: Optional[str] = None,
                                     user_agent: Optional[str] = None,
                                     endpoint: Optional[str] = None,
                                     error: Optional[str] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log authentication event."""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.AUTHENTICATION_SUCCESS if success else AuditEventType.AUTHENTICATION_FAILURE,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            user_id=user_id,
            client_ip=client_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            action="AUTHENTICATE",
            success=success,
            error_message=error,
            metadata=metadata
        )
        await self.log_audit_event(event)

    async def log_authorization_event(self,
                                    user_id: str,
                                    resource: str,
                                    action: str,
                                    success: bool,
                                    client_ip: Optional[str] = None,
                                    endpoint: Optional[str] = None,
                                    error: Optional[str] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log authorization event."""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.AUTHORIZATION_SUCCESS if success else AuditEventType.AUTHORIZATION_FAILURE,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.INFO if success else AuditSeverity.ERROR,
            user_id=user_id,
            client_ip=client_ip,
            endpoint=endpoint,
            resource=resource,
            action=action,
            success=success,
            error_message=error,
            metadata=metadata
        )
        await self.log_audit_event(event)

    async def log_security_event(self,
                               event_type: str,
                               endpoint: Optional[str] = None,
                               client_ip: Optional[str] = None,
                               user_agent: Optional[str] = None,
                               user: Optional[str] = None,
                               security_level: Optional[str] = None,
                               request_id: Optional[str] = None,
                               processing_time: Optional[float] = None,
                               error: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log general security event."""
        # Map event type string to enum
        try:
            audit_event_type = AuditEventType(event_type.lower())
        except ValueError:
            audit_event_type = AuditEventType.SECURITY_CHECK_PASSED

        # Determine severity
        severity = AuditSeverity.INFO
        if error:
            severity = AuditSeverity.ERROR
        elif "failed" in event_type.lower() or "exceeded" in event_type.lower():
            severity = AuditSeverity.WARNING
        elif "critical" in security_level.lower() if security_level else False:
            severity = AuditSeverity.CRITICAL

        event_metadata = metadata or {}
        if processing_time:
            event_metadata['processing_time_ms'] = processing_time * 1000
        if security_level:
            event_metadata['security_level'] = security_level

        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=audit_event_type,
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            user_id=user,
            client_ip=client_ip,
            user_agent=user_agent,
            endpoint=endpoint,
            request_id=request_id,
            success=error is None,
            error_message=error,
            metadata=event_metadata
        )
        await self.log_audit_event(event)

    async def log_rate_limit_event(self,
                                 rule_name: str,
                                 identifier: str,
                                 client_ip: Optional[str] = None,
                                 endpoint: Optional[str] = None,
                                 retry_after: Optional[int] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log rate limit exceeded event."""
        event_metadata = metadata or {}
        event_metadata.update({
            'rule_name': rule_name,
            'identifier': identifier,
            'retry_after': retry_after
        })

        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.WARNING,
            client_ip=client_ip,
            endpoint=endpoint,
            action="RATE_LIMIT_CHECK",
            success=False,
            error_message=f"Rate limit exceeded for rule: {rule_name}",
            metadata=event_metadata
        )
        await self.log_audit_event(event)

    async def log_input_validation_event(self,
                                       field_name: str,
                                       validation_errors: List[str],
                                       client_ip: Optional[str] = None,
                                       endpoint: Optional[str] = None,
                                       user_id: Optional[str] = None,
                                       attack_type: Optional[str] = None) -> None:
        """Log input validation failure event."""
        severity = AuditSeverity.ERROR if attack_type else AuditSeverity.WARNING

        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.INPUT_VALIDATION_FAILED,
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            user_id=user_id,
            client_ip=client_ip,
            endpoint=endpoint,
            action="INPUT_VALIDATION",
            success=False,
            error_message=f"Input validation failed for field: {field_name}",
            metadata={
                'field_name': field_name,
                'validation_errors': validation_errors,
                'attack_type': attack_type
            }
        )
        await self.log_audit_event(event)

    async def log_configuration_change(self,
                                     component: str,
                                     change_type: str,
                                     old_value: Any,
                                     new_value: Any,
                                     user_id: Optional[str] = None,
                                     client_ip: Optional[str] = None) -> None:
        """Log security configuration change."""
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.SECURITY_CONFIGURATION_CHANGED,
            timestamp=datetime.now(timezone.utc),
            severity=AuditSeverity.INFO,
            user_id=user_id,
            client_ip=client_ip,
            resource=component,
            action=change_type,
            success=True,
            metadata={
                'component': component,
                'change_type': change_type,
                'old_value': self._mask_sensitive_data(old_value),
                'new_value': self._mask_sensitive_data(new_value)
            }
        )
        await self.log_audit_event(event)

    async def search_audit_logs(self,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              event_types: Optional[List[AuditEventType]] = None,
                              user_id: Optional[str] = None,
                              client_ip: Optional[str] = None,
                              severity: Optional[AuditSeverity] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Search audit logs with filters."""
        # This is a simplified implementation
        # In production, you would use a proper log aggregation system

        results = []

        try:
            if not os.path.exists(self.audit_file_path):
                return results

            with open(self.audit_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # Apply filters
                        if start_time:
                            entry_time = datetime.fromisoformat(log_entry['timestamp'])
                            if entry_time < start_time:
                                continue

                        if end_time:
                            entry_time = datetime.fromisoformat(log_entry['timestamp'])
                            if entry_time > end_time:
                                continue

                        if event_types:
                            if log_entry['event_type'] not in [et.value for et in event_types]:
                                continue

                        if user_id and log_entry.get('user_id') != user_id:
                            continue

                        if client_ip and log_entry.get('client_ip') != client_ip:
                            continue

                        if severity and log_entry.get('severity') != severity.value:
                            continue

                        results.append(log_entry)

                        if len(results) >= limit:
                            break

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            self.logger.error(f"Failed to search audit logs: {e}")

        return results

    async def get_audit_statistics(self,
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get audit log statistics."""
        stats = {
            'total_events': 0,
            'events_by_type': {},
            'events_by_severity': {},
            'unique_users': set(),
            'unique_ips': set(),
            'success_rate': 0.0,
            'top_endpoints': {},
            'top_errors': {}
        }

        logs = await self.search_audit_logs(start_time=start_time, end_time=end_time, limit=10000)

        successful_events = 0

        for log_entry in logs:
            stats['total_events'] += 1

            # Count by type
            event_type = log_entry.get('event_type', 'unknown')
            stats['events_by_type'][event_type] = stats['events_by_type'].get(event_type, 0) + 1

            # Count by severity
            severity = log_entry.get('severity', 'unknown')
            stats['events_by_severity'][severity] = stats['events_by_severity'].get(severity, 0) + 1

            # Track unique users and IPs
            if log_entry.get('user_id'):
                stats['unique_users'].add(log_entry['user_id'])
            if log_entry.get('client_ip'):
                stats['unique_ips'].add(log_entry['client_ip'])

            # Track success rate
            if log_entry.get('success'):
                successful_events += 1

            # Track endpoints
            if log_entry.get('endpoint'):
                endpoint = log_entry['endpoint']
                stats['top_endpoints'][endpoint] = stats['top_endpoints'].get(endpoint, 0) + 1

            # Track errors
            if log_entry.get('error_message'):
                error = log_entry['error_message']
                stats['top_errors'][error] = stats['top_errors'].get(error, 0) + 1

        # Calculate success rate
        if stats['total_events'] > 0:
            stats['success_rate'] = successful_events / stats['total_events']

        # Convert sets to counts
        stats['unique_users'] = len(stats['unique_users'])
        stats['unique_ips'] = len(stats['unique_ips'])

        return stats

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, '_flush_task'):
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self._flush_events()
