"""Secure credential management with vault integration and rotation."""

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .audit_logger import AuditLogger
from ...domain.exceptions.infrastructure_exceptions import (
    CredentialError,
    VaultConnectionError,
    CredentialRotationError
)


@dataclass
class CredentialMetadata:
    """Metadata for credential tracking and rotation."""
    service: str
    credential_type: str
    created_at: datetime
    expires_at: Optional[datetime]
    rotation_interval: Optional[timedelta]
    last_rotated: Optional[datetime]
    rotation_count: int = 0
    is_active: bool = True


class VaultClient(ABC):
    """Abstract vault client interface."""

    @abstractmethod
    async def get_secret(self, path: str) -> Dict[str, Any]:
        """Retrieve secret from vault."""
        pass

    @abstractmethod
    async def put_secret(self, path: str, data: Dict[str, Any]) -> bool:
        """Store secret in vault."""
        pass

    @abstractmethod
    async def delete_secret(self, path: str) -> bool:
        """Delete secret from vault."""
        pass

    @abstractmethod
    async def list_secrets(self, path: str) -> List[str]:
        """List secrets at path."""
        pass


class HashiCorpVaultClient(VaultClient):
    """HashiCorp Vault client implementation."""

    def __init__(self, vault_url: str, vault_token: str):
        self.vault_url = vault_url.rstrip('/')
        self.vault_token = vault_token
        self.session = None

    async def get_secret(self, path: str) -> Dict[str, Any]:
        """Retrieve secret from HashiCorp Vault."""
        import aiohttp

        headers = {'X-Vault-Token': self.vault_token}
        url = f"{self.vault_url}/v1/secret/data/{path}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('data', {}).get('data', {})
                    elif response.status == 404:
                        return {}
                    else:
                        raise VaultConnectionError(
                            f"Vault request failed with status {response.status}",
                            error_code="VAULT_REQUEST_FAILED",
                            context={"path": path, "status": response.status}
                        )
        except Exception as e:
            if isinstance(e, VaultConnectionError):
                raise
            raise VaultConnectionError(
                f"Failed to connect to vault: {str(e)}",
                error_code="VAULT_CONNECTION_ERROR",
                context={"path": path}
            )

    async def put_secret(self, path: str, data: Dict[str, Any]) -> bool:
        """Store secret in HashiCorp Vault."""
        import aiohttp

        headers = {
            'X-Vault-Token': self.vault_token,
            'Content-Type': 'application/json'
        }
        url = f"{self.vault_url}/v1/secret/data/{path}"
        payload = {"data": data}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    return response.status in [200, 204]
        except Exception:
            return False

    async def delete_secret(self, path: str) -> bool:
        """Delete secret from HashiCorp Vault."""
        import aiohttp

        headers = {'X-Vault-Token': self.vault_token}
        url = f"{self.vault_url}/v1/secret/data/{path}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers) as response:
                    return response.status in [200, 204]
        except Exception:
            return False

    async def list_secrets(self, path: str) -> List[str]:
        """List secrets at path in HashiCorp Vault."""
        import aiohttp

        headers = {'X-Vault-Token': self.vault_token}
        url = f"{self.vault_url}/v1/secret/metadata/{path}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('data', {}).get('keys', [])
                    return []
        except Exception:
            return []


class SecureCredentialManager:
    """Secure credential management with vault integration and rotation."""

    def __init__(self,
                 vault_client: Optional[VaultClient] = None,
                 audit_logger: Optional[AuditLogger] = None,
                 encryption_key: Optional[bytes] = None):
        self.vault_client = vault_client
        self.audit_logger = audit_logger or AuditLogger()
        self.logger = logging.getLogger(__name__)

        # Initialize encryption for local storage
        if encryption_key:
            self.cipher_suite = Fernet(encryption_key)
        else:
            # Generate key from environment or create new one
            key = self._get_or_create_encryption_key()
            self.cipher_suite = Fernet(key)

        # In-memory credential cache with metadata
        self._credential_cache: Dict[str, Dict[str, Any]] = {}
        self._credential_metadata: Dict[str, CredentialMetadata] = {}

        # Rotation schedule
        self._rotation_tasks: Dict[str, asyncio.Task] = {}

        # Sensitive keys that should be masked in logs
        self.sensitive_keys = {
            'api_key', 'token', 'password', 'secret', 'key', 'credential',
            'auth', 'bearer', 'oauth', 'jwt', 'private_key', 'cert'
        }

    def _get_or_create_encryption_key(self) -> bytes:
        """Get encryption key from environment or create new one."""
        key_env = os.getenv('CREDENTIAL_ENCRYPTION_KEY')
        if key_env:
            return base64.urlsafe_b64decode(key_env.encode())

        # Generate new key from password
        password = os.getenv('CREDENTIAL_PASSWORD', 'default-password').encode()
        salt = os.getenv('CREDENTIAL_SALT', 'default-salt').encode()

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password))

    async def get_credential(self, service: str, credential_type: str) -> str:
        """Retrieve credential with automatic rotation check."""
        credential_key = f"{service}:{credential_type}"

        try:
            # Check if rotation is needed
            await self._check_rotation_needed(credential_key)

            # Try cache first
            if credential_key in self._credential_cache:
                credential = self._credential_cache[credential_key]

                # Log access (without credential value)
                await self.audit_logger.log_credential_access(
                    service=service,
                    credential_type=credential_type,
                    action="ACCESS",
                    success=True
                )

                return credential['value']

            # Try vault if available
            if self.vault_client:
                vault_data = await self.vault_client.get_secret(f"credentials/{service}")
                if credential_type in vault_data:
                    credential_value = vault_data[credential_type]

                    # Cache the credential
                    self._credential_cache[credential_key] = {
                        'value': credential_value,
                        'source': 'vault',
                        'retrieved_at': datetime.utcnow()
                    }

                    await self.audit_logger.log_credential_access(
                        service=service,
                        credential_type=credential_type,
                        action="ACCESS",
                        success=True,
                        source="vault"
                    )

                    return credential_value

            # Fallback to environment variables
            env_key = f"{service.upper()}_{credential_type.upper()}"
            credential_value = os.getenv(env_key)

            if credential_value:
                # Cache the credential
                self._credential_cache[credential_key] = {
                    'value': credential_value,
                    'source': 'environment',
                    'retrieved_at': datetime.utcnow()
                }

                await self.audit_logger.log_credential_access(
                    service=service,
                    credential_type=credential_type,
                    action="ACCESS",
                    success=True,
                    source="environment"
                )

                return credential_value

            # Credential not found
            await self.audit_logger.log_credential_access(
                service=service,
                credential_type=credential_type,
                action="ACCESS",
                success=False,
                error="CREDENTIAL_NOT_FOUND"
            )

            raise CredentialError(
                f"Credential not found: {service}:{credential_type}",
                error_code="CREDENTIAL_NOT_FOUND",
                context={"service": service, "credential_type": credential_type}
            )

        except Exception as e:
            if isinstance(e, CredentialError):
                raise

            await self.audit_logger.log_credential_access(
                service=service,
                credential_type=credential_type,
                action="ACCESS",
                success=False,
                error=str(e)
            )

            raise CredentialError(
                f"Failed to retrieve credential: {str(e)}",
                error_code="CREDENTIAL_RETRIEVAL_ERROR",
                context={"service": service, "credential_type": credential_type}
            )

    async def store_credential(self,
                             service: str,
                             credential_type: str,
                             credential_value: str,
                             rotation_interval: Optional[timedelta] = None) -> bool:
        """Store credential securely with metadata."""
        credential_key = f"{service}:{credential_type}"

        try:
            # Store in vault if available
            if self.vault_client:
                vault_data = await self.vault_client.get_secret(f"credentials/{service}")
                vault_data[credential_type] = credential_value

                success = await self.vault_client.put_secret(
                    f"credentials/{service}",
                    vault_data
                )

                if success:
                    # Update cache
                    self._credential_cache[credential_key] = {
                        'value': credential_value,
                        'source': 'vault',
                        'stored_at': datetime.utcnow()
                    }

                    # Update metadata
                    self._credential_metadata[credential_key] = CredentialMetadata(
                        service=service,
                        credential_type=credential_type,
                        created_at=datetime.utcnow(),
                        expires_at=None,
                        rotation_interval=rotation_interval,
                        last_rotated=datetime.utcnow()
                    )

                    await self.audit_logger.log_credential_access(
                        service=service,
                        credential_type=credential_type,
                        action="STORE",
                        success=True,
                        source="vault"
                    )

                    # Schedule rotation if interval provided
                    if rotation_interval:
                        await self._schedule_rotation(credential_key, rotation_interval)

                    return True

            # Store encrypted in local cache as fallback
            encrypted_value = self.cipher_suite.encrypt(credential_value.encode())
            self._credential_cache[credential_key] = {
                'value': credential_value,  # Keep unencrypted in memory for performance
                'encrypted_value': encrypted_value,
                'source': 'local',
                'stored_at': datetime.utcnow()
            }

            await self.audit_logger.log_credential_access(
                service=service,
                credential_type=credential_type,
                action="STORE",
                success=True,
                source="local"
            )

            return True

        except Exception as e:
            await self.audit_logger.log_credential_access(
                service=service,
                credential_type=credential_type,
                action="STORE",
                success=False,
                error=str(e)
            )

            raise CredentialError(
                f"Failed to store credential: {str(e)}",
                error_code="CREDENTIAL_STORAGE_ERROR",
                context={"service": service, "credential_type": credential_type}
            )

    async def rotate_credential(self, service: str, credential_type: str) -> bool:
        """Rotate credential and update all dependent services."""
        credential_key = f"{service}:{credential_type}"

        try:
            # Generate new credential (this would typically call the service's API)
            new_credential = await self._generate_new_credential(service, credential_type)

            if not new_credential:
                raise CredentialRotationError(
                    f"Failed to generate new credential for {service}:{credential_type}",
                    error_code="CREDENTIAL_GENERATION_FAILED"
                )

            # Store the new credential
            await self.store_credential(service, credential_type, new_credential)

            # Update metadata
            if credential_key in self._credential_metadata:
                metadata = self._credential_metadata[credential_key]
                metadata.last_rotated = datetime.utcnow()
                metadata.rotation_count += 1

            await self.audit_logger.log_credential_access(
                service=service,
                credential_type=credential_type,
                action="ROTATE",
                success=True
            )

            self.logger.info(f"Successfully rotated credential for {service}:{credential_type}")
            return True

        except Exception as e:
            await self.audit_logger.log_credential_access(
                service=service,
                credential_type=credential_type,
                action="ROTATE",
                success=False,
                error=str(e)
            )

            raise CredentialRotationError(
                f"Failed to rotate credential: {str(e)}",
                error_code="CREDENTIAL_ROTATION_ERROR",
                context={"service": service, "credential_type": credential_type}
            )

    async def _generate_new_credential(self, service: str, credential_type: str) -> Optional[str]:
        """Generate new credential for service (placeholder for service-specific logic)."""
        # This would typically call the service's API to generate a new API key
        # For now, return a placeholder
        import secrets
        import string

        if credential_type.lower() in ['api_key', 'token']:
            # Generate a secure random string
            alphabet = string.ascii_letters + string.digits
            return ''.join(secrets.choice(alphabet) for _ in range(32))

        return None

    async def _check_rotation_needed(self, credential_key: str) -> None:
        """Check if credential rotation is needed."""
        if credential_key not in self._credential_metadata:
            return

        metadata = self._credential_metadata[credential_key]
        if not metadata.rotation_interval:
            return

        if metadata.last_rotated:
            next_rotation = metadata.last_rotated + metadata.rotation_interval
            if datetime.utcnow() >= next_rotation:
                # Schedule immediate rotation
                service, credential_type = credential_key.split(':', 1)
                asyncio.create_task(self.rotate_credential(service, credential_type))

    async def _schedule_rotation(self, credential_key: str, interval: timedelta) -> None:
        """Schedule automatic credential rotation."""
        async def rotation_task():
            while True:
                await asyncio.sleep(interval.total_seconds())
                service, credential_type = credential_key.split(':', 1)
                try:
                    await self.rotate_credential(service, credential_type)
                except Exception as e:
                    self.logger.error(f"Scheduled rotation failed for {credential_key}: {e}")

        # Cancel existing task if any
        if credential_key in self._rotation_tasks:
            self._rotation_tasks[credential_key].cancel()

        # Start new rotation task
        self._rotation_tasks[credential_key] = asyncio.create_task(rotation_task())

    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in logs and responses."""
        if not isinstance(data, dict):
            return data

        masked_data = {}

        for key, value in data.items():
            key_lower = key.lower()

            # Check if key contains sensitive terms
            is_sensitive = any(sensitive in key_lower for sensitive in self.sensitive_keys)

            if is_sensitive:
                if isinstance(value, str) and len(value) > 8:
                    # Show first 4 and last 4 characters
                    masked_data[key] = f"{value[:4]}***{value[-4:]}"
                else:
                    masked_data[key] = "***MASKED***"
            elif isinstance(value, dict):
                # Recursively mask nested dictionaries
                masked_data[key] = self.mask_sensitive_data(value)
            elif isinstance(value, list):
                # Handle lists that might contain dictionaries
                masked_data[key] = [
                    self.mask_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                masked_data[key] = value

        return masked_data

    async def list_credentials(self, service: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all credentials with metadata (values masked)."""
        credentials = []

        for credential_key, metadata in self._credential_metadata.items():
            if service and not credential_key.startswith(f"{service}:"):
                continue

            credential_info = {
                'service': metadata.service,
                'credential_type': metadata.credential_type,
                'created_at': metadata.created_at.isoformat(),
                'last_rotated': metadata.last_rotated.isoformat() if metadata.last_rotated else None,
                'rotation_count': metadata.rotation_count,
                'is_active': metadata.is_active,
                'has_rotation_schedule': metadata.rotation_interval is not None
            }

            credentials.append(credential_info)

        return credentials

    async def revoke_credential(self, service: str, credential_type: str) -> bool:
        """Revoke and remove credential."""
        credential_key = f"{service}:{credential_type}"

        try:
            # Remove from vault if available
            if self.vault_client:
                vault_data = await self.vault_client.get_secret(f"credentials/{service}")
                if credential_type in vault_data:
                    del vault_data[credential_type]
                    await self.vault_client.put_secret(f"credentials/{service}", vault_data)

            # Remove from cache
            if credential_key in self._credential_cache:
                del self._credential_cache[credential_key]

            # Update metadata
            if credential_key in self._credential_metadata:
                self._credential_metadata[credential_key].is_active = False

            # Cancel rotation task
            if credential_key in self._rotation_tasks:
                self._rotation_tasks[credential_key].cancel()
                del self._rotation_tasks[credential_key]

            await self.audit_logger.log_credential_access(
                service=service,
                credential_type=credential_type,
                action="REVOKE",
                success=True
            )

            return True

        except Exception as e:
            await self.audit_logger.log_credential_access(
                service=service,
                credential_type=credential_type,
                action="REVOKE",
                success=False,
                error=str(e)
            )

            raise CredentialError(
                f"Failed to revoke credential: {str(e)}",
                error_code="CREDENTIAL_REVOCATION_ERROR",
                context={"service": service, "credential_type": credential_type}
            )

    async def cleanup(self) -> None:
        """Cleanup resources and cancel rotation tasks."""
        for task in self._rotation_tasks.values():
            task.cancel()

        self._rotation_tasks.clear()
        self._credential_cache.clear()
