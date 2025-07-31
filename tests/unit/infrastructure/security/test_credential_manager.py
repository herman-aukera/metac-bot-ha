"""Unit tests for SecureCredentialManager."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

from src.infrastructure.security.credential_manager import (
    SecureCredentialManager,
    CredentialMetadata,
    HashiCorpVaultClient
)
from src.infrastructure.security.rate_limiter import InMemoryRedis
from src.infrastructure.security.audit_logger import AuditLogger
from src.domain.exceptions.infrastructure_exceptions import (
    CredentialError,
    VaultConnectionError,
    CredentialRotationError
)


class TestSecureCredentialManager:
    """Test SecureCredentialManager functionality."""

    @pytest.fixture
    def mock_audit_logger(self):
        return Mock(spec=AuditLogger)

    @pytest.fixture
    def mock_vault_client(self):
        vault_client = Mock()
        vault_client.get_secret = AsyncMock(return_value={})
        vault_client.put_secret = AsyncMock(return_value=True)
        vault_client.delete_secret = AsyncMock(return_value=True)
        vault_client.list_secrets = AsyncMock(return_value=[])
        return vault_client

    @pytest.fixture
    def credential_manager(self, mock_audit_logger, mock_vault_client):
        return SecureCredentialManager(
            vault_client=mock_vault_client,
            audit_logger=mock_audit_logger
        )

    @pytest.mark.asyncio
    async def test_get_credential_from_vault(self, credential_manager, mock_vault_client, mock_audit_logger):
        """Test retrieving credential from vault."""
        # Setup
        mock_vault_client.get_secret.return_value = {"api_key": "test-key-123"}
        mock_audit_logger.log_credential_access = AsyncMock()

        # Execute
        result = await credential_manager.get_credential("openai", "api_key")

        # Verify
        assert result == "test-key-123"
        mock_vault_client.get_secret.assert_called_once_with("credentials/openai")
        mock_audit_logger.log_credential_access.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_credential_from_environment(self, credential_manager, mock_vault_client, mock_audit_logger):
        """Test retrieving credential from environment variables."""
        # Setup
        mock_vault_client.get_secret.return_value = {}
        mock_audit_logger.log_credential_access = AsyncMock()

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key-456'}):
            # Execute
            result = await credential_manager.get_credential("openai", "api_key")

            # Verify
            assert result == "env-key-456"
            mock_audit_logger.log_credential_access.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_credential_not_found(self, credential_manager, mock_vault_client, mock_audit_logger):
        """Test handling of missing credentials."""
        # Setup
        mock_vault_client.get_secret.return_value = {}
        mock_audit_logger.log_credential_access = AsyncMock()

        # Execute & Verify
        with pytest.raises(CredentialError) as exc_info:
            await credential_manager.get_credential("missing", "api_key")

        assert exc_info.value.error_code == "CREDENTIAL_NOT_FOUND"
        mock_audit_logger.log_credential_access.assert_called()

    @pytest.mark.asyncio
    async def test_store_credential_in_vault(self, credential_manager, mock_vault_client, mock_audit_logger):
        """Test storing credential in vault."""
        # Setup
        mock_vault_client.get_secret.return_value = {}
        mock_vault_client.put_secret.return_value = True
        mock_audit_logger.log_credential_access = AsyncMock()

        # Execute
        result = await credential_manager.store_credential("test_service", "api_key", "new-key-789")

        # Verify
        assert result is True
        mock_vault_client.put_secret.assert_called_once()
        mock_audit_logger.log_credential_access.assert_called_once()

    @pytest.mark.asyncio
    async def test_rotate_credential(self, credential_manager, mock_audit_logger):
        """Test credential rotation."""
        # Setup
        mock_audit_logger.log_credential_access = AsyncMock()

        with patch.object(credential_manager, '_generate_new_credential', return_value="rotated-key-123"):
            with patch.object(credential_manager, 'store_credential', return_value=True):
                # Execute
                result = await credential_manager.rotate_credential("test_service", "api_key")

                # Verify
                assert result is True
                mock_audit_logger.log_credential_access.assert_called()

    @pytest.mark.asyncio
    async def test_rotate_credential_generation_failure(self, credential_manager, mock_audit_logger):
        """Test credential rotation when generation fails."""
        # Setup
        mock_audit_logger.log_credential_access = AsyncMock()

        with patch.object(credential_manager, '_generate_new_credential', return_value=None):
            # Execute & Verify
            with pytest.raises(CredentialRotationError):
                await credential_manager.rotate_credential("test_service", "api_key")

    def test_mask_sensitive_data(self, credential_manager):
        """Test masking of sensitive data."""
        # Setup
        data = {
            "api_key": "secret-key-12345678",
            "password": "short",
            "username": "testuser",
            "nested": {
                "token": "bearer-token-87654321",
                "public_info": "visible"
            }
        }

        # Execute
        masked = credential_manager.mask_sensitive_data(data)

        # Verify
        assert masked["api_key"] == "secr***5678"  # First 4 and last 4 chars
        assert masked["password"] == "***MASKED***"  # Too short for partial masking
        assert masked["username"] == "testuser"
        assert masked["nested"]["token"] == "bear***4321"  # First 4 and last 4 chars
        assert masked["nested"]["public_info"] == "visible"

    @pytest.mark.asyncio
    async def test_list_credentials(self, credential_manager):
        """Test listing credentials with metadata."""
        # Setup
        credential_manager._credential_metadata["service1:api_key"] = CredentialMetadata(
            service="service1",
            credential_type="api_key",
            created_at=datetime.utcnow(),
            expires_at=None,
            rotation_interval=None,
            last_rotated=datetime.utcnow(),
            rotation_count=1
        )

        # Execute
        credentials = await credential_manager.list_credentials()

        # Verify
        assert len(credentials) == 1
        assert credentials[0]["service"] == "service1"
        assert credentials[0]["credential_type"] == "api_key"
        assert credentials[0]["rotation_count"] == 1

    @pytest.mark.asyncio
    async def test_revoke_credential(self, credential_manager, mock_vault_client, mock_audit_logger):
        """Test credential revocation."""
        # Setup
        mock_vault_client.get_secret.return_value = {"api_key": "test-key"}
        mock_vault_client.put_secret.return_value = True
        mock_audit_logger.log_credential_access = AsyncMock()

        # Add credential to cache
        credential_manager._credential_cache["service:api_key"] = {
            "value": "test-key",
            "source": "vault"
        }

        # Execute
        result = await credential_manager.revoke_credential("service", "api_key")

        # Verify
        assert result is True
        assert "service:api_key" not in credential_manager._credential_cache
        mock_audit_logger.log_credential_access.assert_called_once()


class TestHashiCorpVaultClient:
    """Test HashiCorp Vault client."""

    @pytest.fixture
    def vault_client(self):
        return HashiCorpVaultClient("https://vault.example.com", "test-token")

    @pytest.mark.asyncio
    async def test_get_secret_success(self, vault_client):
        """Test successful secret retrieval."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": {"data": {"api_key": "secret-value"}}
        })

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            result = await vault_client.get_secret("test/path")

            assert result == {"api_key": "secret-value"}

    @pytest.mark.asyncio
    async def test_get_secret_not_found(self, vault_client):
        """Test secret not found."""
        mock_response = Mock()
        mock_response.status = 404

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            result = await vault_client.get_secret("test/path")

            assert result == {}

    @pytest.mark.asyncio
    async def test_get_secret_error(self, vault_client):
        """Test vault connection error."""
        mock_response = Mock()
        mock_response.status = 500

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response

            with pytest.raises(VaultConnectionError):
                await vault_client.get_secret("test/path")

    @pytest.mark.asyncio
    async def test_put_secret_success(self, vault_client):
        """Test successful secret storage."""
        mock_response = Mock()
        mock_response.status = 200

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response

            result = await vault_client.put_secret("test/path", {"key": "value"})

            assert result is True

    @pytest.mark.asyncio
    async def test_delete_secret_success(self, vault_client):
        """Test successful secret deletion."""
        mock_response = Mock()
        mock_response.status = 204

        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.delete.return_value.__aenter__.return_value = mock_response

            result = await vault_client.delete_secret("test/path")

            assert result is True


class TestInMemoryRedis:
    """Test in-memory Redis fallback."""

    @pytest.fixture
    def redis_client(self):
        return InMemoryRedis()

    @pytest.mark.asyncio
    async def test_set_and_get(self, redis_client):
        """Test basic set and get operations."""
        await redis_client.set("test_key", "test_value")
        result = await redis_client.get("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_incr(self, redis_client):
        """Test increment operation."""
        result1 = await redis_client.incr("counter")
        result2 = await redis_client.incr("counter")

        assert result1 == 1
        assert result2 == 2

    @pytest.mark.asyncio
    async def test_expire(self, redis_client):
        """Test expiration functionality."""
        await redis_client.set("temp_key", "temp_value")
        await redis_client.expire("temp_key", 1)

        # Should still exist immediately
        result = await redis_client.get("temp_key")
        assert result == "temp_value"

        # Simulate time passing
        import time
        time.sleep(1.1)

        result = await redis_client.get("temp_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_sorted_set_operations(self, redis_client):
        """Test sorted set operations."""
        # Add members
        await redis_client.zadd("test_set", {"member1": 1.0, "member2": 2.0})

        # Check cardinality
        count = await redis_client.zcard("test_set")
        assert count == 2

        # Remove by score range
        removed = await redis_client.zremrangebyscore("test_set", 1.0, 1.0)
        assert removed == 1

        # Check cardinality after removal
        count = await redis_client.zcard("test_set")
        assert count == 1


@pytest.mark.asyncio
async def test_credential_manager_integration():
    """Integration test for credential manager with all components."""
    # Setup
    audit_logger = Mock(spec=AuditLogger)
    audit_logger.log_credential_access = AsyncMock()

    vault_client = Mock()
    vault_client.get_secret = AsyncMock(return_value={"api_key": "vault-key"})
    vault_client.put_secret = AsyncMock(return_value=True)

    manager = SecureCredentialManager(
        vault_client=vault_client,
        audit_logger=audit_logger
    )

    # Test full workflow
    # 1. Store credential
    await manager.store_credential("test_service", "api_key", "new-key")

    # 2. Retrieve credential
    result = await manager.get_credential("test_service", "api_key")
    assert result == "vault-key"  # Should get from vault, not cache

    # 3. List credentials
    credentials = await manager.list_credentials("test_service")
    assert len(credentials) >= 0

    # 4. Revoke credential
    await manager.revoke_credential("test_service", "api_key")

    # Verify audit logging was called
    assert audit_logger.log_credential_access.call_count >= 3
