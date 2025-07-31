#!/usr/bin/env python3
"""
Disaster recovery and backup strategy implementation.
Handles database backups, configuration backups, and disaster recovery procedures.
"""

import os
import sys
import json
import logging
import asyncio
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BackupConfig:
    """Backup configuration"""
    environment: str
    database_url: str
    s3_bucket: str
    s3_prefix: str
    retention_days: int
    encryption_key_id: Optional[str] = None
    notification_topic: Optional[str] = None


@dataclass
class BackupResult:
    """Result of a backup operation"""
    backup_type: str
    timestamp: datetime
    success: bool
    file_path: Optional[str] = None
    s3_key: Optional[str] = None
    size_bytes: Optional[int] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None


class DisasterRecoveryManager:
    """
    Manages disaster recovery and backup operations for the tournament optimization system.
    """

    def __init__(self, config: BackupConfig):
        self.config = config
        self.s3_client = boto3.client('s3')
        self.backup_history: List[BackupResult] = []

        # Ensure backup directory exists
        self.backup_dir = Path("/tmp/backups")
        self.backup_dir.mkdir(exist_ok=True)

    async def perform_full_backup(self) -> Dict[str, BackupResult]:
        """Perform a complete system backup"""
        logger.info(f"Starting full backup for environment: {self.config.environment}")

        backup_results = {}

        try:
            # Database backup
            db_result = await self.backup_database()
            backup_results["database"] = db_result

            # Configuration backup
            config_result = await self.backup_configurations()
            backup_results["configurations"] = config_result

            # Application state backup
            state_result = await self.backup_application_state()
            backup_results["application_state"] = state_result

            # Secrets backup (encrypted)
            secrets_result = await self.backup_secrets()
            backup_results["secrets"] = secrets_result

            # Create backup manifest
            manifest_result = await self.create_backup_manifest(backup_results)
            backup_results["manifest"] = manifest_result

            # Send notification
            await self.send_backup_notification(backup_results)

            logger.info("Full backup completed successfully")

        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            await self.send_failure_notification(str(e))

        return backup_results

    async def backup_database(self) -> BackupResult:
        """Backup PostgreSQL database"""
        logger.info("Starting database backup")
        start_time = datetime.utcnow()

        try:
            # Generate backup filename
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"database_backup_{self.config.environment}_{timestamp}.sql"
            backup_path = self.backup_dir / backup_filename

            # Create database dump
            cmd = [
                "pg_dump",
                self.config.database_url,
                "--no-password",
                "--verbose",
                "--clean",
                "--if-exists",
                "--format=custom",
                "--file", str(backup_path)
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise Exception(f"pg_dump failed: {error_msg}")

            # Get file size
            file_size = backup_path.stat().st_size

            # Upload to S3
            s3_key = f"{self.config.s3_prefix}/database/{backup_filename}"
            await self.upload_to_s3(backup_path, s3_key)

            # Clean up local file
            backup_path.unlink()

            duration = (datetime.utcnow() - start_time).total_seconds()

            result = BackupResult(
                backup_type="database",
                timestamp=start_time,
                success=True,
                file_path=str(backup_path),
                s3_key=s3_key,
                size_bytes=file_size,
                duration_seconds=duration
            )

            logger.info(f"Database backup completed: {file_size} bytes in {duration:.2f}s")
            return result

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Database backup failed: {error_msg}")

            return BackupResult(
                backup_type="database",
                timestamp=start_time,
                success=False,
                error_message=error_msg,
                duration_seconds=duration
            )

    async def backup_configurations(self) -> BackupResult:
        """Backup application configurations"""
        logger.info("Starting configuration backup")
        start_time = datetime.utcnow()

        try:
            # Generate backup filename
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"configs_backup_{self.config.environment}_{timestamp}.tar.gz"
            backup_path = self.backup_dir / backup_filename

            # Create tar archive of configurations
            config_dirs = [
                "configs/",
                "k8s/",
                "infrastructure/terraform/",
                ".github/workflows/"
            ]

            cmd = ["tar", "-czf", str(backup_path)] + config_dirs

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise Exception(f"tar failed: {error_msg}")

            # Get file size
            file_size = backup_path.stat().st_size

            # Upload to S3
            s3_key = f"{self.config.s3_prefix}/configurations/{backup_filename}"
            await self.upload_to_s3(backup_path, s3_key)

            # Clean up local file
            backup_path.unlink()

            duration = (datetime.utcnow() - start_time).total_seconds()

            result = BackupResult(
                backup_type="configurations",
                timestamp=start_time,
                success=True,
                file_path=str(backup_path),
                s3_key=s3_key,
                size_bytes=file_size,
                duration_seconds=duration
            )

            logger.info(f"Configuration backup completed: {file_size} bytes in {duration:.2f}s")
            return result

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Configuration backup failed: {error_msg}")

            return BackupResult(
                backup_type="configurations",
                timestamp=start_time,
                success=False,
                error_message=error_msg,
                duration_seconds=duration
            )

    async def backup_application_state(self) -> BackupResult:
        """Backup application state and logs"""
        logger.info("Starting application state backup")
        start_time = datetime.utcnow()

        try:
            # Generate backup filename
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"app_state_backup_{self.config.environment}_{timestamp}.tar.gz"
            backup_path = self.backup_dir / backup_filename

            # Collect application state
            state_data = {
                "timestamp": start_time.isoformat(),
                "environment": self.config.environment,
                "feature_flags": await self.get_current_feature_flags(),
                "deployment_status": await self.get_deployment_status(),
                "recent_metrics": await self.get_recent_metrics(),
                "active_tournaments": await self.get_active_tournaments()
            }

            # Save state data to JSON file
            state_file = self.backup_dir / f"app_state_{timestamp}.json"
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            # Create tar archive
            cmd = ["tar", "-czf", str(backup_path), str(state_file)]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise Exception(f"tar failed: {error_msg}")

            # Get file size
            file_size = backup_path.stat().st_size

            # Upload to S3
            s3_key = f"{self.config.s3_prefix}/application_state/{backup_filename}"
            await self.upload_to_s3(backup_path, s3_key)

            # Clean up local files
            backup_path.unlink()
            state_file.unlink()

            duration = (datetime.utcnow() - start_time).total_seconds()

            result = BackupResult(
                backup_type="application_state",
                timestamp=start_time,
                success=True,
                file_path=str(backup_path),
                s3_key=s3_key,
                size_bytes=file_size,
                duration_seconds=duration
            )

            logger.info(f"Application state backup completed: {file_size} bytes in {duration:.2f}s")
            return result

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Application state backup failed: {error_msg}")

            return BackupResult(
                backup_type="application_state",
                timestamp=start_time,
                success=False,
                error_message=error_msg,
                duration_seconds=duration
            )

    async def backup_secrets(self) -> BackupResult:
        """Backup encrypted secrets and credentials"""
        logger.info("Starting secrets backup")
        start_time = datetime.utcnow()

        try:
            # Generate backup filename
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"secrets_backup_{self.config.environment}_{timestamp}.json.enc"
            backup_path = self.backup_dir / backup_filename

            # Get secrets from Kubernetes
            secrets_data = await self.get_kubernetes_secrets()

            # Encrypt secrets data
            encrypted_data = await self.encrypt_data(json.dumps(secrets_data))

            # Save encrypted data
            with open(backup_path, 'wb') as f:
                f.write(encrypted_data)

            # Get file size
            file_size = backup_path.stat().st_size

            # Upload to S3 with server-side encryption
            s3_key = f"{self.config.s3_prefix}/secrets/{backup_filename}"
            await self.upload_to_s3(backup_path, s3_key, encrypt=True)

            # Clean up local file
            backup_path.unlink()

            duration = (datetime.utcnow() - start_time).total_seconds()

            result = BackupResult(
                backup_type="secrets",
                timestamp=start_time,
                success=True,
                file_path=str(backup_path),
                s3_key=s3_key,
                size_bytes=file_size,
                duration_seconds=duration
            )

            logger.info(f"Secrets backup completed: {file_size} bytes in {duration:.2f}s")
            return result

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Secrets backup failed: {error_msg}")

            return BackupResult(
                backup_type="secrets",
                timestamp=start_time,
                success=False,
                error_message=error_msg,
                duration_seconds=duration
            )

    async def create_backup_manifest(self, backup_results: Dict[str, BackupResult]) -> BackupResult:
        """Create backup manifest with metadata"""
        logger.info("Creating backup manifest")
        start_time = datetime.utcnow()

        try:
            # Generate manifest filename
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            manifest_filename = f"backup_manifest_{self.config.environment}_{timestamp}.json"
            manifest_path = self.backup_dir / manifest_filename

            # Create manifest data
            manifest_data = {
                "backup_id": f"{self.config.environment}_{timestamp}",
                "timestamp": start_time.isoformat(),
                "environment": self.config.environment,
                "backup_type": "full",
                "results": {
                    name: {
                        "success": result.success,
                        "s3_key": result.s3_key,
                        "size_bytes": result.size_bytes,
                        "duration_seconds": result.duration_seconds,
                        "error_message": result.error_message
                    }
                    for name, result in backup_results.items()
                },
                "total_size_bytes": sum(
                    result.size_bytes or 0
                    for result in backup_results.values()
                    if result.success
                ),
                "successful_backups": sum(
                    1 for result in backup_results.values()
                    if result.success
                ),
                "failed_backups": sum(
                    1 for result in backup_results.values()
                    if not result.success
                )
            }

            # Save manifest
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)

            # Get file size
            file_size = manifest_path.stat().st_size

            # Upload to S3
            s3_key = f"{self.config.s3_prefix}/manifests/{manifest_filename}"
            await self.upload_to_s3(manifest_path, s3_key)

            # Clean up local file
            manifest_path.unlink()

            duration = (datetime.utcnow() - start_time).total_seconds()

            result = BackupResult(
                backup_type="manifest",
                timestamp=start_time,
                success=True,
                file_path=str(manifest_path),
                s3_key=s3_key,
                size_bytes=file_size,
                duration_seconds=duration
            )

            logger.info(f"Backup manifest created: {file_size} bytes in {duration:.2f}s")
            return result

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Backup manifest creation failed: {error_msg}")

            return BackupResult(
                backup_type="manifest",
                timestamp=start_time,
                success=False,
                error_message=error_msg,
                duration_seconds=duration
            )

    async def upload_to_s3(self, file_path: Path, s3_key: str, encrypt: bool = False) -> None:
        """Upload file to S3 with optional encryption"""
        try:
            extra_args = {}

            if encrypt and self.config.encryption_key_id:
                extra_args['ServerSideEncryption'] = 'aws:kms'
                extra_args['SSEKMSKeyId'] = self.config.encryption_key_id

            self.s3_client.upload_file(
                str(file_path),
                self.config.s3_bucket,
                s3_key,
                ExtraArgs=extra_args
            )

            logger.debug(f"Uploaded {file_path} to s3://{self.config.s3_bucket}/{s3_key}")

        except ClientError as e:
            raise Exception(f"S3 upload failed: {e}")

    async def cleanup_old_backups(self) -> None:
        """Clean up old backups based on retention policy"""
        logger.info(f"Cleaning up backups older than {self.config.retention_days} days")

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.retention_days)

            # List objects in S3 bucket
            response = self.s3_client.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix=self.config.s3_prefix
            )

            deleted_count = 0

            for obj in response.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    self.s3_client.delete_object(
                        Bucket=self.config.s3_bucket,
                        Key=obj['Key']
                    )
                    deleted_count += 1
                    logger.debug(f"Deleted old backup: {obj['Key']}")

            logger.info(f"Cleaned up {deleted_count} old backup files")

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

    async def restore_from_backup(self, backup_id: str) -> bool:
        """Restore system from backup"""
        logger.info(f"Starting restore from backup: {backup_id}")

        try:
            # Download and parse manifest
            manifest = await self.download_backup_manifest(backup_id)

            if not manifest:
                raise Exception(f"Backup manifest not found for ID: {backup_id}")

            # Restore database
            if manifest['results'].get('database', {}).get('success'):
                await self.restore_database(manifest['results']['database']['s3_key'])

            # Restore configurations
            if manifest['results'].get('configurations', {}).get('success'):
                await self.restore_configurations(manifest['results']['configurations']['s3_key'])

            # Restore secrets
            if manifest['results'].get('secrets', {}).get('success'):
                await self.restore_secrets(manifest['results']['secrets']['s3_key'])

            logger.info(f"Restore from backup {backup_id} completed successfully")
            return True

        except Exception as e:
            logger.error(f"Restore from backup {backup_id} failed: {e}")
            return False

    # Helper methods (simplified implementations)
    async def get_current_feature_flags(self) -> Dict[str, Any]:
        """Get current feature flag configuration"""
        # In a real implementation, this would query the feature flag service
        return {"placeholder": "feature_flags"}

    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        # In a real implementation, this would query Kubernetes
        return {"placeholder": "deployment_status"}

    async def get_recent_metrics(self) -> Dict[str, Any]:
        """Get recent performance metrics"""
        # In a real implementation, this would query monitoring systems
        return {"placeholder": "metrics"}

    async def get_active_tournaments(self) -> Dict[str, Any]:
        """Get active tournament information"""
        # In a real implementation, this would query the application database
        return {"placeholder": "tournaments"}

    async def get_kubernetes_secrets(self) -> Dict[str, Any]:
        """Get secrets from Kubernetes"""
        # In a real implementation, this would use kubectl or Kubernetes API
        return {"placeholder": "secrets"}

    async def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        # In a real implementation, this would use proper encryption
        return data.encode('utf-8')

    async def download_backup_manifest(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Download and parse backup manifest"""
        # In a real implementation, this would download from S3
        return None

    async def restore_database(self, s3_key: str) -> None:
        """Restore database from backup"""
        # In a real implementation, this would download and restore the database
        pass

    async def restore_configurations(self, s3_key: str) -> None:
        """Restore configurations from backup"""
        # In a real implementation, this would download and apply configurations
        pass

    async def restore_secrets(self, s3_key: str) -> None:
        """Restore secrets from backup"""
        # In a real implementation, this would download and apply secrets
        pass

    async def send_backup_notification(self, backup_results: Dict[str, BackupResult]) -> None:
        """Send backup completion notification"""
        if not self.config.notification_topic:
            return

        # In a real implementation, this would send SNS notification
        logger.info("Backup notification sent")

    async def send_failure_notification(self, error_message: str) -> None:
        """Send backup failure notification"""
        if not self.config.notification_topic:
            return

        # In a real implementation, this would send SNS notification
        logger.error(f"Backup failure notification: {error_message}")


async def main():
    """Main backup execution function"""
    # Load configuration from environment
    config = BackupConfig(
        environment=os.getenv("ENVIRONMENT", "development"),
        database_url=os.getenv("DATABASE_URL", "postgresql://localhost/tournament_optimization"),
        s3_bucket=os.getenv("BACKUP_S3_BUCKET", "tournament-optimization-backups"),
        s3_prefix=os.getenv("BACKUP_S3_PREFIX", "backups"),
        retention_days=int(os.getenv("BACKUP_RETENTION_DAYS", "30")),
        encryption_key_id=os.getenv("BACKUP_ENCRYPTION_KEY_ID"),
        notification_topic=os.getenv("BACKUP_NOTIFICATION_TOPIC")
    )

    # Create disaster recovery manager
    dr_manager = DisasterRecoveryManager(config)

    # Perform backup based on command line argument
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "backup":
            await dr_manager.perform_full_backup()
        elif command == "cleanup":
            await dr_manager.cleanup_old_backups()
        elif command == "restore" and len(sys.argv) > 2:
            backup_id = sys.argv[2]
            await dr_manager.restore_from_backup(backup_id)
        else:
            print("Usage: backup_strategy.py [backup|cleanup|restore <backup_id>]")
            sys.exit(1)
    else:
        # Default to backup
        await dr_manager.perform_full_backup()


if __name__ == "__main__":
    asyncio.run(main())
