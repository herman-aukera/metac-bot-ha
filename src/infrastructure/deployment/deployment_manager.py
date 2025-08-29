"""
Production deployment manager for AI forecasting bot.
Handles deployment orchestration, health checks, and rollback capabilities.
"""

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import requests


class DeploymentStatus(Enum):
    """Deployment status enumeration."""

    PENDING = "pending"
    DEPLOYING = "deploying"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""

    image_tag: str
    environment: str
    replicas: int = 1
    health_check_url: str = "http://localhost:8080/health"
    health_check_timeout: int = 300
    health_check_interval: int = 10
    rollback_on_failure: bool = True
    backup_enabled: bool = True


@dataclass
class DeploymentInfo:
    """Deployment information."""

    id: str
    config: DeploymentConfig
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    health_checks_passed: int = 0
    health_checks_failed: int = 0


class HealthChecker:
    """Handles health checking for deployments."""

    def __init__(self, url: str, timeout: int = 10):
        self.url = url
        self.timeout = timeout

    def check_health(self) -> tuple[bool, str]:
        """Perform health check."""
        try:
            response = requests.get(self.url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    return True, "Health check passed"
                else:
                    return False, f"Service reports unhealthy: {data}"
            else:
                return False, f"HTTP {response.status_code}: {response.text}"
        except requests.exceptions.RequestException as e:
            return False, f"Health check failed: {str(e)}"

    def wait_for_health(self, max_attempts: int, interval: int = 10) -> bool:
        """Wait for service to become healthy."""
        for attempt in range(max_attempts):
            healthy, message = self.check_health()
            if healthy:
                logging.info(f"Health check passed on attempt {attempt + 1}")
                return True

            logging.warning(
                f"Health check attempt {attempt + 1}/{max_attempts} failed: {message}"
            )
            if attempt < max_attempts - 1:
                time.sleep(interval)

        return False


class BackupManager:
    """Manages deployment backups."""

    def __init__(self, backup_dir: str = "./backups"):
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)

    def create_backup(self, deployment_id: str) -> str:
        """Create deployment backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{deployment_id}_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_name)

        try:
            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)

            # Backup configuration files
            config_files = ["docker-compose.yml", ".env", "config/config.prod.yaml"]

            for config_file in config_files:
                if os.path.exists(config_file):
                    subprocess.run(
                        [
                            "cp",
                            config_file,
                            os.path.join(backup_path, os.path.basename(config_file)),
                        ],
                        check=True,
                    )

            # Backup application data
            if os.path.exists("data"):
                subprocess.run(
                    ["cp", "-r", "data", os.path.join(backup_path, "data")], check=True
                )

            # Backup logs
            if os.path.exists("logs"):
                subprocess.run(
                    ["cp", "-r", "logs", os.path.join(backup_path, "logs")], check=True
                )

            # Create backup metadata
            metadata = {
                "deployment_id": deployment_id,
                "timestamp": timestamp,
                "backup_path": backup_path,
                "created_at": datetime.now().isoformat(),
            }

            with open(os.path.join(backup_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            logging.info(f"Backup created: {backup_path}")
            return backup_path

        except subprocess.CalledProcessError as e:
            logging.error(f"Backup creation failed: {e}")
            raise

    def restore_backup(self, backup_path: str) -> bool:
        """Restore from backup."""
        try:
            if not os.path.exists(backup_path):
                logging.error(f"Backup path does not exist: {backup_path}")
                return False

            # Restore configuration files
            config_files = ["docker-compose.yml", ".env", "config.prod.yaml"]

            for config_file in config_files:
                backup_file = os.path.join(backup_path, os.path.basename(config_file))
                if os.path.exists(backup_file):
                    if config_file.startswith("config/"):
                        os.makedirs("config", exist_ok=True)
                    subprocess.run(["cp", backup_file, config_file], check=True)

            # Restore data
            backup_data = os.path.join(backup_path, "data")
            if os.path.exists(backup_data):
                if os.path.exists("data"):
                    subprocess.run(["rm", "-rf", "data"], check=True)
                subprocess.run(["cp", "-r", backup_data, "data"], check=True)

            logging.info(f"Backup restored from: {backup_path}")
            return True

        except subprocess.CalledProcessError as e:
            logging.error(f"Backup restoration failed: {e}")
            return False

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups."""
        backups = []

        for item in os.listdir(self.backup_dir):
            backup_path = os.path.join(self.backup_dir, item)
            metadata_file = os.path.join(backup_path, "metadata.json")

            if os.path.isdir(backup_path) and os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    backups.append(metadata)
                except Exception as e:
                    logging.warning(f"Failed to read backup metadata: {e}")

        return sorted(backups, key=lambda x: x["created_at"], reverse=True)


class DeploymentOrchestrator:
    """Orchestrates deployment process."""

    def __init__(self):
        self.deployments: Dict[str, DeploymentInfo] = {}
        self.backup_manager = BackupManager()
        self._lock = threading.Lock()

    def deploy(self, config: DeploymentConfig) -> str:
        """Deploy new version."""
        deployment_id = f"deploy_{int(time.time())}"

        deployment_info = DeploymentInfo(
            id=deployment_id,
            config=config,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now(),
        )

        with self._lock:
            self.deployments[deployment_id] = deployment_info

        # Start deployment in background thread
        deployment_thread = threading.Thread(
            target=self._execute_deployment, args=(deployment_id,), daemon=True
        )
        deployment_thread.start()

        return deployment_id

    def _execute_deployment(self, deployment_id: str):
        """Execute deployment process."""
        deployment = self.deployments[deployment_id]

        try:
            # Update status
            deployment.status = DeploymentStatus.DEPLOYING
            logging.info(f"Starting deployment {deployment_id}")

            # Create backup if enabled
            backup_path = None
            if deployment.config.backup_enabled:
                backup_path = self.backup_manager.create_backup(deployment_id)
                self._save_last_good_deployment(deployment_id, backup_path)

            # Execute deployment steps
            self._execute_deployment_steps(deployment)

            # Perform health checks
            if self._perform_health_checks(deployment):
                deployment.status = DeploymentStatus.HEALTHY
                deployment.end_time = datetime.now()
                logging.info(f"Deployment {deployment_id} completed successfully")
            else:
                raise Exception("Health checks failed")

        except Exception as e:
            logging.error(f"Deployment {deployment_id} failed: {e}")
            deployment.status = DeploymentStatus.FAILED
            deployment.error_message = str(e)
            deployment.end_time = datetime.now()

            # Rollback if enabled
            if deployment.config.rollback_on_failure:
                self._rollback_deployment(deployment_id)

    def _execute_deployment_steps(self, deployment: DeploymentInfo):
        """Execute deployment steps."""
        config = deployment.config

        # Update docker-compose with new image
        self._update_compose_file(config.image_tag)

        # Deploy using docker-compose
        if config.environment == "production":
            # Use blue-green deployment
            self._blue_green_deploy(config.image_tag)
        else:
            # Standard deployment
            self._standard_deploy()

    def _update_compose_file(self, image_tag: str):
        """Update docker-compose file with new image tag."""
        compose_file = "docker-compose.yml"

        # Read current compose file
        with open(compose_file, "r") as f:
            content = f.read()

        # Update image tag
        # This is a simple replacement - in production, use proper YAML parsing
        updated_content = content.replace(
            "image: ghcr.io/your-org/ai-forecasting-bot:latest",
            f"image: ghcr.io/your-org/ai-forecasting-bot:{image_tag}",
        )

        # Write updated compose file
        with open(compose_file, "w") as f:
            f.write(updated_content)

    def _blue_green_deploy(self, image_tag: str):
        """Execute blue-green deployment."""
        script_path = "./scripts/blue-green-deploy.sh"

        if not os.path.exists(script_path):
            raise Exception("Blue-green deployment script not found")

        result = subprocess.run(
            [script_path, image_tag], capture_output=True, text=True
        )

        if result.returncode != 0:
            raise Exception(f"Blue-green deployment failed: {result.stderr}")

    def _standard_deploy(self):
        """Execute standard deployment."""
        # Stop current containers
        subprocess.run(["docker-compose", "down"], check=True)

        # Pull new images
        subprocess.run(["docker-compose", "pull"], check=True)

        # Start new containers
        subprocess.run(["docker-compose", "up", "-d"], check=True)

    def _perform_health_checks(self, deployment: DeploymentInfo) -> bool:
        """Perform health checks."""
        config = deployment.config
        health_checker = HealthChecker(config.health_check_url)

        max_attempts = config.health_check_timeout // config.health_check_interval

        success = health_checker.wait_for_health(
            max_attempts, config.health_check_interval
        )

        if success:
            deployment.health_checks_passed += 1
        else:
            deployment.health_checks_failed += 1

        return success

    def _rollback_deployment(self, deployment_id: str):
        """Rollback failed deployment."""
        deployment = self.deployments[deployment_id]
        deployment.status = DeploymentStatus.ROLLING_BACK

        try:
            # Get last good deployment
            last_good = self._get_last_good_deployment()

            if last_good:
                logging.info(f"Rolling back to: {last_good}")

                # Restore backup
                if self.backup_manager.restore_backup(last_good["backup_path"]):
                    # Restart services
                    subprocess.run(["docker-compose", "down"], check=True)
                    subprocess.run(["docker-compose", "up", "-d"], check=True)

                    # Verify rollback
                    health_checker = HealthChecker(deployment.config.health_check_url)
                    if health_checker.wait_for_health(10, 10):
                        logging.info(
                            f"Rollback successful for deployment {deployment_id}"
                        )
                        deployment.status = DeploymentStatus.HEALTHY
                    else:
                        logging.error(
                            f"Rollback health check failed for deployment {deployment_id}"
                        )
                        deployment.status = DeploymentStatus.FAILED
                else:
                    logging.error(
                        f"Failed to restore backup for deployment {deployment_id}"
                    )
                    deployment.status = DeploymentStatus.FAILED
            else:
                logging.error(
                    f"No previous deployment found for rollback of {deployment_id}"
                )
                deployment.status = DeploymentStatus.FAILED

        except Exception as e:
            logging.error(f"Rollback failed for deployment {deployment_id}: {e}")
            deployment.status = DeploymentStatus.FAILED

        deployment.end_time = datetime.now()

    def _save_last_good_deployment(self, deployment_id: str, backup_path: str):
        """Save last good deployment info."""
        last_good_file = "./backups/last_good_deployment.json"

        data = {
            "deployment_id": deployment_id,
            "backup_path": backup_path,
            "timestamp": datetime.now().isoformat(),
        }

        with open(last_good_file, "w") as f:
            json.dump(data, f, indent=2)

    def _get_last_good_deployment(self) -> Optional[Dict[str, Any]]:
        """Get last good deployment info."""
        last_good_file = "./backups/last_good_deployment.json"

        if os.path.exists(last_good_file):
            with open(last_good_file, "r") as f:
                return json.load(f)

        return None

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentInfo]:
        """Get deployment status."""
        return self.deployments.get(deployment_id)

    def list_deployments(self) -> List[DeploymentInfo]:
        """List all deployments."""
        return list(self.deployments.values())

    def cleanup_old_deployments(self, max_age_days: int = 30):
        """Clean up old deployment records."""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)

        to_remove = []
        for deployment_id, deployment in self.deployments.items():
            if deployment.end_time and deployment.end_time < cutoff_date:
                to_remove.append(deployment_id)

        for deployment_id in to_remove:
            del self.deployments[deployment_id]

        logging.info(f"Cleaned up {len(to_remove)} old deployment records")


class ProductionDeploymentManager:
    """Main production deployment manager."""

    def __init__(self):
        self.orchestrator = DeploymentOrchestrator()
        self.monitoring_enabled = True

    def deploy_to_production(self, image_tag: str) -> str:
        """Deploy to production environment."""
        config = DeploymentConfig(
            image_tag=image_tag,
            environment="production",
            replicas=1,
            health_check_url="http://localhost:8080/health",
            health_check_timeout=300,
            health_check_interval=10,
            rollback_on_failure=True,
            backup_enabled=True,
        )

        return self.orchestrator.deploy(config)

    def deploy_to_staging(self, image_tag: str) -> str:
        """Deploy to staging environment."""
        config = DeploymentConfig(
            image_tag=image_tag,
            environment="staging",
            replicas=1,
            health_check_url="http://staging.localhost:8080/health",
            health_check_timeout=180,
            health_check_interval=10,
            rollback_on_failure=True,
            backup_enabled=False,
        )

        return self.orchestrator.deploy(config)

    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status."""
        deployment = self.orchestrator.get_deployment_status(deployment_id)

        if not deployment:
            return {"error": "Deployment not found"}

        return {
            "id": deployment.id,
            "status": deployment.status.value,
            "environment": deployment.config.environment,
            "image_tag": deployment.config.image_tag,
            "start_time": deployment.start_time.isoformat(),
            "end_time": (
                deployment.end_time.isoformat() if deployment.end_time else None
            ),
            "error_message": deployment.error_message,
            "health_checks": {
                "passed": deployment.health_checks_passed,
                "failed": deployment.health_checks_failed,
            },
        }

    def list_recent_deployments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent deployments."""
        deployments = self.orchestrator.list_deployments()
        deployments.sort(key=lambda x: x.start_time, reverse=True)

        return [
            {
                "id": d.id,
                "status": d.status.value,
                "environment": d.config.environment,
                "image_tag": d.config.image_tag,
                "start_time": d.start_time.isoformat(),
                "end_time": d.end_time.isoformat() if d.end_time else None,
            }
            for d in deployments[:limit]
        ]

    def emergency_rollback(self) -> bool:
        """Perform emergency rollback."""
        try:
            # Execute rollback script
            result = subprocess.run(
                ["./scripts/rollback.sh", "Emergency rollback"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                logging.info("Emergency rollback completed successfully")
                return True
            else:
                logging.error(f"Emergency rollback failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logging.error("Emergency rollback timed out")
            return False
        except Exception as e:
            logging.error(f"Emergency rollback error: {e}")
            return False
