"""
Graceful shutdown handler for production deployments.
Ensures proper cleanup and health check coordination during deployments.
"""

import asyncio
import signal
import sys
import logging
import subprocess
import time
from typing import Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class GracefulShutdownHandler:
    """Handles graceful shutdown for production deployments."""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.main_process: Optional[subprocess.Popen] = None
        self.monitoring_process: Optional[subprocess.Popen] = None
        self.shutdown_timeout = 30  # seconds

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()

    async def start_services(self):
        """Start main application and monitoring services."""
        try:
            # Start monitoring service
            logger.info("Starting monitoring service...")
            self.monitoring_process = subprocess.Popen([
                sys.executable, "-m", "src.infrastructure.monitoring.metrics_collector"
            ])

            # Wait a bit for monitoring to start
            await asyncio.sleep(2)

            # Start main application
            logger.info("Starting main application...")
            self.main_process = subprocess.Popen([
                sys.executable, "main_agent.py"
            ])

            logger.info("All services started successfully")

        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            await self.shutdown_services()
            raise

    async def shutdown_services(self):
        """Gracefully shutdown all services."""
        logger.info("Initiating graceful shutdown...")

        # Stop accepting new requests by updating health check
        await self._update_health_status(healthy=False)

        # Give load balancer time to detect unhealthy status
        logger.info("Waiting for load balancer to detect unhealthy status...")
        await asyncio.sleep(10)

        # Shutdown main application
        if self.main_process:
            logger.info("Shutting down main application...")
            self.main_process.terminate()

            try:
                self.main_process.wait(timeout=self.shutdown_timeout)
                logger.info("Main application shutdown gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Main application did not shutdown gracefully, forcing termination")
                self.main_process.kill()
                self.main_process.wait()

        # Shutdown monitoring service
        if self.monitoring_process:
            logger.info("Shutting down monitoring service...")
            self.monitoring_process.terminate()

            try:
                self.monitoring_process.wait(timeout=10)
                logger.info("Monitoring service shutdown gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Monitoring service did not shutdown gracefully, forcing termination")
                self.monitoring_process.kill()
                self.monitoring_process.wait()

        logger.info("All services shutdown completed")

    async def _update_health_status(self, healthy: bool):
        """Update health check status."""
        try:
            # Create/update health status file
            status_file = "/tmp/health_status"
            with open(status_file, "w") as f:
                f.write("healthy" if healthy else "unhealthy")

            logger.info(f"Health status updated to: {'healthy' if healthy else 'unhealthy'}")

        except Exception as e:
            logger.error(f"Failed to update health status: {e}")

    async def run(self):
        """Main run loop with graceful shutdown handling."""
        self.setup_signal_handlers()

        try:
            await self.start_services()

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self.shutdown_services()


async def main():
    """Main entry point for graceful shutdown handler."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    handler = GracefulShutdownHandler()
    await handler.run()


if __name__ == "__main__":
    asyncio.run(main())
