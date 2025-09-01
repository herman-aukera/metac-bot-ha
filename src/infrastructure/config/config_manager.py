"""
Advanced configuration management system with hot-reloading and validation.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import yaml

# File watching imports - optional dependency
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None
    FileSystemEventHandler = None
    WATCHDOG_AVAILABLE = False
import structlog

from .settings import Config, Settings

logger = structlog.get_logger(__name__)


class ConfigChangeType(Enum):
    """Types of configuration changes."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class ConfigChangeEvent:
    """Configuration change event."""

    change_type: ConfigChangeType
    file_path: Path
    timestamp: datetime
    old_config: Optional[Dict[str, Any]] = None
    new_config: Optional[Dict[str, Any]] = None


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConfigFileHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """File system event handler for configuration changes."""

    def __init__(self, config_manager: "ConfigManager"):
        self.config_manager = config_manager
        self.logger = structlog.get_logger(__name__)

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and self._is_config_file(event.src_path):
            self.logger.info("Configuration file modified", path=event.src_path)
            try:
                # Check if there's a running event loop
                loop = asyncio.get_running_loop()
                if loop and not loop.is_closed():
                    loop.create_task(
                        self.config_manager._handle_config_change(
                            ConfigChangeType.MODIFIED, Path(event.src_path)
                        )
                    )
                else:
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                self.logger.warning("No running event loop, deferring config change handling")
                if not hasattr(self.config_manager, '_deferred_changes'):
                    self.config_manager._deferred_changes = []
                self.config_manager._deferred_changes.append(
                    (ConfigChangeType.MODIFIED, Path(event.src_path))
                )

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and self._is_config_file(event.src_path):
            self.logger.info("Configuration file created", path=event.src_path)
            try:
                # Try to create task if event loop is running
                loop = asyncio.get_running_loop()
                if loop and not loop.is_closed():
                    loop.create_task(
                        self.config_manager._handle_config_change(
                            ConfigChangeType.CREATED, Path(event.src_path)
                        )
                    )
                else:
                    raise RuntimeError("Event loop is closed")
            except RuntimeError:
                # No running event loop, defer to main thread
                self.logger.warning("No running event loop, deferring config change handling")
                # Store change for later processing
                if not hasattr(self.config_manager, '_deferred_changes'):
                    self.config_manager._deferred_changes = []
                self.config_manager._deferred_changes.append(
                    (ConfigChangeType.CREATED, Path(event.src_path))
                )

    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory and self._is_config_file(event.src_path):
            self.logger.info("Configuration file deleted", path=event.src_path)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.config_manager._handle_config_change(
                        ConfigChangeType.DELETED, Path(event.src_path)
                    )
                )
            except RuntimeError:
                self.logger.warning("No running event loop, deferring config change handling")
                if not hasattr(self.config_manager, '_deferred_changes'):
                    self.config_manager._deferred_changes = []
                self.config_manager._deferred_changes.append(
                    (ConfigChangeType.DELETED, Path(event.src_path))
                )

    def on_moved(self, event):
        """Handle file move events."""
        if not event.is_directory and (
            self._is_config_file(event.src_path)
            or self._is_config_file(event.dest_path)
        ):
            self.logger.info(
                "Configuration file moved", src=event.src_path, dest=event.dest_path
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.config_manager._handle_config_change(
                        ConfigChangeType.MOVED, Path(event.dest_path)
                    )
                )
            except RuntimeError:
                self.logger.warning("No running event loop, deferring config change handling")
                if not hasattr(self.config_manager, '_deferred_changes'):
                    self.config_manager._deferred_changes = []
                self.config_manager._deferred_changes.append(
                    (ConfigChangeType.MOVED, Path(event.dest_path))
                )

    def _is_config_file(self, file_path: str) -> bool:
        """Check if file is a configuration file."""
        path = Path(file_path)
        return path.suffix.lower() in {".yaml", ".yml", ".json", ".toml"}


class ConfigManager:
    """
    Advanced configuration management system with hot-reloading, validation, and change tracking.
    """

    def __init__(
        self,
        config_paths: Optional[List[Path]] = None,
        watch_directories: Optional[List[Path]] = None,
        enable_hot_reload: bool = True,
        validation_enabled: bool = True,
    ):
        """
        Initialize configuration manager.

        Args:
            config_paths: List of configuration file paths to load
            watch_directories: List of directories to watch for changes
            enable_hot_reload: Whether to enable hot-reloading
            validation_enabled: Whether to enable configuration validation
        """
        self.config_paths = config_paths or []
        self.watch_directories = watch_directories or []
        self.enable_hot_reload = enable_hot_reload
        self.validation_enabled = validation_enabled

        # Current configuration state
        self.current_settings: Optional[Settings] = None
        self.config_history: List[ConfigChangeEvent] = []
        self.last_reload_time: Optional[datetime] = None

        # Change listeners
        self.change_listeners: List[Callable[[ConfigChangeEvent], None]] = []
        self.validation_listeners: List[Callable[[ConfigValidationResult], None]] = []

    # File watching
        self.observer: Optional[Observer] = None
        self.file_handler: Optional[ConfigFileHandler] = None
    self._polling_task: Optional[asyncio.Task] = None

        # Configuration cache
        self.config_cache: Dict[str, Any] = {}
        self.file_timestamps: Dict[Path, datetime] = {}

        # Validation rules
        self.validation_rules: Dict[str, Callable[[Any], bool]] = {}
        self._setup_default_validation_rules()

    def _setup_default_validation_rules(self) -> None:
        """Setup default configuration validation rules."""
        self.validation_rules.update(
            {
                "llm.temperature": lambda x: 0.0 <= x <= 2.0,
                "llm.max_tokens": lambda x: x is None or (isinstance(x, int) and x > 0),
                "llm.rate_limit_rpm": lambda x: isinstance(x, int) and x > 0,
                "search.max_results": lambda x: isinstance(x, int) and 1 <= x <= 100,
                "search.timeout": lambda x: isinstance(x, (int, float)) and x > 0,
                "pipeline.max_concurrent_questions": lambda x: isinstance(x, int)
                and 1 <= x <= 50,
                "pipeline.batch_delay_seconds": lambda x: isinstance(x, (int, float))
                and x >= 0,
                "bot.min_confidence_threshold": lambda x: 0.0 <= x <= 1.0,
                "ensemble.confidence_threshold": lambda x: 0.0 <= x <= 1.0,
            }
        )

    async def initialize(self) -> Settings:
        """Initialize configuration manager and load initial configuration."""
        logger.info("Initializing configuration manager")

        try:
            # Load initial configuration
            self.current_settings = await self._load_configuration()

            # Start file watching if enabled
            if self.enable_hot_reload:
                await self._start_file_watching()

            # Record initial load
            self.last_reload_time = datetime.now(timezone.utc)

            logger.info(
                "Configuration manager initialized",
                config_paths=len(self.config_paths),
                watch_directories=len(self.watch_directories),
                hot_reload_enabled=self.enable_hot_reload,
            )

            return self.current_settings

        except Exception as e:
            logger.error("Failed to initialize configuration manager", error=str(e))
            raise

    async def _load_configuration(self) -> Settings:
        """Load configuration from all specified sources."""
        merged_config = {}

        # Load from each configuration file
        for config_path in self.config_paths:
            if config_path.exists():
                try:
                    file_config = await self._load_config_file(config_path)
                    merged_config = self._merge_configs(merged_config, file_config)

                    # Update file timestamp
                    self.file_timestamps[config_path] = datetime.fromtimestamp(
                        config_path.stat().st_mtime, tz=timezone.utc
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to load config file {config_path}", error=str(e)
                    )
                    continue

        # Create Settings instance
        if merged_config:
            # Create temporary config file for Settings.load_from_yaml
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(merged_config, f)
                temp_path = f.name

            try:
                settings = Settings.load_from_yaml(temp_path)
            finally:
                Path(temp_path).unlink(missing_ok=True)
        else:
            # Fall back to environment variables
            settings = Settings.load_from_env()

        # Validate configuration if enabled
        if self.validation_enabled:
            validation_result = await self._validate_configuration(settings)
            if not validation_result.is_valid:
                logger.error(
                    "Configuration validation failed", errors=validation_result.errors
                )
                # Notify validation listeners
                for listener in self.validation_listeners:
                    try:
                        listener(validation_result)
                    except Exception as e:
                        logger.error("Validation listener failed", error=str(e))

        return settings

    async def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from a single file."""
        try:
            with open(config_path, "r") as f:
                if config_path.suffix.lower() in {".yaml", ".yml"}:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == ".json":
                    return json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {config_path}")
                    return {}
        except Exception as e:
            logger.error(f"Failed to load config file {config_path}", error=str(e))
            return {}

    def _merge_configs(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    async def _validate_configuration(
        self, settings: Settings
    ) -> ConfigValidationResult:
        """Validate configuration against defined rules."""
        errors = []
        warnings = []

        # Convert settings to dictionary for validation
        config_dict = settings.__dict__

        # Apply validation rules
        for rule_path, validator in self.validation_rules.items():
            try:
                value = self._get_nested_value(config_dict, rule_path)
                if value is not None and not validator(value):
                    errors.append(f"Validation failed for {rule_path}: {value}")
            except KeyError:
                warnings.append(f"Configuration path not found: {rule_path}")
            except Exception as e:
                errors.append(f"Validation error for {rule_path}: {str(e)}")

        # Additional custom validations
        await self._perform_custom_validations(settings, errors, warnings)

        return ConfigValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    async def _perform_custom_validations(
        self, settings: Settings, errors: List[str], warnings: List[str]
    ) -> None:
        """Perform custom validation logic."""
        # Validate API keys are present for enabled services
        if settings.llm.provider == "openai" and not settings.llm.openai_api_key:
            errors.append("OpenAI API key is required when using OpenAI provider")

        if settings.search.provider == "serpapi" and not settings.search.serpapi_key:
            warnings.append(
                "SerpAPI key not configured, search functionality may be limited"
            )

        # Validate ensemble configuration
        if settings.ensemble.min_agents > len(settings.ensemble.agent_weights):
            errors.append("Minimum agents exceeds available agent weights")

        # Validate tournament configuration
        if settings.metaculus.tournament_id and settings.metaculus.tournament_id <= 0:
            errors.append("Invalid tournament ID")

    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested value from configuration dictionary."""
        keys = path.split(".")
        current = config

        for key in keys:
            if hasattr(current, key):
                current = getattr(current, key)
            elif isinstance(current, dict) and key in current:
                current = current[key]
            else:
                raise KeyError(f"Path not found: {path}")

        return current

    async def _start_file_watching(self) -> None:
        """Start file system watching for configuration changes."""
        if not self.watch_directories:
            return

        if WATCHDOG_AVAILABLE:
            try:
                self.observer = Observer()
                self.file_handler = ConfigFileHandler(self)

                for watch_dir in self.watch_directories:
                    if watch_dir.exists():
                        self.observer.schedule(
                            self.file_handler, str(watch_dir), recursive=True
                        )
                        logger.info(
                            f"Watching directory for config changes: {watch_dir}"
                        )

                self.observer.start()
                logger.info("File watching started")

            except Exception as e:
                logger.error("Failed to start file watching", error=str(e))
        else:
            # Fallback to polling watcher
            logger.warning(
                "Watchdog not available, starting polling-based config watcher"
            )
            self._polling_task = asyncio.create_task(self._poll_for_changes())

    async def _poll_for_changes(self) -> None:
        """Polling fallback to detect configuration changes when watchdog is unavailable."""
        try:
            # Initialize timestamps
            for p in self.config_paths:
                if p.exists():
                    self.file_timestamps[p] = datetime.fromtimestamp(
                        p.stat().st_mtime, tz=timezone.utc
                    )

            while True:
                await asyncio.sleep(0.2)

                candidate_files: Set[Path] = set(self.config_paths)
                for watch_dir in self.watch_directories:
                    if watch_dir.exists():
                        for fp in watch_dir.rglob("*"):
                            if fp.is_file() and fp.suffix.lower() in {".yaml", ".yml", ".json", ".toml"}:
                                candidate_files.add(fp)

                for fp in candidate_files:
                    try:
                        if not fp.exists():
                            if fp in self.file_timestamps:
                                await self._handle_config_change(ConfigChangeType.DELETED, fp)
                                del self.file_timestamps[fp]
                            continue

                        mtime = datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc)
                        last = self.file_timestamps.get(fp)
                        if last is None:
                            self.file_timestamps[fp] = mtime
                            await self._handle_config_change(ConfigChangeType.CREATED, fp)
                        elif mtime > last:
                            self.file_timestamps[fp] = mtime
                            await self._handle_config_change(ConfigChangeType.MODIFIED, fp)
                    except Exception as e:
                        logger.warning("Polling watcher error", path=str(fp), error=str(e))
                        continue
        except asyncio.CancelledError:
            logger.info("Polling watcher cancelled")
        except Exception as e:
            logger.error("Polling watcher crashed", error=str(e))

    async def _handle_config_change(
        self, change_type: ConfigChangeType, file_path: Path
    ) -> None:
        """Handle configuration file changes."""
        logger.info(f"Handling config change: {change_type.value}", path=str(file_path))

        try:
            # Store old configuration
            old_config = (
                self.current_settings.__dict__.copy() if self.current_settings else None
            )

            # Reload configuration
            new_settings = await self._load_configuration()

            # Create change event
            change_event = ConfigChangeEvent(
                change_type=change_type,
                file_path=file_path,
                timestamp=datetime.now(timezone.utc),
                old_config=old_config,
                new_config=new_settings.__dict__.copy(),
            )

            # Update current settings
            self.current_settings = new_settings
            self.last_reload_time = change_event.timestamp

            # Add to history
            self.config_history.append(change_event)

            # Notify listeners
            await self._notify_change_listeners(change_event)

            logger.info(
                "Configuration reloaded successfully",
                change_type=change_type.value,
                path=str(file_path),
            )

        except Exception as e:
            logger.error(
                "Failed to handle config change",
                change_type=change_type.value,
                path=str(file_path),
                error=str(e),
            )

    async def _notify_change_listeners(self, change_event: ConfigChangeEvent) -> None:
        """Notify all registered change listeners."""
        for listener in self.change_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(change_event)
                else:
                    listener(change_event)
            except Exception as e:
                logger.error("Change listener failed", error=str(e))

    def add_change_listener(
        self, listener: Callable[[ConfigChangeEvent], None]
    ) -> None:
        """Add a configuration change listener."""
        self.change_listeners.append(listener)
        logger.info("Configuration change listener added")

    def remove_change_listener(
        self, listener: Callable[[ConfigChangeEvent], None]
    ) -> None:
        """Remove a configuration change listener."""
        if listener in self.change_listeners:
            self.change_listeners.remove(listener)
            logger.info("Configuration change listener removed")

    def add_validation_listener(
        self, listener: Callable[[ConfigValidationResult], None]
    ) -> None:
        """Add a configuration validation listener."""
        self.validation_listeners.append(listener)
        logger.info("Configuration validation listener added")

    def add_validation_rule(self, path: str, validator: Callable[[Any], bool]) -> None:
        """Add a custom validation rule."""
        self.validation_rules[path] = validator
        logger.info(f"Validation rule added for path: {path}")

    async def reload_configuration(self) -> Settings:
        """Manually reload configuration."""
        logger.info("Manual configuration reload requested")

        try:
            old_config = (
                self.current_settings.__dict__.copy() if self.current_settings else None
            )
            new_settings = await self._load_configuration()

            # Create change event
            change_event = ConfigChangeEvent(
                change_type=ConfigChangeType.MODIFIED,
                file_path=Path("manual_reload"),
                timestamp=datetime.now(timezone.utc),
                old_config=old_config,
                new_config=new_settings.__dict__.copy(),
            )

            self.current_settings = new_settings
            self.last_reload_time = change_event.timestamp
            self.config_history.append(change_event)

            await self._notify_change_listeners(change_event)

            logger.info("Manual configuration reload completed")
            return new_settings

        except Exception as e:
            logger.error("Manual configuration reload failed", error=str(e))
            raise

    def get_current_settings(self) -> Optional[Settings]:
        """Get current configuration settings."""
        return self.current_settings

    def get_config_history(self) -> List[ConfigChangeEvent]:
        """Get configuration change history."""
        return self.config_history.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get configuration manager status."""
        return {
            "initialized": self.current_settings is not None,
            "hot_reload_enabled": self.enable_hot_reload,
            "validation_enabled": self.validation_enabled,
            "config_paths": [str(p) for p in self.config_paths],
            "watch_directories": [str(p) for p in self.watch_directories],
            "last_reload_time": (
                self.last_reload_time.isoformat() if self.last_reload_time else None
            ),
            "change_listeners": len(self.change_listeners),
            "validation_listeners": len(self.validation_listeners),
            "config_history_count": len(self.config_history),
            "file_watching_active": (
                (self.observer is not None and self.observer.is_alive())
                if WATCHDOG_AVAILABLE
                else (self._polling_task is not None and not self._polling_task.done())
            ),
        }

    async def shutdown(self) -> None:
        """Shutdown configuration manager."""
        logger.info("Shutting down configuration manager")

        try:
            # Stop file watching
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5.0)
                self.observer = None

            # Stop polling watcher
            if self._polling_task and not self._polling_task.done():
                try:
                    self._polling_task.cancel()
                    await self._polling_task
                except Exception:
                    pass
                finally:
                    self._polling_task = None

            # Clear listeners
            self.change_listeners.clear()
            self.validation_listeners.clear()

            # Clear cache
            self.config_cache.clear()
            self.file_timestamps.clear()

            logger.info("Configuration manager shutdown complete")

        except Exception as e:
            logger.error("Error during configuration manager shutdown", error=str(e))


# Factory function for easy instantiation
def create_config_manager(
    config_paths: Optional[List[str]] = None,
    watch_directories: Optional[List[str]] = None,
    enable_hot_reload: bool = True,
    validation_enabled: bool = True,
) -> ConfigManager:
    """Factory function to create configuration manager."""
    paths = [Path(p) for p in config_paths] if config_paths else []
    watch_dirs = [Path(p) for p in watch_directories] if watch_directories else []

    return ConfigManager(
        config_paths=paths,
        watch_directories=watch_dirs,
        enable_hot_reload=enable_hot_reload,
        validation_enabled=validation_enabled,
    )
