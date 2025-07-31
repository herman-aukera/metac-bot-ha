# Multi-stage Docker build for tournament optimization system
# Stage 1: Base Python environment with security scanning
FROM python:3.11-slim as base

# Set build arguments for security scanning
ARG BUILDKIT_INLINE_CACHE=1
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add labels for container metadata
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.title="Tournament Optimization System" \
      org.opencontainers.image.description="Production-grade AI forecasting platform" \
      org.opencontainers.image.vendor="Tournament Optimization Team"

# Install security updates and required system packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    ca-certificates \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/app --shell=/bin/bash appuser

# Set working directory
WORKDIR /app

# Stage 2: Dependencies installation with security scanning
FROM base as dependencies

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install Poetry with specific version for reproducibility
RUN pip install --no-cache-dir --upgrade pip==23.3.1 && \
    pip install --no-cache-dir poetry==1.7.1

# Configure Poetry for production
RUN poetry config virtualenvs.create false && \
    poetry config cache-dir /tmp/poetry-cache

# Install dependencies with security checks
RUN poetry install --only=main --no-dev --no-interaction --no-ansi

# Clean up poetry cache
RUN rm -rf /tmp/poetry-cache

# Stage 3: Security scanning stage
FROM dependencies as security-scan

# Install security scanning tools
RUN pip install --no-cache-dir \
    safety==2.3.5 \
    bandit==1.7.5 \
    semgrep==1.45.0

# Copy source code for scanning
COPY src/ ./src/
COPY main.py main_agent.py ./

# Run comprehensive security scanning
RUN safety check --json --output /tmp/safety-report.json || true && \
    bandit -r src/ -f json -o /tmp/bandit-report.json || true && \
    semgrep --config=auto src/ --json --output=/tmp/semgrep-report.json || true

# Stage 4: Development environment (for testing)
FROM dependencies as development

# Install development dependencies
RUN poetry install --with=dev,test --no-interaction --no-ansi

# Copy source code
COPY . .

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Development command
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Stage 5: Production build with optimizations
FROM dependencies as production

# Copy only necessary files with proper permissions
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser main.py main_agent.py ./
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser data/ ./data/

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appuser /app/logs /app/tmp

# Switch to non-root user
USER appuser

# Set environment variables for production
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Health check with improved reliability
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose application port
EXPOSE 8000

# Default production command
CMD ["python", "main_agent.py"]

# Stage 6: Production with comprehensive monitoring
FROM production as production-monitored

# Switch to root temporarily for package installation
USER root

# Install monitoring and observability tools
RUN pip install --no-cache-dir \
    prometheus-client==0.19.0 \
    structlog==23.2.0 \
    opentelemetry-api==1.21.0 \
    opentelemetry-sdk==1.21.0 \
    opentelemetry-instrumentation==0.42b0

# Switch back to non-root user
USER appuser

# Add monitoring endpoint
EXPOSE 9090

# Enhanced health check for monitored version
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health && curl -f http://localhost:9090/metrics || exit 1

# Command with monitoring enabled
CMD ["python", "-c", "import subprocess; import sys; subprocess.Popen(['python', '-m', 'src.infrastructure.monitoring.metrics_collector']); subprocess.call(['python', 'main_agent.py'])"]

# Stage 7: Blue-Green deployment ready
FROM production-monitored as blue-green

# Add deployment metadata
ENV DEPLOYMENT_COLOR="" \
    DEPLOYMENT_VERSION="" \
    DEPLOYMENT_TIMESTAMP=""

# Add graceful shutdown handling
STOPSIGNAL SIGTERM

# Enhanced command with graceful shutdown
CMD ["python", "-m", "src.infrastructure.deployment.graceful_shutdown_handler"]
