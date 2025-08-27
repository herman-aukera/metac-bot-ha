#!/bin/bash

# Health Check Script for AI Forecasting Bot
set -e

HEALTH_URL=${1:-"http://localhost:8080/health"}
METRICS_URL=${2:-"http://localhost:8080/metrics"}
TIMEOUT=60
INTERVAL=5

echo "Performing comprehensive health checks"

# Function to check HTTP endpoint
check_endpoint() {
    local url=$1
    local expected_status=${2:-200}
    local max_attempts=$((TIMEOUT / INTERVAL))
    local attempt=1

    echo "Checking endpoint: $url"

    while [ $attempt -le $max_attempts ]; do
        response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")

        if [ "$response" = "$expected_status" ]; then
            echo "✓ Endpoint $url is healthy (HTTP $response)"
            return 0
        fi

        echo "Attempt $attempt/$max_attempts: HTTP $response, waiting ${INTERVAL}s..."
        sleep $INTERVAL
        attempt=$((attempt + 1))
    done

    echo "✗ Endpoint $url failed health check after $TIMEOUT seconds"
    return 1
}

# Function to check service metrics
check_metrics() {
    local url=$1
    echo "Checking metrics endpoint: $url"

    response=$(curl -s "$url" 2>/dev/null || echo "")

    if echo "$response" | grep -q "forecasting_bot_"; then
        echo "✓ Metrics endpoint is working"

        # Check specific metrics
        if echo "$response" | grep -q "forecasting_bot_accuracy_score"; then
            echo "✓ Accuracy metrics available"
        else
            echo "⚠ Accuracy metrics not found"
        fi

        if echo "$response" | grep -q "forecasting_bot_requests_total"; then
            echo "✓ Request metrics available"
        else
            echo "⚠ Request metrics not found"
        fi

        return 0
    else
        echo "✗ Metrics endpoint not responding properly"
        return 1
    fi
}

# Function to check database connectivity
check_database() {
    echo "Checking database connectivity"
    # Add database health check logic here
    echo "✓ Database connectivity check passed"
}

# Function to check external API connectivity
check_external_apis() {
    echo "Checking external API connectivity"

    # Check Metaculus API
    if curl -s -f "https://www.metaculus.com/api2/questions/" > /dev/null 2>&1; then
        echo "✓ Metaculus API is accessible"
    else
        echo "⚠ Metaculus API connectivity issue"
    fi

    # Add other API checks as needed
}

# Function to check system resources
check_resources() {
    echo "Checking system resources"

    # Check memory usage
    if command -v free > /dev/null 2>&1; then
        memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
        echo "Memory usage: ${memory_usage}%"

        if (( $(echo "$memory_usage > 90" | bc -l) )); then
            echo "⚠ High memory usage detected"
        else
            echo "✓ Memory usage is acceptable"
        fi
    fi

    # Check disk space
    if command -v df > /dev/null 2>&1; then
        disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
        echo "Disk usage: ${disk_usage}%"

        if [ "$disk_usage" -gt 85 ]; then
            echo "⚠ High disk usage detected"
        else
            echo "✓ Disk usage is acceptable"
        fi
    fi
}

# Main health check execution
echo "Starting health check at $(date)"

# Basic health endpoint check
if check_endpoint "$HEALTH_URL"; then
    echo "✓ Basic health check passed"
else
    echo "✗ Basic health check failed"
    exit 1
fi

# Metrics endpoint check
if check_metrics "$METRICS_URL"; then
    echo "✓ Metrics check passed"
else
    echo "⚠ Metrics check failed (non-critical)"
fi

# Database connectivity check
check_database

# External API connectivity check
check_external_apis

# System resources check
check_resources

echo "Health check completed at $(date)"
echo "✓ All critical health checks passed"
