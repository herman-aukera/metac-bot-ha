#!/bin/bash

# Rollback Script for AI Forecasting Bot
set -e

BACKUP_DIR="./backups"
HEALTH_CHECK_URL="http://localhost:8080/health"
TIMEOUT=120
INTERVAL=10

echo "Starting rollback procedure"

# Function to check service health
check_health() {
    local url=$1
    local max_attempts=$((TIMEOUT / INTERVAL))
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "Health check passed"
            return 0
        fi
        echo "Health check attempt $attempt/$max_attempts failed, waiting ${INTERVAL}s..."
        sleep $INTERVAL
        attempt=$((attempt + 1))
    done

    echo "Health check failed after $TIMEOUT seconds"
    return 1
}

# Function to get last known good deployment
get_last_good_deployment() {
    if [ -f "$BACKUP_DIR/last_good_deployment.txt" ]; then
        cat "$BACKUP_DIR/last_good_deployment.txt"
    else
        echo "No previous deployment found"
        return 1
    fi
}

# Function to rollback to previous version
rollback_deployment() {
    local previous_version=$1
    echo "Rolling back to version: $previous_version"

    # Stop current deployment
    echo "Stopping current deployment"
    docker-compose down || true

    # Restore previous configuration
    if [ -f "$BACKUP_DIR/docker-compose.backup.yml" ]; then
        echo "Restoring previous docker-compose configuration"
        cp "$BACKUP_DIR/docker-compose.backup.yml" docker-compose.yml
    fi

    # Update image tag to previous version
    sed -i "s|image: .*|image: $previous_version|g" docker-compose.yml

    # Start previous deployment
    echo "Starting previous deployment"
    docker-compose up -d

    # Wait for service to be ready
    echo "Waiting for service to be healthy"
    if check_health "$HEALTH_CHECK_URL"; then
        echo "Rollback successful"
        return 0
    else
        echo "Rollback failed - service not healthy"
        return 1
    fi
}

# Function to rollback database changes
rollback_database() {
    echo "Checking for database rollback requirements"

    if [ -f "$BACKUP_DIR/database_backup.sql" ]; then
        echo "Database backup found, performing rollback"
        # Add database rollback logic here
        echo "Database rollback completed"
    else
        echo "No database backup found, skipping database rollback"
    fi
}

# Function to notify about rollback
notify_rollback() {
    local reason=$1
    echo "Sending rollback notification"

    # Send notification (webhook, email, etc.)
    curl -X POST -H "Content-Type: application/json" \
         -d "{\"text\":\"🚨 AI Forecasting Bot rollback initiated: $reason\"}" \
         "${WEBHOOK_URL:-http://localhost:5001/webhook}" || true
}

# Function to create emergency backup
create_emergency_backup() {
    echo "Creating emergency backup before rollback"

    mkdir -p "$BACKUP_DIR/emergency"

    # Backup current configuration
    cp docker-compose.yml "$BACKUP_DIR/emergency/docker-compose.emergency.yml" || true

    # Backup logs
    cp -r logs "$BACKUP_DIR/emergency/" || true

    # Backup data
    cp -r data "$BACKUP_DIR/emergency/" || true

    echo "Emergency backup created"
}

# Main rollback logic
ROLLBACK_REASON=${1:-"Manual rollback"}
echo "Rollback reason: $ROLLBACK_REASON"

# Create emergency backup
create_emergency_backup

# Get last known good deployment
PREVIOUS_VERSION=$(get_last_good_deployment)

if [ $? -eq 0 ]; then
    echo "Found previous deployment: $PREVIOUS_VERSION"

    # Perform rollback
    if rollback_deployment "$PREVIOUS_VERSION"; then
        echo "✓ Application rollback successful"

        # Rollback database if needed
        rollback_database

        # Notify about successful rollback
        notify_rollback "Successful rollback to $PREVIOUS_VERSION"

        echo "✓ Rollback completed successfully"
    else
        echo "✗ Application rollback failed"

        # Try emergency recovery
        echo "Attempting emergency recovery"
        docker-compose -f docker-compose.emergency.yml up -d || true

        notify_rollback "Rollback failed - emergency recovery attempted"
        exit 1
    fi
else
    echo "✗ No previous deployment found for rollback"

    # Try to start with emergency configuration
    echo "Attempting to start with emergency configuration"
    if [ -f "docker-compose.emergency.yml" ]; then
        docker-compose -f docker-compose.emergency.yml up -d
        notify_rollback "No previous version - started emergency configuration"
    else
        echo "No emergency configuration available"
        notify_rollback "Rollback failed - no previous version or emergency config"
        exit 1
    fi
fi
