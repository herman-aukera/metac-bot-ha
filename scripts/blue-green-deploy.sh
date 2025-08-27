#!/bin/bash

# Blue-Green Deployment Script for AI Forecasting Bot
set -e

IMAGE_TAG=${1:-latest}
HEALTH_CHECK_URL="http://localhost:8080/health"
TIMEOUT=300
INTERVAL=10

echo "Starting blue-green deployment with image tag: $IMAGE_TAG"

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

# Function to get current active environment
get_active_env() {
    if docker-compose -f docker-compose.blue.yml ps -q forecasting-bot | grep -q .; then
        echo "blue"
    elif docker-compose -f docker-compose.green.yml ps -q forecasting-bot | grep -q .; then
        echo "green"
    else
        echo "none"
    fi
}

# Function to switch traffic
switch_traffic() {
    local new_env=$1
    echo "Switching traffic to $new_env environment"

    # Update nginx configuration or load balancer
    if [ "$new_env" = "blue" ]; then
        cp nginx/blue.conf nginx/nginx.conf
    else
        cp nginx/green.conf nginx/nginx.conf
    fi

    # Reload nginx
    docker-compose exec nginx nginx -s reload
}

# Main deployment logic
CURRENT_ENV=$(get_active_env)
echo "Current active environment: $CURRENT_ENV"

# Determine target environment
if [ "$CURRENT_ENV" = "blue" ] || [ "$CURRENT_ENV" = "none" ]; then
    TARGET_ENV="green"
    TARGET_COMPOSE="docker-compose.green.yml"
else
    TARGET_ENV="blue"
    TARGET_COMPOSE="docker-compose.blue.yml"
fi

echo "Deploying to $TARGET_ENV environment"

# Update image tag in target compose file
sed -i "s|image: .*|image: ghcr.io/your-org/ai-forecasting-bot:$IMAGE_TAG|g" $TARGET_COMPOSE

# Deploy to target environment
echo "Starting $TARGET_ENV environment"
docker-compose -f $TARGET_COMPOSE up -d

# Wait for service to be ready
echo "Waiting for $TARGET_ENV environment to be healthy"
if [ "$TARGET_ENV" = "blue" ]; then
    HEALTH_URL="http://localhost:8081/health"
else
    HEALTH_URL="http://localhost:8082/health"
fi

if check_health "$HEALTH_URL"; then
    echo "$TARGET_ENV environment is healthy"

    # Switch traffic to new environment
    switch_traffic "$TARGET_ENV"

    # Wait a bit for traffic to switch
    sleep 30

    # Verify main endpoint is working
    if check_health "$HEALTH_CHECK_URL"; then
        echo "Traffic switch successful"

        # Stop old environment
        if [ "$CURRENT_ENV" != "none" ]; then
            if [ "$CURRENT_ENV" = "blue" ]; then
                OLD_COMPOSE="docker-compose.blue.yml"
            else
                OLD_COMPOSE="docker-compose.green.yml"
            fi

            echo "Stopping old $CURRENT_ENV environment"
            docker-compose -f $OLD_COMPOSE down
        fi

        echo "Blue-green deployment completed successfully"
    else
        echo "Traffic switch failed, rolling back"
        switch_traffic "$CURRENT_ENV"
        docker-compose -f $TARGET_COMPOSE down
        exit 1
    fi
else
    echo "$TARGET_ENV environment failed health check, cleaning up"
    docker-compose -f $TARGET_COMPOSE down
    exit 1
fi
