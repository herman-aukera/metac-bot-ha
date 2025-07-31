#!/bin/bash

# Tournament Optimization System Deployment Script
# Supports blue-green deployment with rollback capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_CONFIG="${PROJECT_ROOT}/configs/deployment"

# Default values
ENVIRONMENT="${ENVIRONMENT:-staging}"
DEPLOYMENT_STRATEGY="${DEPLOYMENT_STRATEGY:-blue-green}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
NAMESPACE="${NAMESPACE:-tournament-optimization}"
DRY_RUN="${DRY_RUN:-false}"
ROLLBACK="${ROLLBACK:-false}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Tournament Optimization System with blue-green deployment strategy

OPTIONS:
    -e, --environment ENVIRONMENT    Target environment (dev, staging, production)
    -s, --strategy STRATEGY         Deployment strategy (blue-green, rolling)
    -t, --tag TAG                   Docker image tag to deploy
    -n, --namespace NAMESPACE       Kubernetes namespace
    -d, --dry-run                   Perform dry run without actual deployment
    -r, --rollback                  Rollback to previous version
    -h, --help                      Show this help message

EXAMPLES:
    $0 -e staging -t v1.2.3
    $0 -e production -s blue-green -t v1.2.3
    $0 -r -e production  # Rollback production

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--strategy)
                DEPLOYMENT_STRATEGY="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -r|--rollback)
                ROLLBACK="true"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        dev|staging|production)
            log_info "Deploying to environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_error "Valid environments: dev, staging, production"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check required tools
    local required_tools=("kubectl" "helm" "docker" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done

    # Check kubectl context
    local current_context
    current_context=$(kubectl config current-context)
    log_info "Current kubectl context: $current_context"

    # Verify cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Load configuration
load_config() {
    local config_file="${DEPLOYMENT_CONFIG}/${ENVIRONMENT}.yaml"

    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        exit 1
    fi

    log_info "Loading configuration from: $config_file"

    # Export configuration variables
    export CONFIG_FILE="$config_file"
    export REGISTRY_URL=$(yq eval '.registry.url' "$config_file")
    export IMAGE_NAME=$(yq eval '.image.name' "$config_file")
    export REPLICAS=$(yq eval '.deployment.replicas' "$config_file")
    export RESOURCES_CPU_REQUEST=$(yq eval '.resources.requests.cpu' "$config_file")
    export RESOURCES_MEMORY_REQUEST=$(yq eval '.resources.requests.memory' "$config_file")
    export RESOURCES_CPU_LIMIT=$(yq eval '.resources.limits.cpu' "$config_file")
    export RESOURCES_MEMORY_LIMIT=$(yq eval '.resources.limits.memory' "$config_file")
}

# Get current deployment info
get_current_deployment() {
    local current_deployment
    current_deployment=$(kubectl get deployment -n "$NAMESPACE" -l app=tournament-optimization -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [[ -n "$current_deployment" ]]; then
        log_info "Current deployment: $current_deployment"
        echo "$current_deployment"
    else
        log_info "No current deployment found"
        echo ""
    fi
}

# Determine next deployment color
get_next_color() {
    local current_deployment="$1"

    if [[ "$current_deployment" == *"-blue" ]]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Health check function
health_check() {
    local deployment_name="$1"
    local timeout="$2"

    log_info "Performing health check for deployment: $deployment_name"

    # Wait for deployment to be ready
    if ! kubectl wait --for=condition=available --timeout="${timeout}s" deployment/"$deployment_name" -n "$NAMESPACE"; then
        log_error "Deployment $deployment_name failed to become ready within ${timeout}s"
        return 1
    fi

    # Get service endpoint
    local service_name="${deployment_name%-*}"  # Remove color suffix
    local service_port
    service_port=$(kubectl get service "$service_name" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "8000")

    # Perform application health check
    log_info "Checking application health endpoint..."
    local health_url="http://${service_name}.${NAMESPACE}.svc.cluster.local:${service_port}/health"

    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if kubectl run health-check-pod --rm -i --restart=Never --image=curlimages/curl -- \
           curl -f -s "$health_url" &> /dev/null; then
            log_success "Health check passed"
            return 0
        fi

        log_info "Health check attempt $attempt/$max_attempts failed, retrying in 10s..."
        sleep 10
        ((attempt++))
    done

    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Blue-green deployment
deploy_blue_green() {
    local current_deployment
    current_deployment=$(get_current_deployment)

    local next_color
    next_color=$(get_next_color "$current_deployment")

    local new_deployment="tournament-optimization-$next_color"

    log_info "Starting blue-green deployment"
    log_info "Current deployment: ${current_deployment:-none}"
    log_info "New deployment: $new_deployment"

    # Create new deployment
    log_info "Creating new deployment: $new_deployment"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create deployment: $new_deployment"
        log_info "[DRY RUN] Would use image: ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"
        return 0
    fi

    # Apply Kubernetes manifests
    envsubst < "${PROJECT_ROOT}/k8s/deployment-template.yaml" | \
        sed "s/{{DEPLOYMENT_NAME}}/$new_deployment/g" | \
        sed "s/{{IMAGE_TAG}}/$IMAGE_TAG/g" | \
        kubectl apply -f -

    # Wait for new deployment to be ready
    if ! health_check "$new_deployment" "$HEALTH_CHECK_TIMEOUT"; then
        log_error "New deployment failed health check, rolling back..."
        kubectl delete deployment "$new_deployment" -n "$NAMESPACE" || true
        return 1
    fi

    # Switch traffic to new deployment
    log_info "Switching traffic to new deployment"
    kubectl patch service tournament-optimization -n "$NAMESPACE" \
        -p '{"spec":{"selector":{"version":"'$next_color'"}}}'

    # Wait a bit for traffic to switch
    sleep 30

    # Verify new deployment is receiving traffic
    if ! health_check "$new_deployment" 60; then
        log_error "New deployment not receiving traffic properly, rolling back..."
        rollback_deployment "$current_deployment"
        return 1
    fi

    # Clean up old deployment
    if [[ -n "$current_deployment" ]]; then
        log_info "Cleaning up old deployment: $current_deployment"
        kubectl delete deployment "$current_deployment" -n "$NAMESPACE" || true
    fi

    log_success "Blue-green deployment completed successfully"
    log_success "Active deployment: $new_deployment"
}

# Rollback deployment
rollback_deployment() {
    local target_deployment="$1"

    if [[ -z "$target_deployment" ]]; then
        log_error "No target deployment specified for rollback"
        return 1
    fi

    log_info "Rolling back to deployment: $target_deployment"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would rollback to: $target_deployment"
        return 0
    fi

    # Get the color from deployment name
    local color
    if [[ "$target_deployment" == *"-blue" ]]; then
        color="blue"
    elif [[ "$target_deployment" == *"-green" ]]; then
        color="green"
    else
        log_error "Cannot determine color from deployment name: $target_deployment"
        return 1
    fi

    # Switch traffic back
    kubectl patch service tournament-optimization -n "$NAMESPACE" \
        -p '{"spec":{"selector":{"version":"'$color'"}}}'

    log_success "Rollback completed"
}

# Rolling deployment
deploy_rolling() {
    log_info "Starting rolling deployment"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform rolling update with image: ${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}"
        return 0
    fi

    # Update deployment image
    kubectl set image deployment/tournament-optimization \
        tournament-optimization="${REGISTRY_URL}/${IMAGE_NAME}:${IMAGE_TAG}" \
        -n "$NAMESPACE"

    # Wait for rollout to complete
    if ! kubectl rollout status deployment/tournament-optimization -n "$NAMESPACE" --timeout="${HEALTH_CHECK_TIMEOUT}s"; then
        log_error "Rolling deployment failed"
        log_info "Rolling back to previous version"
        kubectl rollout undo deployment/tournament-optimization -n "$NAMESPACE"
        return 1
    fi

    log_success "Rolling deployment completed successfully"
}

# Main deployment function
deploy() {
    log_info "Starting deployment with strategy: $DEPLOYMENT_STRATEGY"

    case $DEPLOYMENT_STRATEGY in
        blue-green)
            deploy_blue_green
            ;;
        rolling)
            deploy_rolling
            ;;
        *)
            log_error "Unknown deployment strategy: $DEPLOYMENT_STRATEGY"
            exit 1
            ;;
    esac
}

# Rollback function
perform_rollback() {
    log_info "Starting rollback process"

    local current_deployment
    current_deployment=$(get_current_deployment)

    if [[ -z "$current_deployment" ]]; then
        log_error "No current deployment found to rollback from"
        exit 1
    fi

    # For blue-green, switch to the other color
    if [[ "$DEPLOYMENT_STRATEGY" == "blue-green" ]]; then
        local other_color
        if [[ "$current_deployment" == *"-blue" ]]; then
            other_color="green"
        else
            other_color="blue"
        fi

        local other_deployment="tournament-optimization-$other_color"

        # Check if other deployment exists
        if kubectl get deployment "$other_deployment" -n "$NAMESPACE" &> /dev/null; then
            rollback_deployment "$other_deployment"
        else
            log_error "No previous deployment found for rollback"
            exit 1
        fi
    else
        # For rolling deployment, use kubectl rollback
        kubectl rollout undo deployment/tournament-optimization -n "$NAMESPACE"
        kubectl rollout status deployment/tournament-optimization -n "$NAMESPACE"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary resources..."
    kubectl delete pod health-check-pod -n "$NAMESPACE" 2>/dev/null || true
}

# Signal handlers
trap cleanup EXIT
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Main execution
main() {
    parse_args "$@"
    validate_environment
    check_prerequisites
    load_config

    if [[ "$ROLLBACK" == "true" ]]; then
        perform_rollback
    else
        deploy
    fi

    log_success "Deployment process completed"
}

# Run main function with all arguments
main "$@"
