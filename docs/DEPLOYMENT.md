# Production Deployment Guide

This guide covers the production deployment and monitoring setup for the AI Forecasting Bot.

## Overview

The deployment system provides:
- Containerized deployment with Docker
- Blue-green deployment strategy
- Comprehensive monitoring with Prometheus and Grafana
- Automated health checks and rollback capabilities
- CI/CD pipeline integration

## Prerequisites

- Docker and Docker Compose
- GitHub Actions (for CI/CD)
- Prometheus and Grafana (included in docker-compose)
- Nginx (for load balancing)

## Quick Start

### 1. Environment Setup

Create environment files:

```bash
# Production environment
cp .env.template .env
# Edit .env with production values

# Staging environment
cp .env.template .env.staging
# Edit .env.staging with staging values
```

### 2. Build and Deploy

```bash
# Build the Docker image
docker build -t ai-forecasting-bot:latest .

# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Deploy to production (blue-green)
./scripts/blue-green-deploy.sh latest
```

### 3. Monitor Deployment

```bash
# Check health
./scripts/health-check.sh

# View logs
docker-compose logs -f forecasting-bot

# Access monitoring
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

## Deployment Strategies

### Blue-Green Deployment

The production environment uses blue-green deployment for zero-downtime updates:

1. **Blue Environment**: Currently active production environment
2. **Green Environment**: New version being deployed
3. **Traffic Switch**: Nginx routes traffic between environments
4. **Rollback**: Automatic rollback on health check failure

```bash
# Manual blue-green deployment
./scripts/blue-green-deploy.sh <image-tag>

# Check deployment status
curl http://localhost:8080/health
```

### Standard Deployment

For staging and development environments:

```bash
# Stop current deployment
docker-compose down

# Deploy new version
docker-compose up -d

# Verify deployment
./scripts/health-check.sh
```

## Monitoring and Alerting

### Metrics

The system exposes metrics at `/metrics` endpoint:

- **Request Metrics**: Request count, duration, error rate
- **Forecasting Metrics**: Accuracy, Brier score, calibration error
- **Tournament Metrics**: Ranking, missed deadlines
- **System Metrics**: Memory usage, CPU usage
- **Agent Metrics**: Individual agent performance

### Dashboards

Grafana dashboards provide visualization for:

- System health overview
- Prediction accuracy trends
- Tournament performance
- Resource utilization
- Error rates and alerts

### Alerts

Configured alerts include:

- **Critical**: Service down, missed deadlines
- **Warning**: High error rate, low accuracy, resource usage
- **Info**: Calibration drift, performance degradation

## Health Checks

### Automated Health Checks

The system performs automated health checks:

```bash
# Basic health check
curl http://localhost:8080/health

# Comprehensive health check
./scripts/health-check.sh

# Health check with custom URL
./scripts/health-check.sh http://staging:8080/health
```

### Health Check Endpoints

- `/health`: Basic service health
- `/metrics`: Prometheus metrics
- `/ready`: Readiness probe (Kubernetes)
- `/live`: Liveness probe (Kubernetes)

## Rollback Procedures

### Automatic Rollback

Automatic rollback triggers on:
- Health check failures during deployment
- High error rates post-deployment
- Service unavailability

### Manual Rollback

```bash
# Emergency rollback
./scripts/rollback.sh "Manual rollback reason"

# Rollback to specific version
./scripts/rollback.sh "Rollback to v1.2.3"
```

### Rollback Verification

After rollback:
1. Verify service health
2. Check metrics and logs
3. Confirm tournament functionality
4. Update monitoring dashboards

## CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline includes:

1. **Test Stage**: Unit tests, integration tests, security scans
2. **Build Stage**: Docker image build and push
3. **Deploy Stage**: Automated deployment to staging/production
4. **Verify Stage**: Health checks and smoke tests

### Pipeline Configuration

```yaml
# .github/workflows/ci-cd.yml
- Automated testing on PR
- Security scanning
- Docker image building
- Staging deployment
- Production deployment (on main branch)
- Rollback on failure
```

### Manual Pipeline Triggers

```bash
# Trigger deployment via GitHub CLI
gh workflow run ci-cd.yml -f environment=production -f image_tag=v1.2.3

# Check workflow status
gh run list --workflow=ci-cd.yml
```

## Configuration Management

### Environment-Specific Configs

- `config/config.dev.yaml`: Development configuration
- `config/config.test.yaml`: Testing configuration
- `config/config.prod.yaml`: Production configuration
- `config/config.production.yaml`: Enhanced production configuration

### Configuration Updates

```bash
# Update production config
kubectl create configmap forecasting-config --from-file=config/config.production.yaml

# Restart deployment to pick up changes
kubectl rollout restart deployment/forecasting-bot
```

## Backup and Recovery

### Automated Backups

Backups are created automatically:
- Before each deployment
- Daily configuration backups
- Weekly data backups

### Manual Backup

```bash
# Create manual backup
./scripts/backup.sh manual-backup-$(date +%Y%m%d)

# List available backups
ls -la backups/

# Restore from backup
./scripts/restore.sh backups/backup_20240101_120000
```

## Troubleshooting

### Common Issues

1. **Deployment Failures**
   ```bash
   # Check deployment logs
   docker-compose logs forecasting-bot

   # Verify configuration
   docker-compose config

   # Check resource usage
   docker stats
   ```

2. **Health Check Failures**
   ```bash
   # Debug health endpoint
   curl -v http://localhost:8080/health

   # Check application logs
   tail -f logs/app.log

   # Verify dependencies
   docker-compose ps
   ```

3. **Performance Issues**
   ```bash
   # Check metrics
   curl http://localhost:8080/metrics

   # Monitor resource usage
   docker stats forecasting-bot

   # Review performance logs
   tail -f logs/performance/performance.log
   ```

### Log Analysis

```bash
# Application logs
tail -f logs/app.log

# Reasoning logs
ls logs/reasoning/

# Performance logs
tail -f logs/performance/performance.log

# System logs
journalctl -u docker -f
```

## Security Considerations

### Container Security

- Non-root user execution
- Minimal base image
- Security scanning in CI/CD
- Regular image updates

### Network Security

- Internal network isolation
- TLS encryption for external APIs
- Rate limiting and DDoS protection
- API key rotation

### Data Security

- Encrypted environment variables
- Secure backup storage
- Access logging and monitoring
- Regular security audits

## Performance Optimization

### Resource Allocation

```yaml
# docker-compose.yml
services:
  forecasting-bot:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### Scaling Configuration

```bash
# Scale horizontally
docker-compose up -d --scale forecasting-bot=3

# Monitor scaling
docker-compose ps
```

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**:
   - Review monitoring dashboards
   - Check backup integrity
   - Update dependencies

2. **Monthly**:
   - Security updates
   - Performance optimization
   - Configuration review

3. **Quarterly**:
   - Disaster recovery testing
   - Capacity planning
   - Architecture review

### Maintenance Windows

Schedule maintenance during low tournament activity:
- Coordinate with tournament calendar
- Notify stakeholders
- Prepare rollback procedures
- Monitor post-maintenance performance

## Support and Escalation

### Monitoring Contacts

- **Level 1**: Automated alerts and dashboards
- **Level 2**: On-call engineer notification
- **Level 3**: Development team escalation

### Emergency Procedures

1. **Service Down**: Execute emergency rollback
2. **Data Loss**: Restore from backup
3. **Security Incident**: Isolate and investigate
4. **Performance Degradation**: Scale resources

For additional support, refer to the troubleshooting guide or contact the development team.
