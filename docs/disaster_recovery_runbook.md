# Disaster Recovery Runbook
## Tournament Optimization System

### Overview

This runbook provides step-by-step procedures for disaster recovery scenarios in the Tournament Optimization System. It covers various failure scenarios and their corresponding recovery procedures.

### Emergency Contacts

- **Primary On-Call**: [Your Team's On-Call Rotation]
- **Secondary On-Call**: [Backup Contact]
- **Infrastructure Team**: [Infrastructure Team Contact]
- **Security Team**: [Security Team Contact]

### Recovery Time Objectives (RTO) and Recovery Point Objectives (RPO)

| Environment | RTO     | RPO       | Description                               |
| ----------- | ------- | --------- | ----------------------------------------- |
| Production  | 1 hour  | 5 minutes | Maximum acceptable downtime and data loss |
| Staging     | 4 hours | 1 hour    | Used for testing recovery procedures      |

## Disaster Scenarios

### 1. Complete Application Failure

**Symptoms:**
- Application health checks failing
- 5xx errors from load balancer
- No response from application endpoints

**Recovery Steps:**

1. **Immediate Assessment**
   ```bash
   # Check application health
   curl -f https://tournament-optimization.example.com/health

   # Check Kubernetes deployment status
   kubectl get deployments -n tournament-optimization-production
   kubectl get pods -n tournament-optimization-production
   ```

2. **Quick Recovery (Rollback)**
   ```bash
   # Rollback to previous version
   ./scripts/deploy.sh --environment production --rollback

   # Monitor rollback progress
   kubectl rollout status deployment/tournament-optimization -n tournament-optimization-production
   ```

3. **If Rollback Fails - Full Recovery**
   ```bash
   # Get latest backup manifest
   aws s3 ls s3://tournament-optimization-production-backups/manifests/ --recursive | tail -1

   # Restore from backup
   python scripts/backup_strategy.py restore <backup_id>

   # Verify restoration
   python scripts/deployment_validation.py --environment production --strict
   ```

### 2. Database Failure

**Symptoms:**
- Database connection errors
- Application unable to read/write data
- RDS instance showing as unavailable

**Recovery Steps:**

1. **Check Database Status**
   ```bash
   # Check RDS instance status
   aws rds describe-db-instances --db-instance-identifier tournament-optimization-production-db

   # Check database connectivity from application
   kubectl exec -it deployment/tournament-optimization -n tournament-optimization-production -- pg_isready -h $DATABASE_HOST
   ```

2. **Restore from Latest Snapshot**
   ```bash
   # List available snapshots
   aws rds describe-db-snapshots --db-instance-identifier tournament-optimization-production-db --snapshot-type manual

   # Restore from snapshot (this creates a new instance)
   aws rds restore-db-instance-from-db-snapshot \
     --db-instance-identifier tournament-optimization-production-db-restored \
     --db-snapshot-identifier <latest-snapshot-id>

   # Update application configuration to point to new database
   kubectl patch secret tournament-optimization-secrets -n tournament-optimization-production \
     --patch='{"data":{"database-url":"<new-database-url-base64>"}}'

   # Restart application
   kubectl rollout restart deployment/tournament-optimization -n tournament-optimization-production
   ```

3. **Point-in-Time Recovery (if needed)**
   ```bash
   # Restore to specific point in time
   aws rds restore-db-instance-to-point-in-time \
     --source-db-instance-identifier tournament-optimization-production-db \
     --target-db-instance-identifier tournament-optimization-production-db-pitr \
     --restore-time 2024-01-15T10:30:00.000Z
   ```

### 3. Complete Infrastructure Failure (Region-wide)

**Symptoms:**
- All AWS services in primary region unavailable
- Cannot access EKS cluster
- Load balancer not responding

**Recovery Steps:**

1. **Activate Cross-Region Recovery**
   ```bash
   # Switch to backup region
   export AWS_DEFAULT_REGION=us-east-1

   # Check backup region infrastructure
   aws eks describe-cluster --name tournament-optimization-production-backup
   ```

2. **Restore Database in Backup Region**
   ```bash
   # Find latest cross-region snapshot
   aws rds describe-db-snapshots --region us-east-1 --snapshot-type manual | grep tournament-optimization

   # Restore database in backup region
   aws rds restore-db-instance-from-db-snapshot \
     --region us-east-1 \
     --db-instance-identifier tournament-optimization-production-dr \
     --db-snapshot-identifier <cross-region-snapshot-id>
   ```

3. **Deploy Application in Backup Region**
   ```bash
   # Update kubectl context to backup region
   aws eks update-kubeconfig --region us-east-1 --name tournament-optimization-production-backup

   # Deploy application
   ./scripts/deploy.sh --environment production --region us-east-1

   # Update DNS to point to backup region
   aws route53 change-resource-record-sets --hosted-zone-id <zone-id> --change-batch file://dns-failover.json
   ```

### 4. Data Corruption

**Symptoms:**
- Application reporting data inconsistencies
- Unexpected data values in database
- User reports of missing or incorrect data

**Recovery Steps:**

1. **Immediate Isolation**
   ```bash
   # Stop application to prevent further corruption
   kubectl scale deployment tournament-optimization --replicas=0 -n tournament-optimization-production

   # Create immediate database snapshot
   aws rds create-db-snapshot \
     --db-instance-identifier tournament-optimization-production-db \
     --db-snapshot-identifier emergency-snapshot-$(date +%Y%m%d-%H%M%S)
   ```

2. **Assess Corruption Scope**
   ```bash
   # Connect to database and run integrity checks
   kubectl run -it --rm debug --image=postgres:15 --restart=Never -- psql $DATABASE_URL

   # Run data integrity queries
   # SELECT COUNT(*) FROM tournaments WHERE created_at > updated_at;
   # SELECT * FROM predictions WHERE confidence < 0 OR confidence > 1;
   ```

3. **Restore from Clean Backup**
   ```bash
   # Identify last known good backup
   python scripts/backup_strategy.py list-backups --environment production

   # Restore from backup
   python scripts/backup_strategy.py restore <backup_id>

   # Verify data integrity
   python scripts/data_integrity_check.py --environment production
   ```

### 5. Security Incident

**Symptoms:**
- Unauthorized access detected
- Suspicious API activity
- Security alerts from monitoring systems

**Recovery Steps:**

1. **Immediate Response**
   ```bash
   # Isolate the system
   kubectl scale deployment tournament-optimization --replicas=0 -n tournament-optimization-production

   # Revoke all API keys
   python scripts/revoke_all_api_keys.py --environment production

   # Enable additional logging
   kubectl patch configmap tournament-optimization-config -n tournament-optimization-production \
     --patch='{"data":{"log_level":"DEBUG","audit_logging":"true"}}'
   ```

2. **Forensic Analysis**
   ```bash
   # Export logs for analysis
   kubectl logs deployment/tournament-optimization -n tournament-optimization-production --since=24h > incident-logs.txt

   # Check access logs
   aws logs filter-log-events --log-group-name /aws/eks/tournament-optimization/application \
     --start-time $(date -d '24 hours ago' +%s)000
   ```

3. **Clean Recovery**
   ```bash
   # Restore from backup before incident
   python scripts/backup_strategy.py restore <pre-incident-backup-id>

   # Rotate all secrets
   python scripts/rotate_secrets.py --environment production

   # Deploy with enhanced security
   ./scripts/deploy.sh --environment production --security-hardened
   ```

## Recovery Validation

After any recovery procedure, perform these validation steps:

### 1. Health Checks
```bash
# Run comprehensive smoke tests
python scripts/smoke_tests.py --environment production --comprehensive

# Run deployment validation
python scripts/deployment_validation.py --environment production --strict

# Monitor for 30 minutes
python scripts/deployment_health_monitor.py --environment production --duration 1800
```

### 2. Functional Testing
```bash
# Test core functionality
curl -X POST https://tournament-optimization.example.com/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"question": "Test question", "type": "binary"}'

# Verify database connectivity
kubectl exec -it deployment/tournament-optimization -n tournament-optimization-production -- \
  python -c "import psycopg2; conn = psycopg2.connect('$DATABASE_URL'); print('DB OK')"
```

### 3. Performance Validation
```bash
# Check response times
for i in {1..10}; do
  curl -w "@curl-format.txt" -s -o /dev/null https://tournament-optimization.example.com/health
done

# Monitor resource usage
kubectl top pods -n tournament-optimization-production
```

## Post-Incident Actions

1. **Document the Incident**
   - Create incident report with timeline
   - Document root cause analysis
   - Update runbook based on lessons learned

2. **Review and Improve**
   - Schedule post-mortem meeting
   - Update monitoring and alerting
   - Improve backup and recovery procedures

3. **Test Recovery Procedures**
   - Schedule regular disaster recovery drills
   - Update and test backup restoration
   - Verify cross-region failover capabilities

## Backup and Recovery Testing Schedule

| Test Type             | Frequency | Environment | Scope                             |
| --------------------- | --------- | ----------- | --------------------------------- |
| Backup Verification   | Daily     | All         | Automated backup integrity checks |
| Database Restore      | Weekly    | Staging     | Full database restoration test    |
| Application Recovery  | Monthly   | Staging     | Complete application recovery     |
| Cross-Region Failover | Quarterly | Staging     | Full disaster recovery simulation |

## Monitoring and Alerting

### Critical Alerts
- Application health check failures
- Database connectivity issues
- High error rates (>5%)
- Resource exhaustion (CPU >80%, Memory >85%)
- Security incidents

### Alert Channels
- **Critical**: PagerDuty + Slack #alerts
- **Warning**: Slack #monitoring
- **Info**: Email to team

## Tools and Scripts

| Script                                 | Purpose                 | Usage                                                                  |
| -------------------------------------- | ----------------------- | ---------------------------------------------------------------------- |
| `scripts/deploy.sh`                    | Deployment and rollback | `./scripts/deploy.sh --environment production --rollback`              |
| `scripts/backup_strategy.py`           | Backup and restore      | `python scripts/backup_strategy.py restore <backup_id>`                |
| `scripts/smoke_tests.py`               | Health validation       | `python scripts/smoke_tests.py --environment production`               |
| `scripts/deployment_validation.py`     | Deployment validation   | `python scripts/deployment_validation.py --environment production`     |
| `scripts/deployment_health_monitor.py` | Health monitoring       | `python scripts/deployment_health_monitor.py --environment production` |

## Emergency Procedures Quick Reference

### Application Down
1. Check health endpoints
2. Rollback deployment
3. If rollback fails, restore from backup
4. Validate recovery

### Database Issues
1. Check RDS status
2. Restore from snapshot
3. Update application configuration
4. Restart application

### Security Incident
1. Isolate system
2. Revoke credentials
3. Analyze logs
4. Clean restore
5. Rotate secrets

### Complete Outage
1. Activate cross-region recovery
2. Restore database in backup region
3. Deploy application in backup region
4. Update DNS routing

---

**Remember**: In any disaster scenario, communication is key. Keep stakeholders informed of the situation and recovery progress.

**Last Updated**: January 2024
**Next Review**: April 2024
