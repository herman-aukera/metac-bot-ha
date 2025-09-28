# Network Resilience Guide for GitHub Actions

This document outlines the network resilience improvements implemented for the Metaculus forecasting bot's CI/CD pipeline.

## Overview

The GitHub Actions workflows have been enhanced with comprehensive network resilience features to handle:
- Network timeouts during dependency installation
- Poetry installation failures
- Package registry connectivity issues
- Runner environment problems
- Emergency deployment scenarios

## Key Improvements

### 1. Timeout Protection

All network-dependent operations now have explicit timeouts:
- **Poetry installation**: 5 minutes
- **Pip operations**: 60 seconds per command with 3-5 retries
- **Dependency installation**: 10-20 minutes depending on complexity
- **Test execution**: 20-45 minutes with per-test timeouts

### 2. Pip Fallback System

When Poetry installation fails, the system automatically falls back to pip:

```yaml
- name: Fallback to pip installation if poetry fails
  if: steps.install-poetry.outcome == 'failure'
  timeout-minutes: 10
  run: |
    echo "Poetry installation failed, falling back to pip..."
    python -m pip install --upgrade pip --timeout 60 --retries 3
    python -m pip install --timeout 60 --retries 3 poetry
```

### 3. Retry Logic

Network-dependent operations include retry logic:
- **Poetry lock**: 3 attempts with 30-second delays
- **Dependency installation**: 3 attempts with 30-second delays
- **Pip commands**: Built-in 3-5 retries with exponential backoff

### 4. Alternative Runner Configurations

Emergency deployment workflow supports multiple runner types:
- `ubuntu-latest` (primary)
- `ubuntu-20.04` (fallback)
- Self-hosted runners (if available)

## Updated Workflows

### 1. test_bot_resilient.yaml
- Complete rewrite with network resilience
- Timeout protection for all steps
- Pip fallback for all Poetry operations
- Emergency deployment verification

### 2. run_bot_on_tournament.yaml
- Enhanced with tournament-specific resilience
- Emergency pip fallback for critical tournament deployment
- Timeout protection for all installation steps

### 3. ci-cd.yml
- Network resilience for CI/CD pipeline
- Fallback mechanisms for build and test phases
- Timeout protection for all operations

### 4. workflows/github-actions.yml
- Basic network resilience improvements
- Retry logic for Poetry and pip operations
- Timeout protection

## Emergency Deployment

### Emergency Deployment Workflow
The `emergency-deployment.yaml` workflow provides maximum resilience:

```bash
# Trigger emergency deployment
gh workflow run emergency-deployment.yaml \
  -f emergency_mode=tournament \
  -f skip_tests=true \
  -f use_pip_only=true
```

### Manual Emergency Deployment

If all automated methods fail, use manual deployment:

```bash
# 1. Clone repository
git clone <repository-url>
cd <repository-name>

# 2. Install minimal dependencies
python -m pip install --upgrade pip --timeout 120 --retries 5
python -m pip install --timeout 120 --retries 5 \
  pydantic requests python-dotenv pyyaml \
  openai anthropic httpx aiohttp

# 3. Set environment variables
export METACULUS_TOKEN="your-token"
export OPENROUTER_API_KEY="your-key"
export APP_ENV="production"

# 4. Run the bot
python main.py
```

## Network Timeout Configuration

### Environment Variables
```yaml
env:
  PIP_TIMEOUT: 60
  PIP_RETRIES: 3
  POETRY_TIMEOUT: 600
  MAX_RETRIES: 3
  RETRY_DELAY: 30
  COMMAND_TIMEOUT: 300
```

### Timeout Hierarchy
1. **Individual commands**: 60-300 seconds
2. **Step-level timeouts**: 5-20 minutes
3. **Job-level timeouts**: 30-45 minutes
4. **Workflow-level**: Default GitHub limits

## Fallback Dependency Lists

### Minimal Tournament Dependencies
```
pydantic>=2.0.0
requests>=2.28.0
python-dotenv>=1.0.0
pyyaml>=6.0
openai>=1.0.0
anthropic>=0.3.0
httpx>=0.24.0
aiohttp>=3.8.0
```

### Testing Dependencies
```
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-timeout>=2.1.0
pytest-cov>=4.0.0
```

### Code Quality Dependencies
```
flake8>=6.0.0
black>=23.0.0
isort>=5.12.0
mypy>=1.0.0
pylint>=2.17.0
```

## Monitoring and Alerts

### Success Indicators
- ✅ All steps complete within timeout limits
- ✅ Dependencies installed successfully
- ✅ Tests pass (if not skipped)
- ✅ Artifacts uploaded successfully

### Failure Indicators
- ❌ Timeout exceeded on critical steps
- ❌ All retry attempts exhausted
- ❌ Emergency fallback required
- ❌ Manual intervention needed

### Alert Conditions
- Poetry installation fails consistently
- Network timeouts exceed 5 minutes
- Emergency deployment triggered
- Multiple runner failures

## Best Practices

### For Development
1. Test workflows locally when possible
2. Use `act` for local GitHub Actions testing
3. Monitor workflow execution times
4. Keep dependency lists minimal

### For Production
1. Monitor workflow success rates
2. Set up alerts for emergency deployments
3. Maintain backup deployment methods
4. Regular testing of fallback mechanisms

### For Tournament Deployment
1. Use emergency deployment for time-critical situations
2. Skip non-essential tests when time is limited
3. Monitor budget and resource usage
4. Have manual deployment procedures ready

## Troubleshooting

### Common Issues

#### Poetry Installation Timeout
```
Solution: Automatic pip fallback is configured
Manual: pip install poetry
```

#### Dependency Installation Failure
```
Solution: Retry logic with exponential backoff
Manual: Use emergency deployment workflow
```

#### Network Connectivity Issues
```
Solution: Multiple retry attempts with different timeouts
Manual: Use alternative runner or self-hosted runner
```

#### Runner Environment Problems
```
Solution: Matrix strategy with multiple OS versions
Manual: Use emergency deployment with minimal dependencies
```

### Emergency Procedures

#### Tournament Deadline Approaching
1. Trigger emergency deployment workflow
2. Skip non-critical tests
3. Use pip-only installation
4. Deploy with minimal dependencies

#### All Workflows Failing
1. Check GitHub Actions status
2. Try alternative runners
3. Use manual deployment procedure
4. Contact support if needed

## Verification

### Testing Network Resilience
```bash
# Test timeout handling
timeout 30s poetry install  # Should fail gracefully

# Test retry logic
# Temporarily block network access and verify retries

# Test fallback mechanisms
# Remove poetry and verify pip fallback works
```

### Deployment Verification
```bash
# Verify emergency deployment package
python verify_deployment.py

# Test minimal functionality
python -c "import src; print('Core imports working')"

# Run basic functionality test
python main.py --dry-run
```

## Maintenance

### Regular Tasks
- [ ] Monitor workflow execution times
- [ ] Update timeout values based on performance
- [ ] Test emergency deployment procedures
- [ ] Review and update dependency lists
- [ ] Verify fallback mechanisms work

### Quarterly Reviews
- [ ] Analyze workflow failure patterns
- [ ] Update network resilience strategies
- [ ] Test disaster recovery procedures
- [ ] Update documentation

This network resilience system ensures maximum uptime and deployment reliability for the Metaculus forecasting bot, especially critical during tournament periods.
