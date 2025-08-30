# Fork Deployment Guide

## 🔄 Fork vs Main Repository Workflows

### The Fork Limitation

GitHub has a security feature where **secrets are not passed to workflows triggered by pull requests from forks**. This is intentional to prevent malicious forks from accessing your API keys.

> "Secrets and variables allow you to manage reusable configuration data... They are not passed to workflows that are triggered by a pull request from a fork."

### How Our CI Handles This

Our CI workflow automatically detects if it's running from a fork and adapts accordingly:

#### 🍴 Fork Mode (Limited Checks)
- ✅ Python version compatibility
- ✅ Essential imports (requests, openai, pydantic, typer)
- ✅ Optional imports (reports what's available/missing)
- ⚠️ **Environment variables: SKIPPED** (secrets not available)
- ✅ Project structure validation

**Result**: Fork checks focus on code quality and structure without requiring secrets.

#### 🏠 Main Repository Mode (Full Checks)
- ✅ All fork mode checks PLUS:
- ✅ Environment variables validation (API keys required)
- ✅ Full deployment readiness assessment

## 🚀 Deployment Workflow

### For Fork Contributors

1. **Development**: Work on your fork normally
2. **Testing**: Fork CI runs limited checks (structure, imports, etc.)
3. **Pull Request**: Submit PR to main repository
4. **Full Validation**: Maintainer merges to main → full CI with secrets runs

### For Main Repository

1. **Direct Push**: Full CI runs with all secrets
2. **Deployment Ready**: All checks pass including API validation
3. **Tournament Ready**: Bot can be deployed with confidence

## 🔧 Local Development

For local development, you still need a `.env` file with real API keys:

```bash
# .env (never commit this!)
ASKNEWS_CLIENT_ID=your_actual_client_id
ASKNEWS_SECRET=your_actual_secret
OPENROUTER_API_KEY=your_actual_api_key
METACULUS_TOKEN=your_actual_token
```

## 🧪 Testing Deployment Readiness

### Fork Testing (Limited)
```bash
python scripts/ci_deployment_check.py --fork-mode
```

### Full Testing (Requires Secrets)
```bash
python scripts/ci_deployment_check.py
```

### Legacy Testing (More Comprehensive)
```bash
python scripts/deployment_readiness_check.py
```

## 📊 Understanding CI Results

### ✅ Fork Check Passed
- Code structure is valid
- Dependencies are available
- Ready for PR submission
- **Note**: Full deployment validation requires main repository

### ✅ Deployment Ready
- All checks passed including API validation
- Bot is ready for tournament deployment
- Environment properly configured

### ❌ Deployment Not Ready
- Critical issues found
- Fix required before deployment
- Check specific failure messages

## 🔐 Security Benefits

This approach provides:
- **Security**: Forks can't access your API keys
- **Validation**: Code quality checked in forks
- **Confidence**: Full validation in main repository
- **Flexibility**: Development continues smoothly despite limitations

## 🎯 Best Practices

1. **Fork Development**: Focus on code quality, structure, and logic
2. **Local Testing**: Use real API keys for full validation
3. **PR Submission**: Ensure fork checks pass before submitting
4. **Main Repository**: Full deployment validation after merge

This workflow ensures both security and functionality while accommodating GitHub's fork limitations.
