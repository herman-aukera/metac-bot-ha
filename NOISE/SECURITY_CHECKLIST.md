# Security Checklist for API Key Management

## Before Each Development Session

- [ ] Verify `.env` file is in `.gitignore`
- [ ] Check that no hardcoded secrets exist in code
- [ ] Run security audit: `grep -r "sk-" . --exclude-dir=.git --exclude="*.env*"`
- [ ] Verify all API keys use environment variables

## Before Each Commit

- [ ] Run: `git diff --cached | grep -E "(sk-|token|secret|key)" | grep -v "env"`
- [ ] Ensure no secrets in commit
- [ ] Double-check test files don't contain real credentials

## API Key Rotation Schedule

- [ ] OpenRouter: Rotate monthly or after any exposure
- [ ] Metaculus: Rotate quarterly or after any exposure
- [ ] ASKnews: Rotate quarterly or after any exposure

## Emergency Response (If Key Exposed)

1. **Immediate**: Disable the exposed key in the provider dashboard
2. **Within 1 hour**: Generate new key and update `.env`
3. **Within 24 hours**: Update all deployment environments
4. **Within 48 hours**: Review git history for other exposures

## Monitoring

- Set up GitHub secret scanning alerts
- Monitor provider dashboards for unusual usage
- Review access logs monthly

## Current Status

✅ All hardcoded secrets removed (August 2025)
✅ Environment variable pattern implemented
✅ Security documentation created
⚠️ Need to rotate Metaculus and ASKnews credentials
