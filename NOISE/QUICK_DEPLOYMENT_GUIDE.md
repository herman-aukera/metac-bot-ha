# 🚨 Quick Emergency Deployment Guide

**Tournament Deadline Approaching? Deploy in 5 minutes!**

## 🎯 Ultra-Quick Deployment (3 commands)

```bash
# 1. Clone and enter directory
git clone <repository-url> && cd metac-bot-ha

# 2. Install dependencies (pip fallback)
python3 -m pip install --upgrade pip && pip install -r requirements-emergency.txt

# 3. Configure and test
cp .env.example .env && nano .env  # Add your API keys
python3 -m src.main --tournament 32813 --max-questions 1 --dry-run
```

## 🚀 Deployment Options (Choose One)

### Option A: Automated Cloud Deployment
```bash
chmod +x scripts/manual_cloud_deployment.sh
./scripts/manual_cloud_deployment.sh
```

### Option B: Manual Local Setup
```bash
python3.11 -m venv tournament-env
source tournament-env/bin/activate
pip install -r requirements-emergency.txt
cp .env.example .env  # Edit with your API keys
```

### Option C: Emergency Pip-Only
```bash
pip install requests openai python-dotenv pydantic typer httpx asknews numpy pandas
export ASKNEWS_CLIENT_ID=your_id
export OPENROUTER_API_KEY=your_key
python3 -m src.main --tournament 32813 --max-questions 5 --dry-run
```

## ✅ Verification Commands

```bash
# Quick verification
python3 scripts/emergency_deployment_verification.py --quick

# Test single forecast
python3 -m src.main --tournament 32813 --max-questions 1 --dry-run --verbose

# Check configuration
python3 -c "from src.infrastructure.config.settings import Config; print('✅ Config OK')"
```

## 🏆 Tournament Execution

```bash
# Production run
python3 -m src.main --tournament 32813 --max-questions 100

# With monitoring
nohup python3 -m src.main --tournament 32813 --max-questions 100 > tournament.log 2>&1 &

# Monitor progress
tail -f tournament.log
```

## 🆘 Emergency Troubleshooting

**Python Version Issues:**
```bash
# Ubuntu/Debian
sudo apt install python3.11 python3.11-pip python3.11-venv

# macOS
brew install python@3.11
```

**Dependency Issues:**
```bash
# Clear cache and retry
pip cache purge
pip install --no-cache-dir --timeout 60 requests openai python-dotenv pydantic
```

**API Key Issues:**
```bash
# Test API connectivity
python3 scripts/test_all_secrets.py

# Verify environment
python3 -c "import os; print('Keys:', [k for k in os.environ if 'API' in k or 'CLIENT' in k])"
```

**Permission Issues:**
```bash
# Fix permissions
chmod +x scripts/*.py scripts/*.sh
mkdir -p logs/performance logs/reasoning data
```

## 📊 Success Criteria

- ✅ Python 3.11+ installed
- ✅ Dependencies installed successfully
- ✅ API keys configured and working
- ✅ Single question forecast test passes
- ✅ No critical errors in logs

## 🎯 Minimum Working Setup

**If everything else fails, this MUST work:**

```bash
# Absolute minimum
pip install requests openai python-dotenv
export OPENROUTER_API_KEY=your_key
python3 -c "
import sys; sys.path.append('src')
from infrastructure.config.settings import Config
print('✅ Minimum setup working')
"
```

## 📞 Emergency Checklist

- [ ] Repository cloned
- [ ] Python 3.11+ available
- [ ] Core dependencies installed
- [ ] API keys configured
- [ ] Test forecast successful
- [ ] Production run started
- [ ] Monitoring in place

**Remember: Partial functionality is better than no participation!**

---

**🏆 Tournament ID: 32813**
**⏰ Deploy fast, iterate later**
**🎯 Goal: Participate and compete**
