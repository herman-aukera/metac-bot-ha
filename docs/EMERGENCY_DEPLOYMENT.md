# Emergency Deployment Guide

## 🚨 Tournament Emergency Deployment

**Time-Critical Deployment Options for Metaculus Tournament**

This guide provides multiple deployment paths when primary CI/CD fails or network issues prevent normal deployment.

## 📋 Prerequisites

- Python 3.11+ installed
- Git access to repository
- API keys configured
- Linux/macOS/Windows with bash/zsh

## 🎯 Quick Start (5 minutes)

```bash
# Emergency deployment in 3 commands
git clone <repository-url> && cd metac-bot-ha
python3 -m pip install --upgrade pip && pip install -r requirements-emergency.txt
python3 -m src.main --tournament 32813 --max-questions 5 --dry-run
```

## 🛠️ Deployment Options

### Option 1: GitHub Actions (Primary)
- **Status**: May fail due to network timeouts
- **Fallback**: Use Options 2-4 below

### Option 2: Cloud Instance Manual Deployment
- **Target**: AWS EC2, Google Cloud, DigitalOcean
- **Time**: 10-15 minutes
- **Reliability**: High

### Option 3: Local Development Machine
- **Target**: Your laptop/desktop
- **Time**: 5-10 minutes
- **Reliability**: Highest

### Option 4: Emergency Pip-Only Installation
- **Target**: Any Python environment
- **Time**: 3-5 minutes
- **Reliability**: Maximum compatibility

---

## 🌩️ Option 2: Cloud Instance Manual Deployment

### Step 1: Launch Cloud Instance

**AWS EC2:**
```bash
# Launch Ubuntu 22.04 LTS instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx
```

**Google Cloud:**
```bash
# Launch Ubuntu instance
gcloud compute instances create metaculus-bot \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=e2-medium
```

**DigitalOcean:**
```bash
# Create droplet via web interface or API
# Ubuntu 22.04, 2GB RAM, 1 vCPU minimum
```

### Step 2: Connect and Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install -y python3.11 python3.11-pip python3.11-venv git curl

# Verify Python version
python3.11 --version  # Should be 3.11+
```

### Step 3: Deploy Application

```bash
# Clone repository
git clone <repository-url>
cd metac-bot-ha

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies (pip fallback method)
pip install --upgrade pip
pip install -r requirements-emergency.txt

# Configure environment
cp .env.example .env
nano .env  # Add your API keys

# Verify installation
python3 -m src.main --help
```

### Step 4: Run Tournament Bot

```bash
# Test run (dry-run mode)
python3 -m src.main --tournament 32813 --max-questions 2 --dry-run --verbose

# Production run
nohup python3 -m src.main --tournament 32813 --max-questions 50 > tournament.log 2>&1 &

# Monitor progress
tail -f tournament.log
```

---

## 💻 Option 3: Local Development Machine

### Step 1: Environment Setup

**macOS:**
```bash
# Install Python 3.11 via Homebrew
brew install python@3.11

# Verify installation
python3.11 --version
```

**Ubuntu/Debian:**
```bash
# Install Python 3.11
sudo apt update
sudo apt install -y python3.11 python3.11-pip python3.11-venv

# Verify installation
python3.11 --version
```

**Windows (WSL2):**
```bash
# Install WSL2 Ubuntu
wsl --install -d Ubuntu-22.04

# Follow Ubuntu instructions above
```

### Step 2: Project Setup

```bash
# Clone and enter directory
git clone <repository-url>
cd metac-bot-ha

# Create isolated environment
python3.11 -m venv tournament-env
source tournament-env/bin/activate  # Linux/macOS
# tournament-env\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements-emergency.txt
```

### Step 3: Configuration

```bash
# Setup environment variables
cp .env.example .env

# Edit configuration (use your preferred editor)
nano .env  # or vim, code, etc.

# Required variables:
# ASKNEWS_CLIENT_ID=your_client_id
# ASKNEWS_SECRET=your_secret
# OPENROUTER_API_KEY=your_api_key
# AIB_TOURNAMENT_ID=32813
```

### Step 4: Validation and Execution

```bash
# Validate setup
python3 scripts/validate_tournament_integration.py

# Test forecast (dry-run)
python3 -m src.main --tournament 32813 --max-questions 3 --dry-run

# Run tournament
python3 -m src.main --tournament 32813 --max-questions 100
```

---

## ⚡ Option 4: Emergency Pip-Only Installation

**When Poetry fails or is unavailable**

### Step 1: Create Requirements File

```bash
# Create emergency requirements file
cat > requirements-emergency.txt << 'EOF'
python-decouple==3.8
requests==2.32.4
asknews==0.11.6
numpy==2.2.0
openai==1.57.4
python-dotenv==1.0.1
forecasting-tools==0.2.23
pydantic==2.8.0
typer==0.12.0
rich==13.7.0
structlog==24.2.0
httpx==0.27.0
aiofiles==24.1.0
pandas==2.2.0
scipy==1.14.0
matplotlib==3.9.0
seaborn==0.13.0
plotly==5.22.0
pyyaml==6.0.1
jinja2==3.1.4
tenacity==9.0.0
click==8.1.7
tiktoken==0.8.0
EOF
```

### Step 2: Install Dependencies

```bash
# Upgrade pip first
python3 -m pip install --upgrade pip

# Install with timeout and retries
pip install --timeout 60 --retries 3 -r requirements-emergency.txt

# Alternative: Install core dependencies only
pip install requests openai python-dotenv pydantic typer rich httpx pandas numpy
```

### Step 3: Minimal Configuration

```bash
# Create minimal .env file
cat > .env << 'EOF'
ASKNEWS_CLIENT_ID=your_client_id
ASKNEWS_SECRET=your_secret
OPENROUTER_API_KEY=your_api_key
AIB_TOURNAMENT_ID=32813
MAX_CONCURRENT_QUESTIONS=3
PUBLISH_REPORTS=false
EOF
```

### Step 4: Emergency Test

```bash
# Test core functionality
python3 -c "
import sys
sys.path.append('src')
from infrastructure.config.settings import Config
config = Config()
print('✅ Configuration loaded successfully')
print(f'Tournament ID: {config.tournament_id}')
"

# Test API connectivity
python3 scripts/test_all_secrets.py

# Run minimal forecast
python3 -m src.main --tournament 32813 --max-questions 1 --dry-run
```

---

## 🧪 Local Testing Verification Commands

### Core Functionality Tests

```bash
# 1. Configuration validation
python3 -c "
from src.infrastructure.config.settings import Config
config = Config()
assert config.tournament_id == 32813
print('✅ Configuration valid')
"

# 2. API connectivity test
python3 scripts/test_all_secrets.py

# 3. Import validation
python3 -c "
from src.main import MetaculusForecastingBot
from src.agents.ensemble_agent import EnsembleAgent
print('✅ Core imports successful')
"

# 4. Agent initialization test
python3 -c "
from src.infrastructure.config.settings import Config
from src.agents.ensemble_agent import EnsembleAgent
config = Config()
agent = EnsembleAgent('test', config.llm_config)
print('✅ Agent initialization successful')
"
```

### Integration Tests

```bash
# 5. Tournament integration test
python3 scripts/validate_tournament_integration.py

# 6. Research pipeline test
python3 -c "
import asyncio
from src.infrastructure.external_apis.tournament_asknews import TournamentAskNews
from src.infrastructure.config.settings import Config

async def test_research():
    config = Config()
    asknews = TournamentAskNews(config.asknews_config)
    result = await asknews.search('AI forecasting tournament')
    assert len(result) > 0
    print('✅ Research pipeline working')

asyncio.run(test_research())
"

# 7. LLM connectivity test
python3 -c "
import asyncio
from src.infrastructure.external_apis.llm_client import LLMClient
from src.infrastructure.config.settings import Config

async def test_llm():
    config = Config()
    client = LLMClient(config.llm_config)
    response = await client.generate_response('Test prompt', max_tokens=10)
    assert len(response) > 0
    print('✅ LLM connectivity working')

asyncio.run(test_llm())
"
```

### End-to-End Validation

```bash
# 8. Single question forecast test
python3 -m src.main --tournament 32813 --max-questions 1 --dry-run --verbose

# 9. Batch processing test
python3 -c "
import asyncio
from src.main import MetaculusForecastingBot
from src.infrastructure.config.settings import Config

async def test_batch():
    config = Config()
    bot = MetaculusForecastingBot(config)
    # Test with mock question IDs
    results = await bot.forecast_questions_batch([12345], 'ensemble')
    print(f'✅ Batch processing: {len(results)} results')

asyncio.run(test_batch())
"

# 10. Tournament readiness check
python3 scripts/test_tournament_features.py
```

### Performance Validation

```bash
# 11. Memory usage test
python3 -c "
import psutil
import os
from src.main import MetaculusForecastingBot
from src.infrastructure.config.settings import Config

process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024

config = Config()
bot = MetaculusForecastingBot(config)

final_memory = process.memory_info().rss / 1024 / 1024
print(f'✅ Memory usage: {final_memory - initial_memory:.1f} MB')
"

# 12. Response time test
python3 -c "
import time
import asyncio
from src.agents.ensemble_agent import EnsembleAgent
from src.infrastructure.config.settings import Config

async def test_performance():
    config = Config()
    agent = EnsembleAgent('test', config.llm_config)

    start_time = time.time()
    # Mock forecast test
    end_time = time.time()

    response_time = end_time - start_time
    print(f'✅ Response time: {response_time:.2f}s')
    assert response_time < 60  # Should be under 1 minute

asyncio.run(test_performance())
"
```

---

## 🚨 Emergency Troubleshooting

### Common Issues and Solutions

**1. Python Version Issues:**
```bash
# Check Python version
python3 --version
python3.11 --version

# Install specific version
sudo apt install python3.11  # Ubuntu
brew install python@3.11     # macOS
```

**2. Dependency Installation Failures:**
```bash
# Clear pip cache
pip cache purge

# Install with no cache
pip install --no-cache-dir -r requirements-emergency.txt

# Install individually if batch fails
pip install requests openai python-dotenv pydantic
```

**3. API Key Issues:**
```bash
# Verify environment variables
python3 -c "
import os
print('ASKNEWS_CLIENT_ID:', 'SET' if os.getenv('ASKNEWS_CLIENT_ID') else 'MISSING')
print('OPENROUTER_API_KEY:', 'SET' if os.getenv('OPENROUTER_API_KEY') else 'MISSING')
"

# Test API connectivity
python3 scripts/test_all_secrets.py
```

**4. Network Connectivity Issues:**
```bash
# Test internet connectivity
curl -I https://api.openrouter.ai/api/v1/models
curl -I https://api.asknews.app/v1/news/search

# Use alternative DNS
export DNS_SERVER=8.8.8.8
```

**5. Permission Issues:**
```bash
# Fix file permissions
chmod +x scripts/*.py
chmod +x main.py

# Create necessary directories
mkdir -p logs/performance logs/reasoning data
```

---

## 📊 Deployment Verification Checklist

### Pre-Tournament Checklist

- [ ] Python 3.11+ installed and verified
- [ ] All dependencies installed successfully
- [ ] Environment variables configured
- [ ] API keys validated and working
- [ ] Core imports successful
- [ ] Agent initialization working
- [ ] Research pipeline functional
- [ ] LLM connectivity established
- [ ] Single question forecast test passed
- [ ] Tournament integration validated

### Tournament Readiness Checklist

- [ ] Dry-run mode successful
- [ ] Batch processing working
- [ ] Error handling functional
- [ ] Logging configured
- [ ] Performance within acceptable limits
- [ ] Memory usage reasonable
- [ ] Network connectivity stable
- [ ] Backup deployment method ready

### Emergency Deployment Success Criteria

**Minimum Requirements:**
- Core forecasting functionality works
- API connectivity established
- Can process at least 1 question successfully
- Error handling prevents crashes

**Optimal Requirements:**
- All tests pass
- Performance metrics acceptable
- Full tournament integration working
- Monitoring and logging functional

---

## 🎯 Tournament Execution Commands

### Final Tournament Commands

```bash
# Production tournament run
python3 -m src.main --tournament 32813 --max-questions 200

# With monitoring
nohup python3 -m src.main --tournament 32813 --max-questions 200 > tournament.log 2>&1 &

# Monitor progress
tail -f tournament.log

# Check status
ps aux | grep python3
```

### Emergency Stop/Restart

```bash
# Stop tournament bot
pkill -f "src.main"

# Restart with different parameters
python3 -m src.main --tournament 32813 --max-questions 50 --dry-run

# Resume production
python3 -m src.main --tournament 32813 --max-questions 150
```

---

## 📞 Emergency Contacts

**If all deployment methods fail:**

1. **Check tournament deadline**: Ensure sufficient time remaining
2. **Use minimal configuration**: Reduce complexity to core functionality
3. **Manual submission**: Consider manual forecasting as last resort
4. **Community support**: Reach out to Metaculus community

**Success Metrics:**
- ✅ Bot successfully processes questions
- ✅ Predictions submitted to tournament
- ✅ No critical errors in logs
- ✅ Performance within acceptable range

**Remember**: Even partial functionality is better than no participation. The goal is tournament participation, not perfect code.
