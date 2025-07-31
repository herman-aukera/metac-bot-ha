#!/bin/bash

# Script to migrate the tournament optimization system to a new repository
# This script will create a clean new repository with only the relevant files

set -e  # Exit on any error

NEW_REPO_NAME="metac-agent-agent"
CURRENT_DIR=$(pwd)

echo "🚀 Migrating Tournament Optimization System to new repository: $NEW_REPO_NAME"
echo "Current directory: $CURRENT_DIR"

# Step 1: Create new repository directory
echo ""
echo "📁 Step 1: Creating new repository directory..."
cd ..
if [ -d "$NEW_REPO_NAME" ]; then
    echo "⚠️  Directory $NEW_REPO_NAME already exists. Removing it..."
    rm -rf "$NEW_REPO_NAME"
fi
mkdir "$NEW_REPO_NAME"
cd "$NEW_REPO_NAME"

# Step 2: Initialize git repository
echo ""
echo "🔧 Step 2: Initializing git repository..."
git init
git branch -M main

# Step 3: Copy essential files and directories
echo ""
echo "📋 Step 3: Copying project files..."

# Copy main entry points
echo "  📄 Copying main entry points..."
cp "$CURRENT_DIR/main.py" .
cp "$CURRENT_DIR/main_agent.py" .
cp "$CURRENT_DIR/main_with_no_framework.py" .

# Copy additional Python entry points
if [ -f "$CURRENT_DIR/community_benchmark.py" ]; then
    cp "$CURRENT_DIR/community_benchmark.py" .
fi

# Copy configuration files
echo "  ⚙️  Copying configuration files..."
cp "$CURRENT_DIR/pyproject.toml" .
cp "$CURRENT_DIR/poetry.lock" .
if [ -f "$CURRENT_DIR/.env.example" ]; then
    cp "$CURRENT_DIR/.env.example" .
fi
if [ -f "$CURRENT_DIR/agent.yaml" ]; then
    cp "$CURRENT_DIR/agent.yaml" .
fi

# Copy source code
echo "  💻 Copying source code..."
cp -r "$CURRENT_DIR/src" .

# Copy tests (all test directories)
echo "  🧪 Copying tests..."
cp -r "$CURRENT_DIR/tests" .

# Copy test data
if [ -d "$CURRENT_DIR/testdata" ]; then
    cp -r "$CURRENT_DIR/testdata" .
fi

# Copy data directory
if [ -d "$CURRENT_DIR/data" ]; then
    cp -r "$CURRENT_DIR/data" .
fi

# Copy configuration directories
echo "  📁 Copying configuration directories..."
if [ -d "$CURRENT_DIR/configs" ]; then
    cp -r "$CURRENT_DIR/configs" .
fi

# Copy scripts
echo "  📜 Copying scripts..."
if [ -d "$CURRENT_DIR/scripts" ]; then
    cp -r "$CURRENT_DIR/scripts" .
fi

# Copy infrastructure
echo "  🏗️  Copying infrastructure..."
if [ -d "$CURRENT_DIR/infrastructure" ]; then
    cp -r "$CURRENT_DIR/infrastructure" .
fi

# Copy Kubernetes configs
if [ -d "$CURRENT_DIR/k8s" ]; then
    cp -r "$CURRENT_DIR/k8s" .
fi

# Copy documentation
echo "  📚 Copying documentation..."
if [ -d "$CURRENT_DIR/docs" ]; then
    cp -r "$CURRENT_DIR/docs" .
fi

# Copy CLI
if [ -d "$CURRENT_DIR/cli" ]; then
    cp -r "$CURRENT_DIR/cli" .
fi

# Copy chatmodes
if [ -d "$CURRENT_DIR/chatmodes" ]; then
    cp -r "$CURRENT_DIR/chatmodes" .
fi

# Copy GitHub workflows and configurations
echo "  🔄 Copying GitHub configurations..."
if [ -d "$CURRENT_DIR/.github" ]; then
    cp -r "$CURRENT_DIR/.github" .
fi

# Copy Kiro specs
if [ -d "$CURRENT_DIR/.kiro" ]; then
    cp -r "$CURRENT_DIR/.kiro" .
fi

# Copy specstory if it exists
if [ -d "$CURRENT_DIR/.specstory" ]; then
    cp -r "$CURRENT_DIR/.specstory" .
fi

# Copy Docker files
echo "  🐳 Copying Docker files..."
if [ -f "$CURRENT_DIR/Dockerfile" ]; then
    cp "$CURRENT_DIR/Dockerfile" .
fi
if [ -f "$CURRENT_DIR/.dockerignore" ]; then
    cp "$CURRENT_DIR/.dockerignore" .
fi

# Copy build and deployment files
echo "  🔨 Copying build and deployment files..."
if [ -f "$CURRENT_DIR/Makefile" ]; then
    cp "$CURRENT_DIR/Makefile" .
fi
if [ -f "$CURRENT_DIR/preprod_audit.sh" ]; then
    cp "$CURRENT_DIR/preprod_audit.sh" .
fi

# Copy documentation files
echo "  📖 Copying documentation files..."
if [ -f "$CURRENT_DIR/CHANGELOG.md" ]; then
    cp "$CURRENT_DIR/CHANGELOG.md" .
fi
if [ -f "$CURRENT_DIR/DEPLOY.md" ]; then
    cp "$CURRENT_DIR/DEPLOY.md" .
fi
if [ -f "$CURRENT_DIR/RELEASE_CHECKLIST.md" ]; then
    cp "$CURRENT_DIR/RELEASE_CHECKLIST.md" .
fi
if [ -f "$CURRENT_DIR/RELEASE_NOTES.md" ]; then
    cp "$CURRENT_DIR/RELEASE_NOTES.md" .
fi

# Copy security and audit files
if [ -f "$CURRENT_DIR/security_report.json" ]; then
    cp "$CURRENT_DIR/security_report.json" .
fi

# Copy any additional important files that might be missed
echo "  🔍 Copying any additional important files..."
# Copy any .json, .yaml, .yml files in root
find "$CURRENT_DIR" -maxdepth 1 -name "*.json" -o -name "*.yaml" -o -name "*.yml" | while read file; do
    if [ -f "$file" ]; then
        cp "$file" .
    fi
done

# Copy any .md files in root (except README which we'll replace)
find "$CURRENT_DIR" -maxdepth 1 -name "*.md" ! -name "README.md" ! -name "README_new_repo.md" | while read file; do
    if [ -f "$file" ]; then
        cp "$file" .
    fi
done

# Step 4: Create new README
echo ""
echo "📝 Step 4: Creating README..."
cp "$CURRENT_DIR/README_new_repo.md" README.md

# Step 5: Create .gitignore
echo ""
echo "🚫 Step 5: Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
logs/
*.log

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/

# nyc test coverage
.nyc_output

# Dependency directories
node_modules/

# Optional npm cache directory
.npm

# Optional eslint cache
.eslintcache

# Output of 'npm pack'
*.tgz

# Yarn Integrity file
.yarn-integrity

# dotenv environment variables file
.env.test

# parcel-bundler cache (https://parceljs.org/)
.cache
.parcel-cache

# next.js build output
.next

# nuxt.js build output
.nuxt

# vuepress build output
.vuepress/dist

# Serverless directories
.serverless/

# FuseBox cache
.fusebox/

# DynamoDB Local files
.dynamodb/

# TernJS port file
.tern-port

# Stores VSCode versions used for testing VSCode extensions
.vscode-test

# Temporary folders
tmp/
temp/

# Security
security_report.json
*.pem
*.key
*.crt

# Local development
.local/
local_config.py
EOF

# Step 6: Create environment example
echo ""
echo "🔧 Step 6: Creating environment example..."
cat > .env.example << 'EOF'
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
EXA_API_KEY=your_exa_api_key_here
ASKNEWS_CLIENT_ID=your_asknews_client_id_here
ASKNEWS_SECRET=your_asknews_secret_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Metaculus
METACULUS_API_TOKEN=your_metaculus_token_here
METACULUS_USERNAME=your_username_here

# Database (optional)
DATABASE_URL=postgresql://user:password@localhost:5432/metac_agent
REDIS_URL=redis://localhost:6379/0

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Features
ENABLE_CACHING=true
ENABLE_MONITORING=true
ENABLE_ADVANCED_FEATURES=true

# Performance
MAX_CONCURRENT_QUESTIONS=5
CACHE_TTL_HOURS=24
REQUEST_TIMEOUT_SECONDS=30

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=localhost,127.0.0.1
EOF

# Step 7: Create initial commit
echo ""
echo "💾 Step 7: Creating initial commit..."
git add .
git commit -m "Initial commit: Tournament Optimization System

🚀 Features:
- Complete tournament orchestration and integration layer
- Advanced forecasting pipeline with multi-agent ensemble
- Comprehensive error handling and resilience mechanisms
- REST API and CLI interfaces
- Backward compatibility with existing main entry points
- Full test coverage and monitoring capabilities

🏗️ Architecture:
- Clean architecture with domain-driven design
- Async processing and concurrent question handling
- Circuit breakers, retry strategies, and graceful degradation
- Distributed tracing and correlation context
- Intelligent caching and performance optimization

🔧 Interfaces:
- CLI: Complete command-line interface
- REST API: Full REST API with OpenAPI documentation
- Legacy compatibility: Seamless integration with existing systems

📊 Monitoring:
- Health checks and system metrics
- Performance benchmarking and regression detection
- Security scanning and audit logging
- Comprehensive error tracking and recovery

🧪 Testing:
- Unit, integration, and end-to-end tests
- Performance and load testing
- Security and penetration testing
- Chaos engineering and resilience testing"

# Step 8: Verify what was copied
echo ""
echo "🔍 Step 8: Verifying copied files..."
echo "📊 File count by type:"
echo "  Python files: $(find . -name "*.py" | wc -l)"
echo "  Test files: $(find ./tests -name "*.py" 2>/dev/null | wc -l)"
echo "  Config files: $(find . -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" | wc -l)"
echo "  Documentation: $(find . -name "*.md" | wc -l)"
echo "  Scripts: $(find ./scripts -name "*.py" -o -name "*.sh" 2>/dev/null | wc -l)"

echo ""
echo "📁 Directory structure:"
ls -la

echo ""
echo "🏗️  Key directories present:"
for dir in src tests configs scripts infrastructure k8s docs cli .github .kiro; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir/ ($(find $dir -type f | wc -l) files)"
    else
        echo "  ❌ $dir/ (missing)"
    fi
done

echo ""
echo "📄 Key files present:"
for file in main.py main_agent.py pyproject.toml Dockerfile Makefile; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (missing)"
    fi
done

# Step 9: Display next steps
echo ""
echo "✅ Repository setup complete!"
echo ""
echo "=========================================="
echo "🎯 Next Steps:"
echo "=========================================="
echo ""
echo "1. 🌐 Create the repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: $NEW_REPO_NAME"
echo "   - Make it public or private as needed"
echo "   - Don't initialize with README (we already have one)"
echo ""
echo "2. 🔗 Connect and push to GitHub:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/$NEW_REPO_NAME.git"
echo "   git push -u origin main"
echo ""
echo "3. 🔧 Set up the development environment:"
echo "   poetry install"
echo "   cp .env.example .env"
echo "   # Edit .env with your API keys"
echo ""
echo "4. 🧪 Run tests to verify everything works:"
echo "   poetry run pytest"
echo ""
echo "5. 🚀 Try the system:"
echo "   python main.py --mode test_questions --use-optimization"
echo ""
echo "📍 Current location: $(pwd)"
echo "📁 Repository size: $(du -sh . | cut -f1)"
echo "📄 Files created: $(find . -type f | wc -l)"
echo ""
echo "🎉 Your new repository is ready!"
echo "🔗 After pushing, it will be available at:"
echo "   https://github.com/YOUR_USERNAME/$NEW_REPO_NAME"
EOF
