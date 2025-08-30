#!/bin/bash

# Manual Cloud Deployment Script for Metaculus Tournament Bot
# Supports AWS EC2, Google Cloud, DigitalOcean, and other Linux instances
#
# Usage:
#   chmod +x scripts/manual_cloud_deployment.sh
#   ./scripts/manual_cloud_deployment.sh
#   ./scripts/manual_cloud_deployment.sh --quick
#   ./scripts/manual_cloud_deployment.sh --tournament-only

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
REPO_URL="https://github.com/your-org/metac-bot-ha.git"  # Update with actual repo URL
TOURNAMENT_ID="32813"
MAX_QUESTIONS="100"

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_success() {
    print_status "$GREEN" "✅ $1"
}

print_error() {
    print_status "$RED" "❌ $1"
}

print_warning() {
    print_status "$YELLOW" "⚠️  $1"
}

print_info() {
    print_status "$BLUE" "ℹ️  $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            echo "ubuntu"
        elif command_exists yum; then
            echo "centos"
        elif command_exists dnf; then
            echo "fedora"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Function to install Python 3.11
install_python() {
    local os=$(detect_os)

    print_info "Installing Python $PYTHON_VERSION..."

    case $os in
        "ubuntu")
            sudo apt update
            sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update
            sudo apt install -y python$PYTHON_VERSION python$PYTHON_VERSION-pip python$PYTHON_VERSION-venv python$PYTHON_VERSION-dev
            ;;
        "centos"|"fedora")
            sudo dnf install -y python$PYTHON_VERSION python$PYTHON_VERSION-pip python$PYTHON_VERSION-devel
            ;;
        "macos")
            if command_exists brew; then
                brew install python@$PYTHON_VERSION
            else
                print_error "Homebrew not found. Please install Python $PYTHON_VERSION manually."
                exit 1
            fi
            ;;
        *)
            print_error "Unsupported OS. Please install Python $PYTHON_VERSION manually."
            exit 1
            ;;
    esac

    # Verify installation
    if command_exists python$PYTHON_VERSION; then
        print_success "Python $PYTHON_VERSION installed successfully"
        python$PYTHON_VERSION --version
    else
        print_error "Python $PYTHON_VERSION installation failed"
        exit 1
    fi
}

# Function to install system dependencies
install_system_deps() {
    local os=$(detect_os)

    print_info "Installing system dependencies..."

    case $os in
        "ubuntu")
            sudo apt update
            sudo apt install -y git curl wget build-essential libssl-dev libffi-dev
            ;;
        "centos"|"fedora")
            sudo dnf install -y git curl wget gcc openssl-devel libffi-devel
            ;;
        "macos")
            # Most dependencies should be available via Xcode command line tools
            if ! command_exists git; then
                print_error "Git not found. Please install Xcode command line tools."
                exit 1
            fi
            ;;
    esac

    print_success "System dependencies installed"
}

# Function to setup project
setup_project() {
    print_info "Setting up project..."

    # Clone repository if not already present
    if [ ! -d "metac-bot-ha" ]; then
        print_info "Cloning repository..."
        git clone "$REPO_URL" metac-bot-ha
    else
        print_info "Repository already exists, pulling latest changes..."
        cd metac-bot-ha
        git pull origin main || git pull origin master
        cd ..
    fi

    cd metac-bot-ha

    # Create virtual environment
    print_info "Creating virtual environment..."
    python$PYTHON_VERSION -m venv tournament-env

    # Activate virtual environment
    source tournament-env/bin/activate

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip

    print_success "Project setup complete"
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."

    # Try Poetry first, fallback to pip
    if [ -f "pyproject.toml" ] && command_exists poetry; then
        print_info "Installing with Poetry..."
        poetry install --only=main
    elif [ -f "requirements-emergency.txt" ]; then
        print_info "Installing with pip (emergency requirements)..."
        pip install --timeout 60 --retries 3 -r requirements-emergency.txt
    elif [ -f "requirements.txt" ]; then
        print_info "Installing with pip (standard requirements)..."
        pip install --timeout 60 --retries 3 -r requirements.txt
    else
        print_warning "No requirements file found, installing core dependencies..."
        pip install --timeout 60 --retries 3 \
            requests openai python-dotenv pydantic typer rich httpx \
            asknews forecasting-tools numpy pandas structlog aiofiles \
            pyyaml jinja2 tenacity click tiktoken
    fi

    print_success "Dependencies installed"
}

# Function to configure environment
configure_environment() {
    print_info "Configuring environment..."

    # Copy example environment file if it exists
    if [ -f ".env.example" ]; then
        if [ ! -f ".env" ]; then
            cp .env.example .env
            print_info "Created .env file from example"
        fi
    else
        # Create minimal .env file
        cat > .env << EOF
# Metaculus Tournament Bot Configuration
ASKNEWS_CLIENT_ID=your_client_id_here
ASKNEWS_SECRET=your_secret_here
OPENROUTER_API_KEY=your_api_key_here
AIB_TOURNAMENT_ID=$TOURNAMENT_ID
MAX_CONCURRENT_QUESTIONS=5
PUBLISH_REPORTS=false
EOF
        print_info "Created minimal .env file"
    fi

    print_warning "Please edit .env file with your actual API keys:"
    print_info "  - ASKNEWS_CLIENT_ID: Your AskNews client ID"
    print_info "  - ASKNEWS_SECRET: Your AskNews secret"
    print_info "  - OPENROUTER_API_KEY: Your OpenRouter API key"

    # Create necessary directories
    mkdir -p logs/performance logs/reasoning data

    print_success "Environment configured"
}

# Function to run verification tests
run_verification() {
    print_info "Running deployment verification..."

    # Run verification script if it exists
    if [ -f "scripts/emergency_deployment_verification.py" ]; then
        python3 scripts/emergency_deployment_verification.py --quick
    else
        # Run basic verification
        print_info "Running basic verification tests..."

        # Test Python version
        python3 --version

        # Test imports
        python3 -c "
import sys
sys.path.append('src')
try:
    from infrastructure.config.settings import Config
    print('✅ Configuration import successful')
except Exception as e:
    print(f'❌ Configuration import failed: {e}')
    sys.exit(1)
"

        # Test environment variables
        python3 -c "
import os
required_vars = ['ASKNEWS_CLIENT_ID', 'ASKNEWS_SECRET', 'OPENROUTER_API_KEY']
missing = [var for var in required_vars if not os.getenv(var) or os.getenv(var) == f'your_{var.lower()}_here']
if missing:
    print(f'❌ Missing or placeholder environment variables: {missing}')
    print('Please edit .env file with actual API keys')
    sys.exit(1)
else:
    print('✅ Environment variables configured')
"
    fi

    print_success "Verification complete"
}

# Function to run tournament bot
run_tournament() {
    local dry_run=${1:-"--dry-run"}

    print_info "Starting tournament bot..."

    if [ "$dry_run" = "--dry-run" ]; then
        print_info "Running in dry-run mode (no actual submissions)..."
        python3 -m src.main --tournament $TOURNAMENT_ID --max-questions 3 --dry-run --verbose
    else
        print_info "Running in production mode..."
        python3 -m src.main --tournament $TOURNAMENT_ID --max-questions $MAX_QUESTIONS
    fi
}

# Function to setup monitoring
setup_monitoring() {
    print_info "Setting up monitoring..."

    # Create monitoring script
    cat > monitor_tournament.sh << 'EOF'
#!/bin/bash
# Tournament monitoring script

echo "🏆 Metaculus Tournament Bot Monitor"
echo "=================================="

# Check if bot is running
if pgrep -f "src.main" > /dev/null; then
    echo "✅ Bot is running"
    echo "Process ID: $(pgrep -f 'src.main')"
else
    echo "❌ Bot is not running"
fi

# Show recent logs
if [ -f "tournament.log" ]; then
    echo ""
    echo "📊 Recent log entries:"
    tail -10 tournament.log
fi

# Show resource usage
echo ""
echo "💻 Resource usage:"
ps aux | grep python3 | grep -v grep | head -5

echo ""
echo "💾 Disk usage:"
df -h . | tail -1

echo ""
echo "🔄 To restart bot: ./restart_tournament.sh"
echo "📋 To view full logs: tail -f tournament.log"
EOF

    chmod +x monitor_tournament.sh

    # Create restart script
    cat > restart_tournament.sh << EOF
#!/bin/bash
# Tournament restart script

echo "🔄 Restarting tournament bot..."

# Stop existing bot
pkill -f "src.main" || true
sleep 2

# Start bot in background
source tournament-env/bin/activate
nohup python3 -m src.main --tournament $TOURNAMENT_ID --max-questions $MAX_QUESTIONS > tournament.log 2>&1 &

echo "✅ Bot restarted"
echo "📋 Monitor with: ./monitor_tournament.sh"
EOF

    chmod +x restart_tournament.sh

    print_success "Monitoring scripts created"
}

# Function to show deployment summary
show_summary() {
    print_success "🎉 Deployment Complete!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Edit .env file with your API keys"
    echo "2. Test with: python3 -m src.main --tournament $TOURNAMENT_ID --max-questions 1 --dry-run"
    echo "3. Run production: ./restart_tournament.sh"
    echo "4. Monitor progress: ./monitor_tournament.sh"
    echo ""
    echo "📁 Important files:"
    echo "  - .env: Configuration file (edit with your API keys)"
    echo "  - tournament.log: Bot execution logs"
    echo "  - monitor_tournament.sh: Monitoring script"
    echo "  - restart_tournament.sh: Restart script"
    echo ""
    echo "🚨 Emergency commands:"
    echo "  - Stop bot: pkill -f 'src.main'"
    echo "  - View logs: tail -f tournament.log"
    echo "  - Check status: ps aux | grep python3"
}

# Main deployment function
main() {
    local mode=${1:-"full"}

    print_info "🚀 Metaculus Tournament Bot - Manual Cloud Deployment"
    print_info "======================================================"

    # Check if running as root (not recommended)
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root is not recommended. Consider using a regular user."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    case $mode in
        "--quick")
            print_info "Quick deployment mode - skipping system dependencies"
            setup_project
            install_dependencies
            configure_environment
            run_verification
            ;;
        "--tournament-only")
            print_info "Tournament-only mode - assuming environment is ready"
            if [ ! -d "tournament-env" ]; then
                print_error "Virtual environment not found. Run full deployment first."
                exit 1
            fi
            source tournament-env/bin/activate
            run_tournament "--production"
            ;;
        *)
            print_info "Full deployment mode"
            install_system_deps

            # Check if Python is already installed
            if ! command_exists python$PYTHON_VERSION; then
                install_python
            else
                print_success "Python $PYTHON_VERSION already installed"
            fi

            setup_project
            install_dependencies
            configure_environment
            run_verification
            setup_monitoring
            show_summary
            ;;
    esac

    print_success "🏆 Ready for tournament!"
}

# Handle script arguments
if [ $# -eq 0 ]; then
    main
else
    main "$1"
fi
