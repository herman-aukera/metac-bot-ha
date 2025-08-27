#!/bin/bash

# Local development script that loads environment variables from .env file
set -e

echo "🤖 Running AI Forecasting Bot locally"
echo "====================================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "Please copy .env.example to .env and configure your API keys"
    exit 1
fi

# Load environment variables from .env file
echo "📋 Loading environment variables from .env file..."
export $(grep -v '^#' .env | xargs)

# Validate required API keys
echo "🔐 Validating API keys..."
python -c "
from src.infrastructure.config.api_keys import api_key_manager
import sys

validation = api_key_manager.validate_required_keys()
api_key_manager.log_key_status()

if not validation['valid']:
    print('❌ Missing required API keys:', [k['key'] for k in validation['missing_keys']])
    print('Please configure the missing keys in your .env file')
    sys.exit(1)
else:
    print('✅ All required API keys are configured')
"

if [ $? -ne 0 ]; then
    echo "❌ API key validation failed"
    exit 1
fi

# Run the bot
echo "🚀 Starting the forecasting bot..."
echo "Mode: ${1:-tournament}"

if [ "$1" = "quarterly_cup" ]; then
    python main.py --mode quarterly_cup
else
    python main.py
fi
