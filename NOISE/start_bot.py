#!/usr/bin/env python3
"""
Quick start script for the tournament bot.
This handles SSL issues and gets the bot running quickly.
"""

import os
import sys
import subprocess

def setup_ssl():
    """Fix SSL certificate issues."""
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        print("✅ SSL certificates configured")
    except ImportError:
        print("⚠️  certifi not found, installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "certifi"])
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

def check_env():
    """Check environment variables."""
    required_vars = ["METACULUS_TOKEN", "OPENROUTER_API_KEY"]
    missing = []

    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please check your .env file")
        return False

    print("✅ Environment variables configured")
    return True

def main():
    """Main startup function."""
    print("🏆 METACULUS TOURNAMENT BOT - QUICK START")
    print("=" * 50)

    # Setup SSL
    setup_ssl()

    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment loaded from .env")
    except ImportError:
        print("⚠️  python-dotenv not found, environment variables may not load")

    # Check environment
    if not check_env():
        print("\n❌ Environment setup failed. Please configure your .env file.")
        return

    print("\n🚀 Starting tournament bot...")
    print("Mode: Tournament (Fall 2025 - ID: 32813)")
    print("Features: Tournament optimizations, AskNews integration, fallback chains")

    # Run the bot
    try:
        print("\n" + "="*50)
        print("🏆 BOT IS READY FOR TOURNAMENT!")
        print("Run: python3 main.py --mode tournament")
        print("Or: python3 main.py --mode test_questions")
        print("="*50)

    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        print("Try running: python3 main.py --mode test_questions")

if __name__ == "__main__":
    main()
