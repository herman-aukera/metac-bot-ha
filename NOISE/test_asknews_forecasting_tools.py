#!/usr/bin/env python3
"""
AskNews Test using forecasting_tools library
Tests the new AskNews credentials using the same approach as the codebase.
"""
import os
import asyncio
from dotenv import load_dotenv

async def test_asknews_with_forecasting_tools():
    """Test AskNews using forecasting_tools library."""
    print("🔍 Testing AskNews with forecasting_tools...")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    client_id = os.getenv('ASKNEWS_CLIENT_ID')
    client_secret = os.getenv('ASKNEWS_SECRET')

    if not client_id or not client_secret:
        print("❌ Missing AskNews credentials")
        return False

    print(f"✅ Client ID: {client_id[:8]}...{client_id[-8:]}")
    print(f"✅ Secret: {client_secret[:4]}...{client_secret[-4:]}")

    try:
        # Try to import forecasting_tools
        print("\n📦 Importing forecasting_tools...")
        from forecasting_tools import AskNewsSearcher
        print("✅ forecasting_tools imported successfully")

        # Initialize AskNews searcher
        print("\n🔧 Initializing AskNews searcher...")
        searcher = AskNewsSearcher()
        print("✅ AskNews searcher initialized")

        # Test a simple query
        print("\n🔍 Testing news search...")
        question = "artificial intelligence developments"

        print(f"📡 Searching for: {question}")
        result = await searcher.get_formatted_news_async(question)

        if result and len(result.strip()) > 0:
            print("✅ AskNews search successful!")
            print(f"📊 Result length: {len(result)} characters")
            print(f"📄 Sample: {result[:200]}...")
            return True
        else:
            print("❌ AskNews returned empty result")
            return False

    except ImportError as e:
        print(f"❌ forecasting_tools not available: {e}")
        print("💡 Try: pip install forecasting_tools")
        return False
    except Exception as e:
        print(f"💥 AskNews test failed: {e}")
        return False

async def main():
    """Main test execution."""
    print("🚀 AskNews forecasting_tools Test")
    print("Testing new credentials with the same library used in codebase")
    print("=" * 60)

    success = await test_asknews_with_forecasting_tools()

    print("\n" + "=" * 60)
    if success:
        print("🎉 AskNews forecasting_tools test PASSED!")
        print("✅ New credentials work with forecasting_tools")
        print("🔗 Ready for tournament integration")
    else:
        print("❌ AskNews forecasting_tools test FAILED!")
        print("⚠️ Check credentials or install forecasting_tools")

    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
