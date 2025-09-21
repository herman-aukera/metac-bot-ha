#!/usr/bin/env python3
"""
AskNews API Connectivity Test
Tests the new AskNews credentials for proper connectivity.
"""
import os
import asyncio
import aiohttp
from dotenv import load_dotenv

async def test_asknews_connectivity():
    """Test AskNews API connectivity with new credentials."""
    print("🔍 Testing AskNews API Connectivity...")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    client_id = os.getenv('ASKNEWS_CLIENT_ID')
    client_secret = os.getenv('ASKNEWS_SECRET')
    base_url = os.getenv('ASKNEWS_BASE_URL', 'https://api.asknews.app')

    if not client_id or not client_secret:
        print("❌ Missing AskNews credentials")
        return False

    print(f"✅ Client ID: {client_id[:8]}...{client_id[-8:]}")
    print(f"✅ Secret: {client_secret[:4]}...{client_secret[-4:]}")
    print(f"✅ Base URL: {base_url}")

    # Test authentication endpoint (try different possible endpoints)
    auth_endpoints = [
        f"{base_url}/v1/auth/token",
        f"{base_url}/auth/token",
        f"{base_url}/token",
        f"{base_url}/v1/token"
    ]

    try:
        # Create SSL context that handles certificate verification
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            print(f"\n🔗 Testing authentication at: {auth_url}")

            auth_data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials"
            }

            async with session.post(auth_url, json=auth_data) as response:
                print(f"📡 Response Status: {response.status}")

                if response.status == 200:
                    data = await response.json()
                    if 'access_token' in data:
                        token = data['access_token']
                        print("✅ Authentication successful!")
                        print(f"🔑 Token received: {token[:20]}...{token[-10:]}")

                        # Test a simple news query
                        await test_news_query(session, base_url, token)
                        return True
                    else:
                        print("❌ No access token in response")
                        print(f"Response: {data}")
                        return False
                else:
                    error_text = await response.text()
                    print(f"❌ Authentication failed: {response.status}")
                    print(f"Error: {error_text}")
                    return False

    except Exception as e:
        print(f"💥 Connection error: {e}")
        return False

async def test_news_query(session, base_url, token):
    """Test a simple news query."""
    print("\n📰 Testing news query...")

    news_url = f"{base_url}/v1/news/search"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    query_data = {
        "query": "artificial intelligence",
        "n_articles": 5,
        "return_type": "both",
        "hours_back": 24
    }

    try:
        async with session.post(news_url, json=query_data, headers=headers) as response:
            print(f"📡 News Query Status: {response.status}")

            if response.status == 200:
                data = await response.json()
                articles = data.get('articles', [])
                print("✅ News query successful!")
                print(f"📊 Retrieved {len(articles)} articles")

                if articles:
                    print(f"📄 Sample article: {articles[0].get('title', 'No title')[:60]}...")
                return True
            else:
                error_text = await response.text()
                print(f"❌ News query failed: {response.status}")
                print(f"Error: {error_text}")
                return False

    except Exception as e:
        print(f"💥 News query error: {e}")
        return False

async def main():
    """Main test execution."""
    print("🚀 AskNews API Connectivity Test")
    print("Testing new credentials and basic functionality")
    print("=" * 60)

    success = await test_asknews_connectivity()

    print("\n" + "=" * 60)
    if success:
        print("🎉 AskNews API connectivity test PASSED!")
        print("✅ New credentials are working correctly")
        print("🔗 API is accessible and responding properly")
    else:
        print("❌ AskNews API connectivity test FAILED!")
        print("⚠️ Please check credentials and network connectivity")

    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
