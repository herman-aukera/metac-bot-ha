#!/usr/bin/env python3
"""
Simple AskNews API Test
Tests the new AskNews credentials with a direct approach.
"""
import os
import requests
from dotenv import load_dotenv

def test_asknews_simple():
    """Simple test of AskNews API."""
    print("🔍 Simple AskNews API Test...")
    print("=" * 40)

    # Load environment variables
    load_dotenv()

    client_id = os.getenv('ASKNEWS_CLIENT_ID')
    client_secret = os.getenv('ASKNEWS_SECRET')

    if not client_id or not client_secret:
        print("❌ Missing AskNews credentials")
        return False

    print(f"✅ Client ID: {client_id[:8]}...{client_id[-8:]}")
    print(f"✅ Secret: {client_secret[:4]}...{client_secret[-4:]}")

    # Try direct news search with basic auth
    print(f"\n🔗 Testing direct news search...")

    try:
        # Use requests with basic auth
        url = "https://api.asknews.app/v1/news/search"

        headers = {
            "Content-Type": "application/json"
        }

        # Try with basic auth first
        auth = (client_id, client_secret)

        data = {
            "query": "artificial intelligence",
            "n_articles": 3,
            "hours_back": 24
        }

        print(f"📡 Making GET request to: {url}")
        # Try GET with params instead of POST with JSON
        params = {
            "query": "artificial intelligence",
            "n_articles": 3,
            "hours_back": 24
        }
        response = requests.get(url, params=params, headers=headers, auth=auth, timeout=10, verify=False)

        print(f"📊 Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            articles = result.get('articles', [])
            print(f"✅ Success! Retrieved {len(articles)} articles")
            if articles:
                print(f"📄 Sample: {articles[0].get('title', 'No title')[:50]}...")
            return True
        elif response.status_code == 401:
            print("🔑 Trying alternative authentication methods...")
            return test_alternative_auth(client_id, client_secret)
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False

    except Exception as e:
        print(f"💥 Error: {e}")
        return False

def test_alternative_auth(client_id, client_secret):
    """Try alternative authentication methods."""
    print("🔄 Trying token-based authentication...")

    try:
        # Try to get token first
        token_url = "https://api.asknews.app/v1/auth/token"

        token_data = {
            "client_id": client_id,
            "client_secret": client_secret
        }

        response = requests.post(token_url, json=token_data, verify=False, timeout=10)
        print(f"Token request status: {response.status_code}")

        if response.status_code == 200:
            token_result = response.json()
            access_token = token_result.get('access_token')

            if access_token:
                print(f"✅ Got token: {access_token[:20]}...")

                # Now try news search with token
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }

                news_url = "https://api.asknews.app/v1/news/search"
                data = {
                    "query": "artificial intelligence",
                    "n_articles": 3,
                    "hours_back": 24
                }

                news_response = requests.post(news_url, json=data, headers=headers, verify=False, timeout=10)

                if news_response.status_code == 200:
                    result = news_response.json()
                    articles = result.get('articles', [])
                    print(f"✅ Token auth success! Retrieved {len(articles)} articles")
                    return True
                else:
                    print(f"❌ News request failed: {news_response.status_code}")
                    print(f"Response: {news_response.text[:200]}...")
                    return False
            else:
                print("❌ No access token in response")
                return False
        else:
            print(f"❌ Token request failed: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return False

    except Exception as e:
        print(f"💥 Alternative auth error: {e}")
        return False

def main():
    """Main test execution."""
    print("🚀 AskNews API Simple Test")
    print("Testing new credentials")
    print("=" * 40)

    success = test_asknews_simple()

    print("\n" + "=" * 40)
    if success:
        print("🎉 AskNews test PASSED!")
        print("✅ Credentials are working")
    else:
        print("❌ AskNews test FAILED!")
        print("⚠️ Check credentials or API endpoint")

    return success

if __name__ == "__main__":
    result = main()
    exit(0 if result else 1)
