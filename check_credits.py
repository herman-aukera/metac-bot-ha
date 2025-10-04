#!/usr/bin/env python3
"""Quick script to check OpenRouter credit status."""

import os
import requests
import json

def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("No OpenRouter API key found")
        return 1

    try:
        # Check credit balance
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers)

        if response.status_code == 200:
            data = response.json()
            print("=== OpenRouter Credit Status ===")
            print(f"Label: {data.get('label', 'N/A')}")
            print(f"Usage: ${data.get('usage', 0):.4f}")
            print(f"Limit: ${data.get('limit', 0):.2f}")

            rate_limit = data.get('rate_limit', {})
            print(f"Rate Limit: {rate_limit.get('requests', 'N/A')} req/min")

            usage = float(data.get('usage', 0))
            limit = float(data.get('limit', 0))
            remaining = limit - usage
            percent_used = (usage / limit * 100) if limit > 0 else 0

            print(f"Remaining: ${remaining:.4f}")
            print(f"Percent Used: {percent_used:.1f}%")

            if percent_used > 95:
                print("⚠️  CRITICAL: Nearly out of credits!")
            elif percent_used > 80:
                print("⚠️  WARNING: Credits running low")
            else:
                print("✅ Credit balance OK")

        else:
            print(f"Error checking credits: {response.status_code}")
            print(f"Response: {response.text}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
