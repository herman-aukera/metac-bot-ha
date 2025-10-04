#!/usr/bin/env python3
"""Investigate DateQuestion structure from Metaculus API."""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import httpx


async def investigate_date_question():
    """Investigate the structure of a failing date question."""

    # One of the failing date question URLs
    question_url = "https://www.metaculus.com/api2/questions/40028/"

    async with httpx.AsyncClient() as client:
        response = await client.get(question_url)
        question_data = response.json()

    print("🔍 Investigating Date Question Structure")
    print("=" * 50)
    print(f"Question ID: {question_data.get('id')}")
    print(f"Title: {question_data.get('title')}")
    print(f"Question Type: {question_data.get('question_type')}")
    print(f"Type: {question_data.get('type')}")
    print()

    # Check for date-specific fields
    date_fields = ['open_time', 'close_time', 'resolve_time', 'scheduled_resolve_time']
    for field in date_fields:
        if field in question_data:
            print(f"{field}: {question_data[field]}")

    print()
    print("Scaling info:")
    scaling = question_data.get('scaling', {})
    if scaling:
        print(f"  Range min: {scaling.get('range_min')}")
        print(f"  Range max: {scaling.get('range_max')}")
        print(f"  Zero point: {scaling.get('zero_point')}")
        print(f"  Scale type: {scaling.get('deriv_ratio')}")

    # Look for any date-related fields in the question data
    print("\nAll fields containing 'time' or 'date':")
    for key, value in question_data.items():
        if 'time' in key.lower() or 'date' in key.lower():
            print(f"  {key}: {value}")

    print(f"\nRaw question_type: '{question_data.get('question_type')}'")
    print(f"Raw type: '{question_data.get('type')}'")


if __name__ == "__main__":
    asyncio.run(investigate_date_question())
