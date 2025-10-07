#!/usr/bin/env python3
"""
Diagnostic script to check why questions aren't being forecasted.
"""
import os
import sys
import json
import requests

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
QUESTION_IDS = [39549, 39398, 39368]

def check_question(qid):
    """Check status of a specific question."""
    print(f"\n{'='*80}")
    print(f"Question {qid}")
    print(f"{'='*80}")
    print(f"URL: https://www.metaculus.com/questions/{qid}/")

    headers = {}
    if METACULUS_TOKEN:
        headers["Authorization"] = f"Token {METACULUS_TOKEN}"

    try:
        resp = requests.get(
            f"https://www.metaculus.com/api/questions/{qid}/",
            headers=headers,
            timeout=10
        )

        if resp.status_code != 200:
            print(f"❌ API returned status {resp.status_code}")
            print(f"Response: {resp.text[:200]}")
            return

        q = resp.json()

        print(f"Title: {q.get('title', 'N/A')}")
        print(f"Status: {q.get('status', 'unknown')}")
        print(f"Type: {q.get('type', 'unknown')}")
        print(f"Close time: {q.get('close_time', 'N/A')}")

        tournaments = q.get('tournaments', [])
        tournament_info = [f"{t.get('name')} (ID: {t.get('id')})" for t in tournaments]
        print(f"Tournaments: {tournament_info}")

        # Check if we have forecasts
        my_forecasts = q.get('my_forecasts', {})
        if my_forecasts:
            latest = my_forecasts.get('latest', {})
            print(f"✅ HAS FORECAST: {latest.get('probability', 'N/A')}")
            print(f"   Forecast time: {latest.get('time', 'N/A')}")
        else:
            print(f"❌ NO FORECAST FOUND")

        # Check prediction count
        num_predictions = q.get('number_of_predictions', 0)
        print(f"Total predictions on question: {num_predictions}")

    except Exception as e:
        print(f"❌ Error: {e}")

def check_tournament(tournament_id):
    """Check tournament status and question list."""
    print(f"\n{'='*80}")
    print(f"Tournament {tournament_id}")
    print(f"{'='*80}")

    headers = {}
    if METACULUS_TOKEN:
        headers["Authorization"] = f"Token {METACULUS_TOKEN}"

    try:
        resp = requests.get(
            f"https://www.metaculus.com/api/tournaments/{tournament_id}/",
            headers=headers,
            timeout=10
        )

        if resp.status_code != 200:
            print(f"❌ API returned status {resp.status_code}")
            return

        t = resp.json()
        print(f"Name: {t.get('name', 'N/A')}")
        print(f"Status: {t.get('status', 'unknown')}")

        # Get questions in tournament
        questions_resp = requests.get(
            f"https://www.metaculus.com/api/questions/?tournaments={tournament_id}",
            headers=headers,
            timeout=10
        )

        if questions_resp.status_code == 200:
            questions_data = questions_resp.json()
            results = questions_data.get('results', [])
            print(f"Total questions: {len(results)}")

            # Check if our questions are in this tournament
            question_ids_in_tournament = [q.get('id') for q in results]
            for qid in QUESTION_IDS:
                if qid in question_ids_in_tournament:
                    print(f"✅ Question {qid} IS in tournament")
                else:
                    print(f"❌ Question {qid} NOT in tournament")

    except Exception as e:
        print(f"❌ Error: {e}")

def check_workflow_runs():
    """Check recent GitHub Actions runs."""
    print(f"\n{'='*80}")
    print("Recent Workflow Runs")
    print(f"{'='*80}")

    workflows = [
        "run_bot_on_tournament.yaml",
        "run_bot_on_minibench.yaml"
    ]

    for workflow in workflows:
        print(f"\n{workflow}:")
        try:
            resp = requests.get(
                f"https://api.github.com/repos/herman-aukera/metac-bot-ha/actions/workflows/{workflow}/runs?per_page=3",
                timeout=10
            )

            if resp.status_code == 200:
                runs = resp.json().get('workflow_runs', [])
                for run in runs[:3]:
                    status = run.get('status')
                    conclusion = run.get('conclusion')
                    created = run.get('created_at', '')[:19]
                    print(f"  {created}: {status:10} {conclusion:10}")
            else:
                print(f"  ❌ Failed to fetch: {resp.status_code}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    print("=" * 80)
    print("METAC FORECAST DIAGNOSTIC")
    print("=" * 80)

    if not METACULUS_TOKEN:
        print("⚠️  METACULUS_TOKEN not set - limited information available")
    else:
        print(f"✅ METACULUS_TOKEN set ({len(METACULUS_TOKEN)} chars)")

    # Check each question
    for qid in QUESTION_IDS:
        check_question(qid)

    # Check tournament
    check_tournament(32813)

    # Check workflow runs
    check_workflow_runs()

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nIf questions show NO FORECAST:")
    print("1. Check if they're in tournament 32813")
    print("2. Check if workflow runs are being cancelled (concurrency)")
    print("3. Check if circuit breaker is open")
    print("4. Review workflow logs in GitHub Actions")
    print()

if __name__ == "__main__":
    main()
