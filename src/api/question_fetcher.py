# question_fetcher.py
"""
Fetches new binary questions from Metaculus API, filters out already answered or skipped questions.
"""
import os
import requests

def fetch_new_questions(token=None, limit=10, dryrun=False):
    import json
    token = token or os.getenv("METACULUS_TOKEN")
    if not token:
        # Stub mode for dry-run/CI
        try:
            with open("./testdata/mock_questions.json") as f:
                questions = json.load(f)
            print("Stub mode: returning mock questions for dry-run validation")
            return questions[:limit]
        except Exception:
            print("Stub mode: returning hardcoded mock question for dry-run validation")
            return [{"question_id": 1, "question_text": "Will AGI be achieved by 2030?"}]
    session = requests.Session()
    session.headers.update({"Authorization": f"Token {token}"})
    url = "https://www.metaculus.com/api2/questions/?status=open&order_by=-publish_time"
    questions = []
    page = 1
    while len(questions) < limit:
        resp = session.get(url + f"&page={page}")
        if resp.status_code != 200:
            break
        data = resp.json()
        for q in data.get("results", []):
            if not q.get("user_prediction") and not q.get("skipped"):
                q_type = q.get("type", "binary")
                if q_type in ("mc", "multiple_choice"):
                    options = q.get("options")
                    if not options:
                        continue
                    questions.append({
                        "question_id": q["id"],
                        "question_text": q["title"],
                        "type": "mc",
                        "options": options
                    })
                else:
                    questions.append({
                        "question_id": q["id"],
                        "question_text": q["title"]
                    })
                if len(questions) >= limit:
                    break
        if not data.get("next"):
            break
        page += 1
    return questions
