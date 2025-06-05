# metaculus_client.py
"""
MetaculusClient: Handles forecast submission to Metaculus via API or forecasting-tools.
- Accepts forecast dicts (question_id, forecast, justification)
- Handles auth via METACULUS_TOKEN from .env
- Returns status dict
"""
import os
import requests

class MetaculusClient:
    def __init__(self, api_url=None, token=None):
        self.api_url = api_url or "https://www.metaculus.com/api2"
        self.token = token or os.getenv("METACULUS_TOKEN")
        if not self.token:
            raise ValueError("METACULUS_TOKEN not set in environment or .env")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {self.token}"})

    def submit(self, forecast: dict) -> dict:
        """
        Submits a forecast dict to Metaculus.
        Args:
            forecast: dict with keys 'question_id', 'forecast', 'justification'
        Returns:
            dict: {status: 'success'|'error', ...}
        """
        try:
            qid = forecast["question_id"]
            value = forecast["forecast"]
            # Metaculus expects a POST to /questions/{id}/predict/
            url = f"{self.api_url}/questions/{qid}/predict/"
            payload = {"value": value}
            resp = self.session.post(url, json=payload)
            if resp.status_code == 200:
                return {"status": "success", "question_id": qid, "response": resp.json()}
            elif resp.status_code == 401:
                return {"status": "error", "error": "Auth failed", "code": 401}
            else:
                return {"status": "error", "error": resp.text, "code": resp.status_code}
        except Exception as e:
            return {"status": "error", "error": str(e)}
