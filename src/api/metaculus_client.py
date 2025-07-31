# metaculus_client.py
"""
MetaculusClient: Handles forecast submission to Metaculus via API or forecasting-tools.
- Accepts forecast dicts (question_id, forecast, justification)
- Handles auth via METACULUS_TOKEN from .env
- Returns status dict
"""
import os
import requests
import time
import jsonschema
from jsonschema import ValidationError

FORECAST_SCHEMA = {
    "type": "object",
    "properties": {
        "question_id": {"type": "integer"},
        "forecast": {
            "oneOf": [
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "minItems": 2
                },
                {
                    "type": "object",
                    "properties": {
                        "prediction": {"type": "number"},
                        "low": {"type": "number"},
                        "high": {"type": "number"}
                    },
                    "required": ["prediction", "low", "high"]
                }
            ]
        },
        "justification": {"type": "string"}
    },
    "required": ["question_id", "forecast", "justification"]
}

class MetaculusClient:
    def __init__(self, api_url=None, token=None, max_retries=3, timeout=10, validate_fn=None):
        self.api_url = api_url or "https://www.metaculus.com/api2"
        self.token = token or os.getenv("METACULUS_TOKEN")
        if not self.token:
            raise ValueError("METACULUS_TOKEN not set in environment or .env")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {self.token}"})
        self.max_retries = max_retries
        self.timeout = timeout
        self._validate = validate_fn or jsonschema.validate

    def submit_forecast(self, question_id, forecast, justification):
        payload = {"question_id": question_id, "forecast": forecast, "justification": justification}
        try:
            self._validate(instance=payload, schema=FORECAST_SCHEMA)
        except ValidationError as ve:
            raise ValueError(f"Invalid payload: {ve.message}")
        url = f"{self.api_url}/questions/{question_id}/predict/"
        if isinstance(forecast, list):
            req_payload = {"values": forecast}
        elif isinstance(forecast, dict) and all(k in forecast for k in ("prediction", "low", "high")):
            req_payload = {"value": forecast["prediction"], "low": forecast["low"], "high": forecast["high"]}
        else:
            req_payload = {"value": forecast}
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(url, json=req_payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return {"status": "success", "question_id": question_id, "response": resp.json()}
                elif resp.status_code == 401:
                    return {"status": "error", "error": "Auth failed", "code": 401}
                elif 400 <= resp.status_code < 500:
                    return {"status": "error", "error": resp.text, "code": resp.status_code}
                elif 500 <= resp.status_code < 600:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return {"status": "error", "error": resp.text, "code": resp.status_code}
            except requests.Timeout:
                if attempt == self.max_retries - 1:
                    return {"status": "error", "error": "Timeout"}
                time.sleep(2 ** attempt)
            except Exception as e:
                return {"status": "error", "error": str(e)}
        return {"status": "error", "error": "Max retries exceeded"}

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
