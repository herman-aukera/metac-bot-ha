# metaculus_client.py
# Minimal Metaculus API client for forecast submission (stub)

class MetaculusClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or "DUMMY_KEY"
        # Cargar de env o config si es necesario

    def submit_forecast(self, question_id, forecast, justification):
        # Implementar API real si es necesario
        print(f"[MetaculusClient] Submitting forecast: {forecast} for QID: {question_id}")
        print(f"Justification: {justification}")
        return {"status": "success", "question_id": question_id, "forecast": forecast}
