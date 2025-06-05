# infrastructure_layer.instructions.md

## MetaculusClient Infrastructure Layer

### Auth Setup
- Requires `METACULUS_TOKEN` in `.env` or environment
- Injected into requests as `Authorization: Token ...`

### Method Contract
#### `submit_forecast(question_id, forecast, justification)`
- Validates payload (question_id: int, forecast: float [0-1], justification: str)
- Submits to `/questions/{id}/predict/` endpoint
- Retries on 5xx, handles 4xx, timeouts, and schema errors
- Returns status dict:
  - `{"status": "success", ...}`
  - `{"status": "error", "error": ...}`

### JSON Schema
```json
{
  "type": "object",
  "properties": {
    "question_id": {"type": "integer"},
    "forecast": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "justification": {"type": "string"}
  },
  "required": ["question_id", "forecast", "justification"]
}
```

### Mocking & Testing
- Use `unittest.mock.patch` on `client.session.post` for all network calls
- Test all error and edge cases (see `test_metaculus_client.py`)
- Integration: run ForecastChain, pass result to submit_forecast, assert payload

---

### Example
```python
from src.api.metaculus_client import MetaculusClient
client = MetaculusClient(token="MY_TOKEN")
result = client.submit_forecast(123, 0.7, "Justification")
```
