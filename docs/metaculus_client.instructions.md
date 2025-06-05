# metaculus_client.instructions.md

## Purpose

Handles forecast submission to Metaculus via API or forecasting-tools.

## Input

```json
{
  "question_id": 123,
  "forecast": 0.7,
  "justification": "Test justification."
}
```

## Output

```json
{
  "status": "success",
  "question_id": 123,
  "response": { "result": "ok" }
}
```

## Reasoning Logic

- Accepts forecast dict
- Authenticates using METACULUS_TOKEN
- Submits forecast to /questions/{id}/predict/
- Handles errors (auth, malformed, network)

## Related Tests

- `tests/unit/api/test_metaculus_client.py`

---

## Example Usage

```python
from src.api.metaculus_client import MetaculusClient
client = MetaculusClient(token="MY_TOKEN")
result = client.submit({"question_id": 123, "forecast": 0.7, "justification": "Test"})
```
