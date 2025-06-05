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

# MetaculusClient Instructions

## Forecast Input Schema

- **Binary:**
  - `forecast`: float (0.0–1.0)
  - Example: `{"question_id": 1, "forecast": 0.7, "justification": "..."}`
- **Multi-Choice (MC):**
  - `forecast`: list[float] (each 0.0–1.0, sum ≈ 1.0)
  - Example: `{"question_id": 2, "forecast": [0.1, 0.3, 0.6], "justification": "..."}`

## Validation Path
- All forecasts are validated against the schema **before** any network/API call.
- MC and binary are both supported and enforced.
- On validation error, a `ValueError` is raised with details.

## Error Handling
- Auth/network errors are handled after validation.
- Validation errors are always local and CI-safe.

## Test Coverage
- All validation logic is tested for both binary and MC.
- Network-dependent tests are skipped or fully mocked.
- See: `tests/unit/api/test_metaculus_client.py`

---
