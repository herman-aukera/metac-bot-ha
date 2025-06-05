# ✅ Testing Instructions

## Testing Philosophy
Every module must be:
- Unit tested before implementation (TDD)
- Covered ≥ 80%
- Modularly testable in isolation

## Test Scope
- **Unit**: All domain + application classes
- **Integration**: From ingestion to forecast
- **E2E**: Full pipeline with dummy API data

## Commands
```bash
make test
python3 -m pytest
pytest --cov=src --cov-report=term-missing