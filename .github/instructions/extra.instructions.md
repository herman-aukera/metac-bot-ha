Commit Style
	•	test(unit): add edge test for invalid timestamps
	•	test(e2e): validate ingestion → forecast flow
Commit Examples
	•	ci: add GitHub Action for forecast every 30m
	•	fix(makefile): ensure python3 used in run

---
# 🧠 Project Bootstrap Prompt — Metaculus AI Forecast Bot

Resume work on the Metaculus AI Forecast Bot using these `.instructions.md` files as system entrypoints.

✅ Use Clean Architecture strictly.
✅ Each file = one concern, one layer.
✅ Use TDD and commit after each test pass.
✅ Target 80%+ coverage before next milestone.
✅ No real API calls (mock only).

Start with application/dispatcher.py and its tests. Don’t wait for approval if tests pass.