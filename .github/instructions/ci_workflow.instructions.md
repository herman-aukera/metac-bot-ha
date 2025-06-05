---
applyTo: ".github/workflows/*.yml"
---

## 🚦 GitHub CI – Agent Execution on Schedule

- Run: `make test`, `make forecast`, `make lint`
- Scheduled runs every 30min:
```yaml
on:
  schedule:
    - cron: '*/30 * * * *'

Actions
	•	Run LangChain agents in test mode
	•	Generate markdown logs
	•	Post Slack/Discord notification if predictions > 10

Commit Tags
	•	ci: add agent forecast workflow
	•	ci: push log to Discord