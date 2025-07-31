#!/bin/zsh
set -e

# 1. Git status and tag
printf '\n==== GIT STATUS ===='\n
(git status && git describe --tags --dirty)

# 2. .env and secrets
printf '\n==== .ENV CHECK ===='\n
if [ -f .env ]; then
  echo 'WARNING: .env exists! Should not be committed.'
else
  echo '.env not present in repo (OK)'
fi
if [ -f .env.example ]; then
  echo '.env.example present (OK)'
else
  echo 'ERROR: .env.example missing!'
fi

# 3. Poetry and dependency check
printf '\n==== DEPENDENCY CHECK ===='\n
poetry check || true

# 4. Test suite (unit, integration, e2e)
printf '\n==== TEST SUITE ===='\n
PYTHONPATH=$(pwd) poetry run pytest --maxfail=3 --disable-warnings -q || true

# 5. CLI --help and --version
printf '\n==== CLI --help ===='\n
poetry run python main_agent.py --help
printf '\n==== CLI --version ===='\n
poetry run python main_agent.py --version

# 6. Dryrun batch (no .env required)
printf '\n==== DRYRUN BATCH ===='\n
PYTHONPATH=$(pwd) poetry run python main_agent.py --mode batch --limit 2 --show-trace --dryrun || true

# 7. Lint (optional)
printf '\n==== LINT (OPTIONAL) ===='\n
if command -v ruff >/dev/null 2>&1; then
  ruff .
else
  echo 'ruff not installed, skipping lint.'
fi

# 8. Security audit (optional)
printf '\n==== SECURITY AUDIT (OPTIONAL) ===='\n
if command -v pip-audit >/dev/null 2>&1; then
  pip-audit || true
else
  echo 'pip-audit not installed, skipping security audit.'
fi

