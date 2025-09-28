# Repository Usage Analysis & Cleanup Map

**Generated**: September 22, 2025
**Purpose**: Systematic analysis of file usage to guide repo cleanup while preserving functional pipelines

## Executive Summary

This analysis identifies files by usage status to enable surgical cleanup:

- **ACTIVE**: Core pipeline files that must be preserved
- **LIKELY_UNUSED**: Candidates for NOISE/ folder (test utilities, docs, duplicates)
- **CLEANUP_CANDIDATES**: Files that may be consolidated or removed

## 1. ACTIVE COMPONENTS (Core Pipeline - DO NOT MOVE)

### Primary Entrypoints & Flow

- `main.py` - Core entrypoint; wires forecasting_tools.TemplateForecaster; writes run_summary.json
- `src/infrastructure/config/tournament_config.py` - Tournament target resolution (MiniBench vs Seasonal)
- `src/infrastructure/config/enhanced_llm_config.py` - LLM routing (GPT-5 tiers only)
- `src/infrastructure/config/openrouter_startup_validator.py` - OpenRouter validation
- `src/domain/services/multi_stage_research_pipeline.py` - Core research pipeline

### GitHub Workflows (Active CI)

- `.github/workflows/run_bot_on_minibench.yaml` - MiniBench tournament pipeline
- `.github/workflows/test_deployment.yaml` - Smoke tests & deployment verification

### Essential Scripts

- `scripts/local_minibench_ci.sh` - Local CI simulator (referenced in docs & tasks)

### Core Architecture (src/ - Domain/Application/Infrastructure)

```
src/
├── domain/           # Pure logic, no I/O
├── application/      # Orchestration via DI
└── infrastructure/   # I/O adapters (Metaculus, OpenRouter, AskNews)
```

## 2. LIKELY_UNUSED COMPONENTS (Stage 1 → NOISE/)

### Documentation Artifacts

- `CodeFlattened_Output/` - Generated flattened code analysis
- Multiple `*_FIXES.md`, `*_REPORT.md` files (48 files)
- `COMPREHENSIVE_PROJECT_ANALYSIS.md`
- `DOCUMENTACION_PROYECTO.md`

### Test/Debug Scripts (Root Level)

- `test_*.py` files in root (18+ files)
- `run_task10_*.py`
- `verify_task9_completion.py`
- `quick_fix.py`
- `start_bot.py`

### Alternative/Legacy Entrypoints

- `main_with_no_framework.py`
- `main_working.py`
- `community_benchmark.py`

### Config Alternatives

- `config/` directory (separate from src/infrastructure/config/)
- `configs/` directory
- Multiple docker-compose variants

### Shell Scripts (Non-essential)

- `auto_fix_linting.sh`
- `fix_linting_issues.sh`
- Most scripts/ files except `local_minibench_ci.sh`

## 3. SUSPECT COMPONENTS (Stage 2 Investigation)

### Workflow Proliferation

**18 GitHub workflow files** - Many appear to be variants/experiments:

- `run_bot_on_tournament_fixed.yaml` vs `run_bot_on_tournament.yaml`
- `test_bot_resilient.yaml` vs `test_bot.yaml`
- Multiple budget/cost monitoring workflows

### Docker Configurations

- `docker-compose.yml` (main)
- `docker-compose.blue.yml`, `.green.yml`, `.staging.yml` (variants)

### Requirements Files

- `requirements-emergency.txt` (vs pyproject.toml)

## 4. USAGE EVIDENCE

### Direct References in Active Code

```bash
# From main.py imports
from forecasting_tools import ForecastBot, MetaculusApi, AskNewsSearcher

# From src/infrastructure/deployment/deployment_manager.py
script_path = "./scripts/blue-green-deploy.sh"
["./scripts/rollback.sh", "Emergency rollback"]
```

### Workflow Dependencies

```yaml
# .github/workflows/run_bot_on_minibench.yaml uses:
- main.py (entrypoint)
- scripts/local_minibench_ci.sh (referenced in docs)
```

### Task References

```json
// VS Code tasks.json equivalent functionality
"Run local_minibench_ci": "bash scripts/local_minibench_ci.sh"
"Run quick OpenRouter config check": "python -m scripts.test_openrouter_simple"
```

## 5. CLEANUP STRATEGY

### Stage 1: Safe Move to NOISE/

Move clearly unused items preserving directory structure:

```bash
mkdir -p NOISE/
mv CodeFlattened_Output/ NOISE/
mv *_FIXES.md *_REPORT.md NOISE/
mv test_*.py NOISE/ # (root level only)
mv main_with_no_framework.py main_working.py NOISE/
mv config/ configs/ NOISE/ # (conflicting with src/infrastructure/config/)
```

### Stage 2: Workflow Consolidation

Keep only verified-working workflows:

- `run_bot_on_minibench.yaml` (active)
- `test_deployment.yaml` (smoke tests)
Move variants to NOISE/

### Stage 3: Final Cleanup

After 2 successful full runs, delete NOISE/

## 6. VERIFICATION PLAN

### Pre-Cleanup Test

```bash
# MiniBench (real)
export DRY_RUN=false AIB_MINIBENCH_TOURNAMENT_SLUG=minibench
export SCHEDULING_FREQUENCY_HOURS=1.4
python3 main.py --mode tournament

# Seasonal (real)
unset AIB_MINIBENCH_TOURNAMENT_SLUG
export AIB_TOURNAMENT_ID=32813 SCHEDULING_FREQUENCY_HOURS=5.5
python3 main.py --mode tournament
```

### Expected Evidence

- Target printed: "minibench" vs "32813"
- Model routing: only GPT-5 tiers (gpt-5, gpt-5-mini, gpt-5-nano)
- Scheduling frequency: parsed from env (1.4→2, 5.5→6)
- Output: run_summary.json with success/failure status

## 7. RISK ASSESSMENT

### Low Risk (Stage 1)

- Documentation artifacts
- Test scripts in root
- Generated code flattener output

### Medium Risk (Stage 2)

- Alternative workflow files
- Docker variants
- Legacy scripts in scripts/

### High Risk (Never Move)

- main.py
- src/ directory structure
- pyproject.toml
- .github/workflows/run_bot_on_minibench.yaml
- .github/workflows/test_deployment.yaml
- scripts/local_minibench_ci.sh

## 8. COMMIT STRATEGY

1. `feat: add USAGE_MAP.md for systematic repo cleanup`
2. `chore(cleanup:stage1): move unused assets to NOISE/ (no functional changes)`
   → Test both pipelines (real runs)
3. `chore(cleanup:stage2): consolidate duplicate workflows and configs`
   → Test both pipelines (real runs)
4. `chore(cleanup:stage3): remove NOISE/ after verification`
   → Test both pipelines (real runs)
5. `docs(readme): align with cleaned, verified pipelines`

---

**Assumptions:**

- Python 3.11+
- .env populated with METACULUS_TOKEN, OPENROUTER_API_KEY, ASKNEWS_CLIENT_ID/SECRET
- Paid GPT-5 models enabled on OpenRouter account
- Temperature 0.2 for deterministic cleanup operations

**Evidence Links:**

- OpenRouter docs: <https://openrouter.ai/docs#chat-completions>
- Forecasting tools: verify import compatibility with current version
- Copilot instructions: /Users/herman/Documents/Code/l1dr/metac-bot-ha/.github/copilot-instructions.md
