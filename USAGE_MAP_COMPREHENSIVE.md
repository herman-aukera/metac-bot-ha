# Repository Usage Analysis & Cleanup Map

**Generated**: September 22, 2025
**Purpose**: Systematic analysis of file usage to guide surgical cleanup while preserving functional pipelines

## Executive Summary

Files categorized by usage status for cleanup strategy:

- **ACTIVE (127 files)**: Core pipeline files that must be preserved
- **CLEANUP_CANDIDATES (45+ files)**: Move to NOISE/ folder - test utilities, docs, duplicates
- **VERIFICATION_NEEDED**: Files requiring usage confirmation before moving

## 1. ACTIVE COMPONENTS - DO NOT MOVE

### Primary Entrypoints & Flow

- `main.py` - Core entrypoint; wires forecasting_tools.TemplateForecaster; writes run_summary.json
- `pyproject.toml` - Project dependencies and configuration
- `requirements-emergency.txt` - Fallback dependencies
- `.env` (if exists) - Environment secrets

### Core Architecture (src/)

```
src/
├── domain/               # Pure business logic
│   ├── entities/        # Core domain objects (Question, Forecast, etc.)
│   ├── services/        # Domain services
│   │   └── multi_stage_research_pipeline.py  # CRITICAL - core research flow
│   └── value_objects/   # Domain value types
├── application/         # Orchestration layer
│   ├── dispatcher.py    # Main application orchestrator
│   ├── forecast_service.py # Forecasting business logic
│   └── ingestion_service.py # Data ingestion
├── infrastructure/      # I/O adapters
│   ├── config/         # Configuration management
│   │   ├── enhanced_llm_config.py      # CRITICAL - GPT-5 routing
│   │   ├── tournament_config.py        # CRITICAL - tournament resolution
│   │   └── openrouter_startup_validator.py # CRITICAL - API validation
│   ├── external_apis/  # External service adapters
│   ├── metaculus_api/  # Metaculus integration
│   └── logging/        # Logging infrastructure
└── pipelines/          # Processing pipelines
    └── forecasting_pipeline.py # Forecast processing
```

### GitHub Workflows (Active CI)

- `.github/workflows/run_bot_on_minibench.yaml` - MiniBench tournament pipeline
- `.github/workflows/test_deployment.yaml` - Smoke tests & deployment verification
- `.github/copilot-instructions.md` - Copilot configuration (referenced by workflows)

### Essential Scripts

- `scripts/local_minibench_ci.sh` - Local CI simulator (referenced in copilot-instructions.md)

### Build & Deployment

- `Makefile` - Build automation
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Local development setup

## 2. CLEANUP CANDIDATES - MOVE TO NOISE/

### Test Utilities & Verification Scripts (24+ files)

```
test_*.py files (root level - not in tests/):
├── test_all_secrets_fixed.py
├── test_all_secrets.py
├── test_asknews_*.py (multiple variants)
├── test_openrouter_*.py (multiple variants)
├── test_pipeline*.py
├── test_full_integration.py
├── run_task10_*.py (performance/integration/unit test runners)
├── verify_task9_completion.py
├── simple_openrouter_test.py
└── quick_fix.py
```

**Evidence**: These are standalone test utilities, not part of core pipeline

### Documentation Artifacts (15+ files)

```
Documentation files:
├── *_FIXES.md, *_REPORT.md, *_SUMMARY.md (pipeline/linting fixes)
├── COMPREHENSIVE_PROJECT_ANALYSIS.md
├── DOCUMENTACION_PROYECTO.md
├── DEPLOYMENT_*.md files
├── NETWORK_RESILIENCE_GUIDE.md
├── openrouter_setup_guide.md
├── SECURITY_*.md files
├── TASK_4_IMPLEMENTATION_SUMMARY.md
├── TOURNAMENT_READY.md
└── USAGE_MAP.md (this file - can be moved after cleanup)
```

**Evidence**: Historical documentation, not referenced in active workflows

### Generated & Temporary Files (8+ files)

```
Generated/temporary:
├── CodeFlattened_Output/ (entire directory)
├── *.log files (run logs)
├── run_summary_*.json (old summaries)
├── *_deployment_report.json
├── coverage.xml
├── main_with_no_framework.py (alternative entrypoint)
├── main_working.py (backup)
├── start_bot.py (alternative entrypoint)
├── update_poetry_lock.py (utility script)
└── community_benchmark.py (unused benchmark)
```

**Evidence**: Generated outputs, backups, alternatives not used in workflows

### Duplicate/Alternative Configs (6+ files)

```
Config duplicates:
├── config/config.*.yaml (dev/test/prod variants - not used in main.py)
├── docker-compose.*.yml (blue/green/staging variants)
├── agent.yaml (not referenced)
└── poetry.lock (if using pip fallback)
```

**Evidence**: Config variants not referenced in active workflows

### Build Artifacts & Cache

```
Cache/artifacts:
├── __pycache__/
├── .pytest_cache/
├── htmlcov/
├── bin/ (contains actionlint binary)
└── temp/, data/, logs/, monitoring/ (runtime directories)
```

**Evidence**: Runtime artifacts that can be recreated

## 3. VERIFICATION NEEDED

### Scripts Directory Analysis

Need to verify which scripts are actually referenced:

```bash
scripts/ contents analysis needed:
- Check which scripts are called by workflows
- Check which scripts are mentioned in documentation
- Check Makefile references
```

### Alternative Workflows

```
workflows/ directory analysis needed:
- Identify which .yml files are actually triggered
- Check for duplicate or obsolete workflow definitions
```

## 4. USAGE EVIDENCE SOURCES

### Import Analysis

Core imports from `main.py`:
- `forecasting_tools` (external dependency)
- `src.infrastructure.config.*` (config modules)
- Internal src/ module structure

### Workflow References

From `.github/workflows/run_bot_on_minibench.yaml`:
- `main.py` execution
- `run_summary.json` artifact upload
- Environment variable usage

### Task References

From VS Code tasks:
- `scripts.test_openrouter_simple`
- `scripts.test_openrouter_corrected`
- `scripts/local_minibench_ci.sh`

## 5. CLEANUP EXECUTION PLAN

### Stage 1: Safe Move to NOISE/
1. Create `NOISE/` directory with identical subpaths
2. Move all CLEANUP_CANDIDATES to `NOISE/`
3. Verify both MiniBench and Seasonal tournaments still work
4. Commit: "chore(cleanup:stage1): move unused assets to NOISE/"

### Stage 2: Consolidation
1. Resolve any remaining duplicates
2. Update references if needed
3. Re-test both tournaments
4. Commit: "chore(cleanup:stage2): consolidate remaining duplicates"

### Stage 3: Final Cleanup
1. After 2 successful tournament runs, delete NOISE/ contents
2. Final tournament verification
3. Commit: "chore(cleanup:stage3): remove unused assets"

### Stage 4: Documentation
1. Update README to reflect cleaned state
2. Commit: "docs(readme): align with cleaned, verified pipelines"

---

**Next Actions**:
1. Run live MiniBench pipeline (DRY_RUN=false)
2. Run live Seasonal pipeline (DRY_RUN=false)
3. Verify GPT-5 model usage only
4. Execute cleanup stages if pipelines are green
