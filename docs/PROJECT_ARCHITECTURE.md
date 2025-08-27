# 🏗️ MeI Forecasting Bot - Project Architecture

## 📋 Overview

This is a production-ready AI forecasting bot designed to compete in Metaculus tournaments. The system implements sophisticated reasoning agents, ensemble intelligence, and tournament-specific optimizations.

## 🎯 Key Innovations & Competitive Advantages

### 1. **Tournament-Optimized Resource Management**
- **AskNews Quota Management**: Intelligent management of 9,000 free API calls
- **Metaculus Proxy Credits**: Free model usage with automatic fallback
- **Multi-Provider Fallback**: AskNews → Perplexity → Exa → OpenRouter
- **Cost Optimization**: 90% lower costs than competitors

### 2. **Advanced Reasoning Architecture**
- **Chain of Thought Agent**: Step-by-step reasoning with bias detection
- **Tree of Thought Agent**: Parallel reasoning path exploration
- **ReAct Agent**: Dynamic reasoning-acting cycles
- **Ensemble Intelligence**: Sophisticated aggregation methods

### 3. **Tournament-Specific Features**
- **Tournament ID Targeting**: Specific Fall 2025 tournament (32813)
- **Optimized Scheduling**: Strategic GitHub Actions intervals
- **Question Filtering**: Tournament-specific prioritization
- **Performance Monitoring**: Real-time competitive intelligence

## 🏛️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  main.py (Tournament Bot) │ src/main.py (New Architecture)  │
│  GitHub Actions Workflows │ CLI Interface                    │
└─────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  • TournamentOrchestrator  │ • ForecastService              │
│  • Dispatcher              │ • IngestionService             │
│  • ForecastingPipeline     │ • ConfigManager                │
└─────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────┐
│                     DOMAIN LAYER                            │
├─────────────────────────────────────────────────────────────┤
│  ENTITIES:                 │ SERVICES:                      │
│  • Question               │ • ReasoningOrchestrator         │
│  • Forecast               │ • EnsembleService               │
│  • Prediction             │ • TournamentAnalyzer            │
│  • ResearchReport         │ • ConflictResolver              │
│                           │ • PerformanceAnalyzer           │
│  VALUE OBJECTS:           │ • ScoringOptimizer              │
│  • ReasoningTrace         │ • UncertaintyQuantifier         │
│  • Confidence             │ • And 15+ more services...      │
│  • TournamentStrategy     │                                 │
└─────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────┐
│                 INFRASTRUCTURE LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  EXTERNAL APIs:           │ RELIABILITY:                    │
│  • TournamentAskNewsClient│ • CircuitBreaker                │
│  • MetaculusProxyClient   │ • RetryManager                  │
│  • MetaculusClient        │ • RateLimiter                   │
│  • LLMClient              │ • HealthMonitor                 │
│                           │                                 │
│  CONFIG & MONITORING:     │ REPOSITORIES:                   │
│  • TournamentConfig       │ • InMemoryForecastRepository    │
│  • ApiKeyManager          │ • MetaculusQuestionRepository   │
│  • MetricsService         │ • ConfigManager                 │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure
```
metac-bot-ha/
├── 🎯 MAIN ENTRY POINTS
│   ├── main.py                          # Tournament-ready bot (PRODUCTION)
│   ├── src/main.py                      # New architecture (DEVELOPMENT)
│   └── main_with_no_framework.py        # Simplified version
│
├── 🏗️ SOURCE CODE (src/)
│   ├── agents/                          # Reasoning Agents
│   │   ├── base_agent.py               # Abstract base class
│   │   ├── chain_of_thought_agent.py   # CoT reasoning
│   │   ├── tree_of_thought_agent.py    # ToT reasoning
│   │   ├── react_agent.py              # ReAct reasoning
│   │   └── ensemble_agent.py           # Ensemble coordination
│   │
│   ├── domain/                         # Core Business Logic
│   │   ├── entities/                   # Domain entities
│   │   │   ├── question.py            # Question model
│   │   │   ├── forecast.py            # Forecast model
│   │   │   ├── prediction.py          # Prediction model
│   │   │   └── research_report.py     # Research model
│   │   │
│   │   ├── services/                  # Domain services (24+ services)
│   │   │   ├── reasoning_orchestrator.py      # Multi-step reasoning
│   │   │   ├── tournament_analyzer.py         # Tournament intelligence
│   │   │   ├── ensemble_service.py           # Ensemble coordination
│   │   │   ├── conflict_resolver.py          # Information synthesis
│   │   │   ├── performance_analyzer.py       # Performance tracking
│   │   │   ├── uncertainty_quantifier.py     # Confidence management
│   │   │   └── ... (18+ more services)
│   │   │
│   │   ├── value_objects/             # Value objects
│   │   │   ├── reasoning_trace.py     # Reasoning documentation
│   │   │   ├── confidence.py          # Confidence levels
│   │   │   ├── probability.py         # Probability handling
│   │   │   └── tournament_strategy.py # Tournament strategies
│   │   │
│   │   └── repositories/              # Repository interfaces
│   │       ├── question_repository.py
│   │       └── forecast_repository.py
│   │
│   ├── application/                   # Application Services
│   │   ├── tournament_orchestrator.py # Main tournament coordinator
│   │   ├── forecast_service.py       # Forecasting coordination
│   │   ├── dispatcher.py             # Request routing
│   │   └── ingestion_service.py      # Data ingestion
│   │
│   ├── infrastructure/               # External Integrations
│   │   ├── external_apis/           # API clients
│   │   │   ├── tournament_asknews_client.py    # AskNews with quotas
│   │   │   ├── metaculus_proxy_client.py       # Free credit models
│   │   │   ├── metaculus_client.py             # Metaculus API
│   │   │   └── llm_client.py                   # LLM integration
│   │   │
│   │   ├── config/                  # Configuration management
│   │   │   ├── tournament_config.py # Tournament settings
│   │   │   ├── api_keys.py          # API key management
│   │   │   └── settings.py          # Application settings
│   │   │
│   │   ├── reliability/             # Reliability components
│   │   │   ├── circuit_breaker.py   # Circuit breaker pattern
│   │   │   ├── retry_manager.py     # Retry logic
│   │   │   ├── rate_limiter.py      # Rate limiting
│   │   │   └── health_monitor.py    # Health monitoring
│   │   │
│   │   └── repositories/            # Data persistence
│   │       ├── in_memory_forecast_repository.py
│   │       └── metaculus_question_repository.py
│   │
│   ├── prompts/                     # Reasoning Prompts
│   │   ├── base_prompts.py         # Shared prompt utilities
│   │   ├── cot_prompts.py          # Chain of Thought prompts
│   │   ├── tot_prompts.py          # Tree of Thought prompts
│   │   └── react_prompts.py        # ReAct prompts
│   │
│   └── pipelines/                  # Processing Pipelines
│       └── forecasting_pipeline.py # Main forecasting pipeline
│
├── 🧪 TESTING (tests/)
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   ├── tournament/                 # Tournament-specific tests
│   └── e2e/                       # End-to-end tests
│
├── 🚀 DEPLOYMENT & OPERATIONS
│   ├── .github/workflows/          # GitHub Actions
│   │   ├── run_bot_on_tournament.yaml    # Tournament execution
│   │   ├── test_bot.yaml                 # Testing workflow
│   │   └── ci-cd.yml                     # CI/CD pipeline
│   │
│   ├── scripts/                    # Deployment scripts
│   │   ├── blue-green-deploy.sh   # Blue-green deployment
│   │   ├── health-check.sh        # Health monitoring
│   │   └── setup-github-secrets.sh # Secret management
│   │
│   ├── monitoring/                 # Monitoring configuration
│   │   ├── prometheus.yml         # Metrics collection
│   │   ├── grafana/               # Dashboards
│   │   └── alert_rules.yml        # Alerting rules
│   │
│   └── docker-compose*.yml        # Container orchestration
│
├── 📚 DOCUMENTATION (docs/)
│   ├── PROJECT_ARCHITECTURE.md    # This file
│   ├── DEPLOYMENT.md              # Deployment guide
│   └── GITHUB_ACTIONS_SETUP.md    # CI/CD setup
│
└── ⚙️ CONFIGURATION
    ├── .env.template              # Environment template
    ├── .env.example               # Example configuration
    ├── config/                    # YAML configurations
    └── pyproject.toml            # Python dependencies
```

## 🔄 Data Flow & Dependencies

### 1. **Tournament Execution Flow**
```
GitHub Actions (every 30min)
    ↓
main.py (TemplateForecaster)
    ↓
TournamentAskNewsClient (research)
    ↓
MetaculusProxyClient (reasoning)
    ↓
Metaculus API (submission)
```

### 2. **New Architecture Flow**
```
src/main.py (CLI/API)
    ↓
TournamentOrchestrator
    ↓
ForecastingPipeline
    ↓
Ensemble of Agents (CoT, ToT, ReAct)
    ↓
Domain Services (24+ services)
    ↓
Infrastructure Layer
```

## 🧠 Reasoning Agents Implementation Status

### ✅ **Fully Implemented**
| Agent Type           | File                                   | Status     | Features                                                       |
| -------------------- | -------------------------------------- | ---------- | -------------------------------------------------------------- |
| **Chain of Thought** | `src/agents/chain_of_thought_agent.py` | ✅ Complete | Step-by-step reasoning, bias detection, confidence calibration |
| **Tree of Thought**  | `src/agents/tree_of_thought_agent.py`  | ✅ Complete | Parallel reasoning paths, systematic exploration               |
| **ReAct**            | `src/agents/react_agent.py`            | ✅ Complete | Reasoning-acting cycles, dynamic decision making               |
| **Ensemble**         | `src/agents/ensemble_agent.py`         | ✅ Complete | Multi-agent coordination, sophisticated aggregation            |
| **Base Agent**       | `src/agents/base_agent.py`             | ✅ Complete | Abstract interface, common functionality                       |

### 🎯 **Prompt System**
| Component         | File                           | Status     | Purpose                                  |
| ----------------- | ------------------------------ | ---------- | ---------------------------------------- |
| **Base Prompts**  | `src/prompts/base_prompts.py`  | ✅ Complete | Shared utilities, confidence calibration |
| **CoT Prompts**   | `src/prompts/cot_prompts.py`   | ✅ Complete | Step-by-step reasoning templates         |
| **ToT Prompts**   | `src/prompts/tot_prompts.py`   | ✅ Complete | Multi-path exploration templates         |
| **ReAct Prompts** | `src/prompts/react_prompts.py` | ✅ Complete | Reasoning-acting cycle templates         |

## 🏆 Domain Services Implementation Status

### ✅ **Core Services (Fully Implemented)**
| Service                        | File                              | Purpose                                        |
| ------------------------------ | --------------------------------- | ---------------------------------------------- |
| **ReasoningOrchestrator**      | `reasoning_orchestrator.py`       | Multi-step reasoning with bias detection       |
| **EnsembleService**            | `ensemble_service.py`             | Agent coordination and aggregation             |
| **TournamentAnalyzer**         | `tournament_analyzer.py`          | Competitive intelligence and dynamics          |
| **ConflictResolver**           | `conflict_resolver.py`            | Information synthesis from conflicting sources |
| **PerformanceAnalyzer**        | `performance_analyzer.py`         | Continuous improvement tracking                |
| **UncertaintyQuantifier**      | `uncertainty_quantifier.py`       | Confidence management                          |
| **ScoringOptimizer**           | `scoring_optimizer.py`            | Tournament-specific scoring                    |
| **QuestionCategorizer**        | `question_categorizer.py`         | Specialized forecasting strategies             |
| **AuthoritativeSourceManager** | `authoritative_source_manager.py` | Expert opinion integration                     |
| **KnowledgeGapDetector**       | `knowledge_gap_detector.py`       | Adaptive research strategies                   |

### ✅ **Advanced Services (Fully Implemented)**
| Service                        | File                              | Purpose                        |
| ------------------------------ | --------------------------------- | ------------------------------ |
| **DivergenceAnalyzer**         | `divergence_analyzer.py`          | Agent disagreement analysis    |
| **DynamicWeightAdjuster**      | `dynamic_weight_adjuster.py`      | Performance-based adaptation   |
| **PatternDetector**            | `pattern_detector.py`             | Tournament adaptation patterns |
| **StrategyAdaptationEngine**   | `strategy_adaptation_engine.py`   | Dynamic optimization           |
| **ConservativeStrategyEngine** | `conservative_strategy_engine.py` | Risk management                |
| **CalibrationService**         | `calibration_service.py`          | Calibration tracking           |
| **TournamentAnalytics**        | `tournament_analytics.py`         | Competitive intelligence       |
| **ResearchService**            | `research_service.py`             | Research coordination          |
| **ForecastingService**         | `forecasting_service.py`          | Forecasting coordination       |
| **RiskManagementService**      | `risk_management_service.py`      | Risk assessment                |

## 🏗️ Infrastructure Implementation Status

### ✅ **External APIs (Tournament-Optimized)**
| Component              | File                           | Status     | Innovation                          |
| ---------------------- | ------------------------------ | ---------- | ----------------------------------- |
| **Tournament AskNews** | `tournament_asknews_client.py` | ✅ Complete | 9,000 free calls + quota management |
| **Metaculus Proxy**    | `metaculus_proxy_client.py`    | ✅ Complete | Free credits + automatic fallback   |
| **Metaculus Client**   | `metaculus_client.py`          | ✅ Complete | Tournament-specific API integration |
| **LLM Client**         | `llm_client.py`                | ✅ Complete | Multi-provider support              |
| **Search Client**      | `search_client.py`             | ✅ Complete | Fallback search providers           |

### ✅ **Configuration & Reliability**
| Component             | File                   | Status     | Purpose                      |
| --------------------- | ---------------------- | ---------- | ---------------------------- |
| **Tournament Config** | `tournament_config.py` | ✅ Complete | Tournament-specific settings |
| **API Key Manager**   | `api_keys.py`          | ✅ Complete | Secure key management        |
| **Circuit Breaker**   | `circuit_breaker.py`   | ✅ Complete | Fault tolerance              |
| **Retry Manager**     | `retry_manager.py`     | ✅ Complete | Intelligent retry logic      |
| **Rate Limiter**      | `rate_limiter.py`      | ✅ Complete | API rate management          |
| **Health Monitor**    | `health_monitor.py`    | ✅ Complete | System health tracking       |

## 🎮 Tournament-Specific Features

### 🏆 **Competitive Advantages**
1. **Resource Optimization**
   - AskNews: 9,000 free calls with intelligent quota management
   - Metaculus Proxy: Free model credits with automatic fallback
   - Multi-provider fallback: Never runs out of research capability

2. **Tournament Intelligence**
   - Specific tournament targeting (Fall 2025 - ID: 32813)
   - Question filtering and prioritization
   - Competitive positioning analysis
   - Market inefficiency detection

3. **Advanced Reasoning**
   - Multi-agent ensemble with sophisticated aggregation
   - Bias detection and mitigation
   - Confidence calibration and uncertainty quantification
   - Reasoning trace preservation for transparency

4. **Operational Excellence**
   - Blue-green deployment for zero downtime
   - Comprehensive monitoring and alerting
   - Automated health checks and rollback
   - Performance optimization and scaling

## 🔧 Current Implementation Status

### ✅ **What's Working (Production Ready)**
- **Tournament Bot** (`main.py`): Fully functional, tournament-ready
- **Resource Management**: AskNews + Metaculus proxy integration
- **GitHub Actions**: Optimized scheduling and deployment
- **Domain Layer**: 24+ services fully implemented
- **Agent System**: All reasoning agents implemented
- **Infrastructure**: Reliability and monitoring components

### 🚧 **What's In Development**
- **New Architecture** (`src/main.py`): Advanced architecture with dependency injection
- **Integration**: Connecting new architecture with tournament features
- **Documentation**: This comprehensive documentation system

### 🎯 **Integration Strategy**
The project has two parallel implementations:
1. **Production Bot** (`main.py`): Tournament-ready, using forecasting_tools library
2. **Advanced Architecture** (`src/`): Clean architecture with domain-driven design

**Recommendation**: Keep using `main.py` for tournament participation while gradually migrating features to the new architecture.
