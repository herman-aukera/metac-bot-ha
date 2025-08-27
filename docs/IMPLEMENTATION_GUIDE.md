# 🛠️ Implementation & Reconstruction Guide

## 📋 Overview

This guide provides everything needed to understand, reconstruct, or extend the Metaculus AI Forecasting Bot from scratch.

## 🚀 Quick Start (Get Running in 5 Minutes)

### 1. **Clone and Setup**
```bash
git clone <repository>
cd metac-bot-ha
cp .env.template .env
# Edit .env with your API keys
```

### 2. **Install Dependencies**
```bash
# Using Poetry (recommended)
poetry install
poetry shell

# Or using pip
pip install -r requirements.txt
```

### 3. **Run Tournament Bot (Production)**
```bash
# Test mode
python main.py --mode test_questions

# Tournament mode (requires METACULUS_TOKEN)
python main.py --mode tournament
```

### 4. **Run New Architecture (Development)**
```bash
# CLI interface
python -m src.main forecast --tournament 32813 --max-questions 5 --dry-run

# Or direct execution
cd src && python main.py
```

## 🏗️ Architecture Reconstruction Guide

### Phase 1: Core Foundation
If rebuilding from scratch, implement in this order:

#### 1.1 Domain Layer (Start Here)
```python
# 1. Create basic entities
src/domain/entities/
├── question.py          # Question model with types
├── prediction.py        # Prediction with confidence
├── forecast.py          # Complete forecast
└── research_report.py   # Research findings

# 2. Create value objects
src/domain/value_objects/
├── confidence.py        # Confidence levels
├── probability.py       # Probability handling
└── reasoning_trace.py   # Reasoning documentation

# 3. Create repository interfaces
src/domain/repositories/
├── question_repository.py
└── forecast_repository.py
```

#### 1.2 Infrastructure Layer (External Dependencies)
```python
# 1. Basic API clients
src/infrastructure/external_apis/
├── llm_client.py           # LLM integration
├── search_client.py        # Search providers
└── metaculus_client.py     # Metaculus API

# 2. Configuration management
src/infrastructure/config/
├── settings.py             # Application settings
└── api_keys.py            # API key management

# 3. Reliability components
src/infrastructure/reliability/
├── circuit_breaker.py      # Fault tolerance
├── retry_manager.py        # Retry logic
└── rate_limiter.py        # Rate limiting
```

#### 1.3 Basic Agent System
```python
# 1. Base agent interface
src/agents/base_agent.py

# 2. Simple chain of thought agent
src/agents/chain_of_thought_agent.py

# 3. Basic prompts
src/prompts/
├── base_prompts.py
└── cot_prompts.py
```

### Phase 2: Advanced Features
Once core is working, add advanced features:

#### 2.1 Tournament Optimizations
```python
# Tournament-specific clients
src/infrastructure/external_apis/
├── tournament_asknews_client.py    # AskNews with quotas
└── metaculus_proxy_client.py       # Free credit models

# Tournament configuration
src/infrastructure/config/
└── tournament_config.py
```

#### 2.2 Advanced Reasoning
```python
# Additional agents
src/agents/
├── tree_of_thought_agent.py
├── react_agent.py
└── ensemble_agent.py

# Advanced prompts
src/prompts/
├── tot_prompts.py
└── react_prompts.py
```

#### 2.3 Domain Services
```python
# Core services (implement in order)
src/domain/services/
├── reasoning_orchestrator.py      # Multi-step reasoning
├── ensemble_service.py           # Agent coordination
├── research_service.py           # Research coordination
├── forecasting_service.py        # Forecasting coordination
└── performance_analyzer.py       # Performance tracking
```

### Phase 3: Production Features
Final phase for production deployment:

#### 3.1 Application Layer
```python
src/application/
├── tournament_orchestrator.py     # Main coordinator
├── forecast_service.py           # Forecasting service
└── dispatcher.py                 # Request routing
```

#### 3.2 Monitoring & Reliability
```python
src/infrastructure/
├── monitoring/metrics_service.py
├── logging/reasoning_logger.py
└── deployment/deployment_manager.py
```

## 🔧 Key Implementation Patterns

### 1. **Dependency Injection Pattern**
```python
# Component Registry Pattern
class ComponentRegistry:
    def __init__(self):
        self.components = {}

    def register(self, name: str, component: Any):
        self.components[name] = component

    def get(self, name: str) -> Any:
        return self.components.get(name)

# Usage in TournamentOrchestrator
class TournamentOrchestrator:
    def __init__(self):
        self.registry = ComponentRegistry()
        self._initialize_components()

    def _initialize_components(self):
        # Infrastructure first
        llm_client = LLMClient(config.llm)
        search_client = SearchClient(config.search)

        # Domain services with dependencies
        research_service = ResearchService(
            search_client=search_client,
            llm_client=llm_client
        )

        # Register all components
        self.registry.register("llm_client", llm_client)
        self.registry.register("research_service", research_service)
```

### 2. **Circuit Breaker Pattern**
```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError()

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

### 3. **Reasoning Trace Pattern**
```python
@dataclass
class ReasoningStep:
    step_type: ReasoningStepType
    content: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class ReasoningTrace:
    question_id: UUID
    agent_id: str
    reasoning_method: str
    steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float

    def add_step(self, step: ReasoningStep):
        self.steps.append(step)

    def get_reasoning_quality_score(self) -> float:
        # Calculate quality based on step consistency, confidence, etc.
        return sum(step.confidence for step in self.steps) / len(self.steps)
```

## 🎯 Tournament-Specific Implementation

### 1. **Resource Management**
```python
class TournamentAskNewsClient:
    def __init__(self):
        self.quota_limit = 9000
        self.usage_stats = AskNewsUsageStats()
        self.fallback_providers = ["perplexity", "exa", "openrouter"]

    async def get_news_research(self, question: str) -> str:
        if self._should_use_asknews():
            try:
                return await self._call_asknews(question)
            except QuotaExhaustedError:
                return await self._call_fallback_providers(question)
        else:
            return await self._call_fallback_providers(question)
```

### 2. **Proxy Model Integration**
```python
class MetaculusProxyClient:
    def __init__(self):
        self.proxy_models = {
            "default": "metaculus/claude-3-5-sonnet",
            "summarizer": "metaculus/gpt-4o-mini"
        }
        self.fallback_models = {
            "default": "openrouter/anthropic/claude-3-5-sonnet",
            "summarizer": "openai/gpt-4o-mini"
        }

    def get_llm_client(self, model_type: str = "default") -> LLMClient:
        if self.proxy_credits_enabled and not self.proxy_exhausted:
            try:
                return self._create_proxy_client(model_type)
            except ProxyUnavailableError:
                return self._create_fallback_client(model_type)
        else:
            return self._create_fallback_client(model_type)
```
## 🧪 Testing Strategy

### 1. **Unit Tests Structure**
```
tests/unit/
├── domain/
│   ├── test_entities.py         # Test domain entities
│   ├── test_value_objects.py    # Test value objects
│   └── services/
│       ├── test_reasoning_orchestrator.py
│       ├── test_ensemble_service.py
│       └── test_tournament_analyzer.py
├── agents/
│   ├── test_base_agent.py
│   ├── test_chain_of_thought_agent.py
│   └── test_ensemble_agent.py
├── infrastructure/
│   ├── test_llm_client.py
│   ├── test_circuit_breaker.py
│   └── test_tournament_asknews_client.py
└── prompts/
    ├── test_base_prompts.py
    └── test_cot_prompts.py
```

### 2. **Integration Tests**
```python
# Example integration test
class TestTournamentIntegration:
    async def test_full_tournament_flow(self):
        # Setup
        orchestrator = TournamentOrchestrator()
        question = create_test_question()

        # Execute
        forecast = await orchestrator.forecast_question(question)

        # Verify
        assert forecast.final_prediction is not None
        assert forecast.confidence_score > 0.5
        assert len(forecast.research_reports) > 0
```

### 3. **Tournament Simulation Tests**
```python
# Tournament-specific tests
class TestTournamentSimulation:
    def test_resource_quota_management(self):
        client = TournamentAskNewsClient()
        # Simulate quota exhaustion
        client.usage_stats.estimated_quota_used = 9000
        assert not client._should_use_asknews()

    def test_proxy_fallback_mechanism(self):
        proxy_client = MetaculusProxyClient()
        proxy_client.proxy_exhausted = True
        llm_client = proxy_client.get_llm_client()
        assert "openrouter" in llm_client.model
```

## 🔄 Configuration Management

### 1. **Environment Configuration**
```python
# .env.template structure
METACULUS_TOKEN=your_token_here
OPENROUTER_API_KEY=your_key_here

# Tournament Configuration
AIB_TOURNAMENT_ID=32813
TOURNAMENT_MODE=false
DRY_RUN=true
PUBLISH_REPORTS=true

# Resource Management
ASKNEWS_QUOTA_LIMIT=9000
ENABLE_PROXY_CREDITS=true
METACULUS_DEFAULT_MODEL=metaculus/claude-3-5-sonnet
```

### 2. **YAML Configuration**
```yaml
# config/config.production.yaml
tournament:
  id: 32813
  mode: "tournament"
  max_concurrent_questions: 5
  scheduling_interval_hours: 2

llm:
  provider: "openrouter"
  model: "metaculus/claude-3-5-sonnet"
  temperature: 0.3
  fallback_models:
    - "openrouter/anthropic/claude-3-5-sonnet"
    - "openai/gpt-4o-mini"

search:
  provider: "multi_source"
  asknews_quota_limit: 9000
  fallback_providers:
    - "perplexity"
    - "exa"
    - "openrouter"
```

### 3. **Hot-Reload Configuration**
```python
class ConfigManager:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.watchers = []
        self.callbacks = []

    def watch_config_changes(self):
        # File watcher implementation
        watcher = FileWatcher(self.config_path)
        watcher.on_change(self._handle_config_change)

    def _handle_config_change(self, event):
        try:
            new_config = self._load_config()
            self._validate_config(new_config)
            self._notify_callbacks(new_config)
        except Exception as e:
            logger.error(f"Config reload failed: {e}")
```

## 🚀 Deployment Guide

### 1. **Local Development**
```bash
# Setup development environment
poetry install
poetry shell

# Run tests
pytest tests/unit/
pytest tests/integration/

# Run tournament simulation
python -m tests.tournament.test_tournament_simulation

# Start development server
python src/main.py forecast --dry-run
```

### 2. **Production Deployment**
```bash
# Build Docker image
docker build -t ai-forecasting-bot:latest .

# Deploy with blue-green strategy
./scripts/blue-green-deploy.sh latest

# Monitor deployment
./scripts/health-check.sh
docker-compose logs -f forecasting-bot
```

### 3. **GitHub Actions Setup**
```yaml
# .github/workflows/run_bot_on_tournament.yaml
name: Tournament Forecasting
on:
  schedule:
    - cron: '*/30 * * * *'  # Every 30 minutes
  workflow_dispatch:

jobs:
  tournament_forecast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: poetry install
      - name: Run tournament bot
        env:
          METACULUS_TOKEN: ${{ secrets.METACULUS_TOKEN }}
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: poetry run python main.py --mode tournament
```

## 🔍 Debugging & Troubleshooting

### 1. **Common Issues**
```python
# Issue: Agent not producing predictions
# Debug: Check reasoning trace
def debug_agent_reasoning(agent, question):
    trace = await agent.reason(question, {})
    print(f"Steps: {len(trace.steps)}")
    print(f"Confidence: {trace.overall_confidence}")
    for step in trace.steps:
        print(f"{step.step_type}: {step.confidence}")

# Issue: Resource quota exhausted
# Debug: Check usage stats
def debug_resource_usage():
    client = TournamentAskNewsClient()
    stats = client.get_usage_stats()
    print(f"Quota used: {stats['quota_usage_percentage']:.1f}%")
    print(f"Fallback rate: {stats['fallback_rate']:.1f}%")
```

### 2. **Logging Configuration**
```python
# Structured logging setup
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

### 3. **Performance Monitoring**
```python
# Performance tracking
class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)

    @contextmanager
    def track_time(self, operation: str):
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.metrics[operation].append(duration)

    def get_stats(self, operation: str) -> Dict[str, float]:
        times = self.metrics[operation]
        return {
            "count": len(times),
            "avg": statistics.mean(times),
            "median": statistics.median(times),
            "max": max(times),
            "min": min(times)
        }
```

## 🎯 Extension Points

### 1. **Adding New Agents**
```python
# Create new agent by extending BaseAgent
class CustomReasoningAgent(BaseAgent):
    async def conduct_research(self, question, search_config=None):
        # Implement custom research logic
        pass

    async def generate_prediction(self, question, research_report):
        # Implement custom prediction logic
        pass

# Register in component registry
registry.register("custom_agent", CustomReasoningAgent("custom", config))
```

### 2. **Adding New Domain Services**
```python
# Create new domain service
class CustomAnalysisService:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def analyze_question(self, question: Question) -> AnalysisResult:
        # Implement custom analysis logic
        pass

# Integrate with dependency injection
def _initialize_domain_services(self):
    custom_service = CustomAnalysisService(
        llm_client=self.registry.get("llm_client")
    )
    self.registry.register("custom_analysis", custom_service)
```

### 3. **Adding New API Integrations**
```python
# Create new API client
class CustomAPIClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.circuit_breaker = CircuitBreaker()

    async def fetch_data(self, query: str) -> Dict[str, Any]:
        return await self.circuit_breaker.call(
            self._make_request, query
        )

# Register in infrastructure layer
registry.register("custom_api", CustomAPIClient(config.custom_api_key, config.custom_base_url))
```

This implementation guide provides everything needed to understand, reconstruct, extend, or debug the system from scratch.
