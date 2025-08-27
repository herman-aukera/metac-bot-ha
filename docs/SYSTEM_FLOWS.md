# 🔄 System Flows & Dependencies

## 📋 Overview

This document describes the key workflows, data flows, and dependencies in the Metaculus AI Forecasting Bot system.

## 🏆 Tournament Execution Flow (Production)

### Main Tournament Flow (`main.py`)
```mermaid
graph TD
    A[GitHub Actions Trigger] --> B[main.py TemplateForecaster]
    B --> C[Tournament Question Fetching]
    C --> D[Research Phase]
    D --> E[TournamentAskNewsClient]
    E --> F{AskNews Available?}
    F -->|Yes| G[AskNews Research]
    F -->|No| H[Fallback Providers]
    H --> I[Perplexity/Exa/OpenRouter]
    G --> J[Research Complete]
    I --> J
    J --> K[Prediction Phase]
    K --> L[MetaculusProxyClient]
    L --> M{Proxy Credits Available?}
    M -->|Yes| N[Free Metaculus Models]
    M -->|No| O[OpenRouter Fallback]
    N --> P[Generate Prediction]
    O --> P
    P --> Q[Submit to Metaculus]
    Q --> R[Log Results]
```

### Resource Management Flow
```mermaid
graph LR
    A[Request] --> B{AskNews Quota Check}
    B -->|Available| C[Use AskNews]
    B -->|Exhausted| D[Use Perplexity]
    D --> E{Perplexity Available?}
    E -->|Yes| F[Perplexity Research]
    E -->|No| G[Use Exa]
    G --> H{Exa Available?}
    H -->|Yes| I[Exa Research]
    H -->|No| J[OpenRouter Fallback]

    C --> K[Research Complete]
    F --> K
    I --> K
    J --> K
```

## 🏗️ New Architecture Flow (`src/main.py`)

### Complete Forecasting Pipeline
```mermaid
graph TD
    A[CLI/API Request] --> B[TournamentOrchestrator]
    B --> C[ComponentRegistry]
    C --> D[ForecastingPipeline]
    D --> E[Question Analysis]
    E --> F[Agent Selection]
    F --> G[Research Phase]
    G --> H[Multi-Agent Reasoning]
    H --> I[Ensemble Aggregation]
    I --> J[Confidence Calibration]
    J --> K[Submission Validation]
    K --> L[Result Logging]
```

### Dependency Injection Flow
```mermaid
graph TD
    A[TournamentOrchestrator] --> B[ComponentRegistry]
    B --> C[Infrastructure Layer]
    C --> D[LLMClient]
    C --> E[SearchClient]
    C --> F[MetaculusClient]
    B --> G[Domain Services]
    G --> H[ReasoningOrchestrator]
    G --> I[EnsembleService]
    G --> J[TournamentAnalyzer]
    B --> K[Application Services]
    K --> L[ForecastService]
    K --> M[Dispatcher]
    L --> N[Agents]
    N --> O[ChainOfThoughtAgent]
    N --> P[TreeOfThoughtAgent]
    N --> Q[ReActAgent]
```

## 🧠 Reasoning Agent Workflows

### Chain of Thought Agent Flow
```mermaid
graph TD
    A[Question Input] --> B[Initial Observation]
    B --> C[Generate Hypotheses]
    C --> D[Systematic Analysis]
    D --> E[Synthesis of Findings]
    E --> F[Final Conclusion]
    F --> G[Bias Detection]
    G --> H[Confidence Calibration]
    H --> I[Reasoning Trace Output]
```

### Tree of Thought Agent Flow
```mermaid
graph TD
    A[Question Input] --> B[Generate Initial Thoughts]
    B --> C[Path 1]
    B --> D[Path 2]
    B --> E[Path 3]
    C --> F[Expand Path 1]
    D --> G[Expand Path 2]
    E --> H[Expand Path 3]
    F --> I[Evaluate Paths]
    G --> I
    H --> I
    I --> J[Select Best Paths]
    J --> K[Synthesize Results]
    K --> L[Final Prediction]
```

### Ensemble Coordination Flow
```mermaid
graph TD
    A[Question Input] --> B[Agent Selection]
    B --> C[Parallel Execution]
    C --> D[CoT Agent]
    C --> E[ToT Agent]
    C --> F[ReAct Agent]
    D --> G[Prediction 1]
    E --> H[Prediction 2]
    F --> I[Prediction 3]
    G --> J[Divergence Analysis]
    H --> J
    I --> J
    J --> K[Weight Adjustment]
    K --> L[Aggregation Method Selection]
    L --> M[Final Ensemble Prediction]
```

## 🔧 Configuration & Dependency Management

### Configuration Loading Flow
```mermaid
graph TD
    A[Application Start] --> B[Load Environment Variables]
    B --> C[Load YAML Config]
    C --> D[Merge Configurations]
    D --> E[Validate Settings]
    E --> F[Initialize Components]
    F --> G[Dependency Injection]
    G --> H[Health Checks]
    H --> I[System Ready]
```

### Hot-Reload Configuration Flow
```mermaid
graph TD
    A[Config File Change] --> B[File Watcher Trigger]
    B --> C[Validate New Config]
    C --> D{Valid?}
    D -->|Yes| E[Update Components]
    D -->|No| F[Rollback & Alert]
    E --> G[Notify Services]
    G --> H[Configuration Updated]
    F --> I[Keep Current Config]
```
## 🏗️ Component Dependencies

### Core Dependencies Map
```
TournamentOrchestrator
├── ComponentRegistry
│   ├── Infrastructure Components
│   │   ├── LLMClient
│   │   ├── SearchClient
│   │   ├── MetaculusClient
│   │   ├── TournamentAskNewsClient
│   │   └── MetaculusProxyClient
│   │
│   ├── Domain Services (24 services)
│   │   ├── ReasoningOrchestrator
│   │   ├── EnsembleService
│   │   ├── TournamentAnalyzer
│   │   ├── ConflictResolver
│   │   ├── PerformanceAnalyzer
│   │   └── ... (19 more services)
│   │
│   ├── Application Services
│   │   ├── ForecastService
│   │   ├── Dispatcher
│   │   └── IngestionService
│   │
│   └── Agents
│       ├── ChainOfThoughtAgent
│       ├── TreeOfThoughtAgent
│       ├── ReActAgent
│       └── EnsembleAgent
│
└── ForecastingPipeline
    └── All above components via dependency injection
```

### Service Dependency Graph
```
ReasoningOrchestrator
├── Depends on: LLMClient
├── Used by: All Agents

EnsembleService
├── Depends on: All Agents, PerformanceAnalyzer
├── Used by: ForecastingPipeline

TournamentAnalyzer
├── Depends on: MetaculusClient, PerformanceAnalyzer
├── Used by: TournamentOrchestrator

ConflictResolver
├── Depends on: LLMClient
├── Used by: ResearchService

PerformanceAnalyzer
├── Depends on: Database, MetricsService
├── Used by: EnsembleService, TournamentAnalyzer

UncertaintyQuantifier
├── Depends on: CalibrationService
├── Used by: All Agents
```

## 📊 Data Flow Patterns

### Research Data Flow
```mermaid
graph LR
    A[Question] --> B[Research Service]
    B --> C[Search Providers]
    C --> D[Raw Results]
    D --> E[Source Evaluation]
    E --> F[Conflict Resolution]
    F --> G[Knowledge Gap Detection]
    G --> H[Research Report]
```

### Prediction Data Flow
```mermaid
graph LR
    A[Research Report] --> B[Agent Selection]
    B --> C[Reasoning Process]
    C --> D[Bias Detection]
    D --> E[Confidence Calibration]
    E --> F[Prediction]
    F --> G[Ensemble Aggregation]
    G --> H[Final Forecast]
```

### Performance Feedback Loop
```mermaid
graph LR
    A[Prediction Submitted] --> B[Wait for Resolution]
    B --> C[Outcome Known]
    C --> D[Performance Analysis]
    D --> E[Pattern Detection]
    E --> F[Strategy Adaptation]
    F --> G[Weight Adjustment]
    G --> H[Improved Future Predictions]
```

## 🔄 Error Handling & Recovery Flows

### Circuit Breaker Pattern
```mermaid
graph TD
    A[API Request] --> B{Circuit State?}
    B -->|Closed| C[Execute Request]
    B -->|Open| D[Return Cached/Default]
    B -->|Half-Open| E[Test Request]
    C --> F{Success?}
    F -->|Yes| G[Reset Failure Count]
    F -->|No| H[Increment Failure Count]
    H --> I{Threshold Exceeded?}
    I -->|Yes| J[Open Circuit]
    I -->|No| K[Continue]
    E --> L{Test Success?}
    L -->|Yes| M[Close Circuit]
    L -->|No| N[Keep Open]
```

### Retry Strategy Flow
```mermaid
graph TD
    A[Request Fails] --> B{Retryable Error?}
    B -->|No| C[Fail Immediately]
    B -->|Yes| D{Attempts < Max?}
    D -->|No| E[Fail After Max Attempts]
    D -->|Yes| F[Calculate Backoff]
    F --> G[Wait]
    G --> H[Retry Request]
    H --> I{Success?}
    I -->|Yes| J[Return Success]
    I -->|No| K[Increment Attempt]
    K --> D
```

### Graceful Degradation Flow
```mermaid
graph TD
    A[Service Failure] --> B[Detect Failure]
    B --> C{Critical Service?}
    C -->|Yes| D[Use Fallback]
    C -->|No| E[Continue Without]
    D --> F[Reduced Functionality]
    E --> G[Log Warning]
    F --> H[Monitor Recovery]
    G --> H
    H --> I{Service Recovered?}
    I -->|Yes| J[Restore Full Function]
    I -->|No| K[Continue Degraded]
```

## 🚀 Deployment & Monitoring Flows

### Blue-Green Deployment Flow
```mermaid
graph TD
    A[New Version Ready] --> B[Deploy to Green]
    B --> C[Health Check Green]
    C --> D{Green Healthy?}
    D -->|Yes| E[Switch Traffic to Green]
    D -->|No| F[Rollback to Blue]
    E --> G[Monitor Green]
    G --> H{Performance OK?}
    H -->|Yes| I[Deployment Success]
    H -->|No| J[Emergency Rollback]
    F --> K[Fix Issues]
    J --> L[Investigate Issues]
```

### Health Monitoring Flow
```mermaid
graph TD
    A[Health Check Timer] --> B[Check All Components]
    B --> C[Database Health]
    B --> D[API Health]
    B --> E[Service Health]
    C --> F[Aggregate Results]
    D --> F
    E --> F
    F --> G{All Healthy?}
    G -->|Yes| H[Report Healthy]
    G -->|No| I[Identify Issues]
    I --> J[Send Alerts]
    J --> K[Trigger Recovery]
```

## 🎯 Tournament-Specific Flows

### Question Prioritization Flow
```mermaid
graph TD
    A[New Questions Available] --> B[Fetch Questions]
    B --> C[Category Analysis]
    C --> D[Priority Scoring]
    D --> E[Resource Allocation]
    E --> F[Deadline Assessment]
    F --> G[Competition Analysis]
    G --> H[Final Priority Queue]
    H --> I[Execute Forecasts]
```

### Competitive Intelligence Flow
```mermaid
graph TD
    A[Tournament Data] --> B[Performance Analysis]
    B --> C[Ranking Trends]
    C --> D[Category Performance]
    D --> E[Market Inefficiencies]
    E --> F[Strategy Opportunities]
    F --> G[Adaptation Recommendations]
    G --> H[Strategy Updates]
```

This comprehensive flow documentation provides the foundation for understanding how all components interact and depend on each other in the system.
