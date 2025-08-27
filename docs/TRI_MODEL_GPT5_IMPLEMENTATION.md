# Tri-Model GPT-5 Implementation Guide

## Overview

This document describes the implementation of the strategic tri-model GPT-5 system with anti-slop directives for the Metaculus AI Benchmark Tournament. The system implements a cost-performance triangle that maximizes forecasting quality per dollar spent.

## Strategic Cost-Performance Triangle

### GPT-5 Nano ($0.05/1M tokens)
- **Use Cases**: Ultra-fast validation, parsing, simple summaries
- **Tasks**: Content validation, basic fact-checking, simple responses
- **Characteristics**: Deterministic (temp=0.1), fast (30s timeout), minimal reasoning

### GPT-5 Mini ($0.25/1M tokens)
- **Use Cases**: Balanced speed/intelligence for research synthesis, intermediate reasoning
- **Tasks**: Research gathering, news summarization, moderate complexity analysis
- **Characteristics**: Moderate creativity (temp=0.3), balanced (60s timeout), good reasoning

### GPT-5 Full ($1.50/1M tokens)
- **Use Cases**: Maximum reasoning power for final forecasting decisions, complex analysis
- **Tasks**: Final predictions, complex scenario analysis, critical decisions
- **Characteristics**: Precise (temp=0.0), patient (90s timeout), maximum reasoning

## Budget Impact Analysis

**Traditional Single-Model Approach (GPT-5 Full only):**
- Cost per question: ~$0.33
- Questions possible with $100: ~300
- Risk: High cost, limited question coverage

**Tri-Model Strategic Approach:**
- Research (Mini): ~$0.02 per question
- Validation (Nano): ~$0.001 per question
- Forecasting (Full): ~$0.03 per question
- **Total per question: ~$0.05**
- **Questions possible with $100: ~2000**
- **Cost reduction: 70% vs single-model**

## Anti-Slop Quality Guard System

### Core Directives
All prompts include these quality guards:
- Think step-by-step internally, output only final reasoning
- Ground every claim with specific evidence sources
- Acknowledge uncertainty explicitly when present
- Use structured bullet points for clarity
- Maintain response ≤ 300 words unless complex analysis required
- Pre-check: Does every statement trace to verifiable evidence?

### Task-Specific Anti-Slop Rules

**Research Tasks:**
- Cite every factual claim with sources (URLs, dates, publications)
- Acknowledge information gaps explicitly
- Prioritize recent developments (48-hour windows)
- Synthesize without speculation or unsupported interpretations

**Forecasting Tasks:**
- Base predictions on verifiable evidence and historical precedents
- Acknowledge uncertainty with appropriate confidence bounds
- Consider multiple scenarios and weight by probability
- Avoid overconfidence penalties through careful calibration

**Validation Tasks:**
- Focus on factual accuracy and source verification
- Flag potential hallucinations or unsupported claims
- Maintain deterministic, concise responses

## Implementation Architecture

### Tri-Model Router (`src/infrastructure/config/tri_model_router.py`)

```python
class TriModelRouter:
    def choose_model(self, task_type, complexity, content_length, budget_remaining):
        """Strategic model selection based on task requirements and budget."""

    async def route_query(self, task_type, content, complexity, budget_remaining):
        """Route query to optimal model with anti-slop directives."""
```

**Key Features:**
- Automatic model selection based on task type and complexity
- Budget-aware routing with emergency mode fallbacks
- Anti-slop directive injection for all prompts
- Quality validation of responses
- Cost estimation and tracking

### Anti-Slop Prompts (`src/prompts/anti_slop_prompts.py`)

```python
class AntiSlopPrompts:
    def get_research_prompt(self, question_text, model_tier):
        """Research prompt with tier-specific anti-slop directives."""

    def get_binary_forecast_prompt(self, question_text, background_info, ...):
        """Binary forecasting with calibration and anti-slop guards."""
```

**Prompt Templates:**
- Research prompts with source citation requirements
- Binary forecasting with scenario analysis and calibration
- Multiple choice with probability distribution guidance
- Numeric forecasting with uncertainty quantification
- Validation prompts for quality checking

### Main Integration (`main.py`)

**Enhanced TemplateForecaster:**
- Tri-model router initialization and status checking
- Budget-aware research with intelligent model selection
- Anti-slop forecasting for all question types (binary, multiple choice, numeric)
- Legacy fallback methods for compatibility
- Comprehensive error handling and graceful degradation

## Budget-Aware Operation Modes

### Normal Mode (Budget > 50%)
- Use optimal model for each task type
- Full feature set enabled
- Standard anti-slop directives

### Conservative Mode (Budget 20-50%)
- Prefer cheaper models when possible
- Downgrade full→mini for non-critical tasks
- Enhanced budget monitoring

### Emergency Mode (Budget < 20%)
- Force nano model for all tasks
- Minimal processing to preserve budget
- Critical questions only

## Quality Assurance Features

### Multi-Stage Validation
1. **Research Stage**: Mini model gathers information with source citations
2. **Validation Stage**: Nano model checks for accuracy and hallucinations
3. **Forecasting Stage**: Full model makes final prediction with calibration

### Response Quality Checks
- Evidence traceability verification
- Uncertainty acknowledgment validation
- Length compliance monitoring
- Citation requirement enforcement

### Tournament Compliance
- Maintains automated operation (no human intervention)
- Ensures reasoning comment publication
- Provides transparent cost tracking
- Supports code inspection requirements

## Performance Optimization

### Token Usage Optimization
- Structured, concise prompts minimize input tokens
- Template-based approach reduces redundancy
- Smart context pruning for long content
- Efficient output formatting requirements

### Context Management
- 48-hour news windows for research efficiency
- Summarization for long documents
- Smart content filtering and prioritization
- Caching of frequently used information

### Scheduling Integration
- Works with 4-hour scheduling frequency
- Deadline-aware model selection
- Critical period optimization
- Emergency mode compatibility

## Testing and Validation

### Test Script (`scripts/test_tri_model_integration.py`)
Comprehensive testing including:
- Model routing validation
- Anti-slop directive verification
- Budget-aware behavior testing
- Complete workflow simulation
- Cost estimation accuracy

### Key Test Scenarios
- Normal operation with full budget
- Conservative mode with limited budget
- Emergency mode with critical budget
- Model fallback and error handling
- Quality validation and anti-slop compliance

## Usage Examples

### Basic Research Task
```python
# Route research query to mini model with anti-slop directives
research = await tri_model_router.route_query(
    task_type="research",
    content=anti_slop_prompts.get_research_prompt(question_text, "mini"),
    complexity="medium",
    budget_remaining=75.0
)
```

### Binary Forecasting
```python
# Route forecast to full model with calibration directives
forecast = await tri_model_router.route_query(
    task_type="forecast",
    content=anti_slop_prompts.get_binary_forecast_prompt(
        question_text=question.question_text,
        background_info=question.background_info,
        research=research_results,
        model_tier="full"
    ),
    complexity="high",
    budget_remaining=budget_remaining
)
```

### Validation Check
```python
# Validate response with nano model
validation = await tri_model_router.route_query(
    task_type="validation",
    content=anti_slop_prompts.get_validation_prompt(response, "research"),
    complexity="minimal",
    budget_remaining=budget_remaining
)
```

## Configuration

### Environment Variables
```bash
# GPT-5 Model Configuration
DEFAULT_MODEL=gpt-5                    # Full model for forecasting
MINI_MODEL=gpt-5-mini                  # Mini model for research
NANO_MODEL=gpt-5-nano                  # Nano model for validation

# Budget Management
BUDGET_LIMIT=100.0                     # Tournament budget limit
CONSERVATIVE_MODE_THRESHOLD=0.50       # Switch to conservative mode
EMERGENCY_MODE_THRESHOLD=0.20          # Switch to emergency mode
```

### Model Fallbacks
If GPT-5 models are not available, the system automatically falls back to:
- Full → `openai/gpt-4o`
- Mini → `openai/gpt-4o-mini`
- Nano → `openai/gpt-4o-mini`

## Monitoring and Alerts

### Cost Tracking
- Real-time budget utilization monitoring
- Cost per question breakdown by model tier
- Projected tournament total based on current usage
- Early warning for budget overrun risk

### Performance Metrics
- Model selection accuracy and efficiency
- Anti-slop compliance rates
- Quality validation success rates
- Tournament competitiveness vs. cost analysis

### Alert Thresholds
- 50% budget: First warning notification
- 75% budget: Conservative mode recommended
- 90% budget: Emergency mode preparation
- 95% budget: Emergency mode activation

## Competitive Advantages

### Cost Efficiency
- **70% cost reduction** vs single GPT-5 strategy
- **6x more questions** processable within budget
- Strategic resource allocation for maximum impact

### Quality Assurance
- **Multi-stage validation** prevents hallucinations
- **Anti-slop directives** ensure clean, cited reasoning
- **Tournament compliance** with automated operation

### Scalability
- **Budget-aware operation** prevents overspending
- **Graceful degradation** maintains functionality under constraints
- **Flexible routing** adapts to changing conditions

### Tournament Optimization
- **Calibrated predictions** optimized for log scoring
- **Source citation** for transparency requirements
- **Overconfidence reduction** prevents scoring penalties
- **Strategic question prioritization** maximizes competitive advantage

This tri-model implementation provides a sophisticated, cost-effective approach to tournament forecasting that maximizes both quality and budget efficiency while maintaining full compliance with tournament rules and requirements.
