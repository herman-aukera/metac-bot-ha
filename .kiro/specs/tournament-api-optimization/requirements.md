# Requirements Document - Tournament API Optimization and Budget Management

## Introduction

This document outlines the requirements for optimizing the Metaculus AI Forecasting Bot based on feedback from Ben at Metaculus regarding API usage, budget management, and tournament scheduling. The bot needs to be reconfigured to work within a $100 budget using the provided OpenRouter API key while maintaining competitive performance in the Fall 2025 AI Benchmark Tournament.

## Requirements

### Requirement 1: API Key Configuration and Budget Management

**User Story:** As a tournament participant, I want to use the provided $100 OpenRouter API credit efficiently, so that I can maximize forecasting quality while staying within budget constraints.

#### Acceptance Criteria

1. WHEN configuring API keys THEN the system SHALL use the provided OpenRouter API key `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` as the primary LLM provider
2. WHEN estimating costs THEN the system SHALL implement token counting and cost tracking to stay within the $100 budget limit
3. WHEN selecting models THEN the system SHALL use GPT-4o for critical forecasting decisions and GPT-4o-mini for research summaries to optimize cost-performance ratio
4. WHEN approaching budget limits THEN the system SHALL implement smart throttling to prevent overspending
5. IF the Metaculus proxy is available THEN the system SHALL use it as a fallback, but prioritize the funded OpenRouter key

### Requirement 2: Tournament Scheduling Optimization

**User Story:** As a tournament strategist, I want optimized scheduling that aligns with tournament scope and resource constraints, so that I participate effectively without wasting resources on over-forecasting.

#### Acceptance Criteria

1. WHEN running in tournament mode THEN the system SHALL reduce GitHub Actions scheduling from every 30 minutes to every 2-4 hours for sustainable operation
2. WHEN targeting tournaments THEN the system SHALL focus on seasonal AIB (Fall 2025 - ID: 32813) and MiniBench tournaments only
3. WHEN estimating question volume THEN the system SHALL plan for 50-100 questions total for the tournament duration, not 50-100 questions per day
4. WHEN near question deadlines THEN the system SHALL implement smart timing for final submissions to optimize log scoring
5. IF tournament dynamics require more frequent updates THEN the system SHALL support configurable scheduling through environment variables

### Requirement 3: Enhanced Prompt Engineering for Budget Efficiency

**User Story:** As a forecasting system, I want optimized prompts that maximize accuracy per token spent, so that I achieve competitive performance within budget constraints.

#### Acceptance Criteria

1. WHEN conducting research THEN the system SHALL use structured prompts that request concise, factual summaries with source citations
2. WHEN generating forecasts THEN the system SHALL use calibrated prompting that includes uncertainty assessment and scenario analysis
3. WHEN processing questions THEN the system SHALL implement token-efficient prompt templates that avoid redundant information
4. WHEN using AskNews THEN the system SHALL focus on 48-hour news windows for recent developments to maximize relevance
5. IF research quality is insufficient THEN the system SHALL implement adaptive research depth based on question complexity

### Requirement 4: Smart Resource Allocation and Model Selection

**User Story:** As a cost-conscious forecasting system, I want intelligent model selection based on task complexity, so that I use expensive models only when necessary for accuracy gains.

#### Acceptance Criteria

1. WHEN performing research tasks THEN the system SHALL use GPT-4o-mini for information gathering and summarization
2. WHEN making final forecasts THEN the system SHALL use GPT-4o for critical prediction decisions that require higher reasoning capability
3. WHEN processing simple questions THEN the system SHALL implement task complexity assessment to select appropriate models
4. WHEN tracking usage THEN the system SHALL log token consumption and estimated costs per question
5. IF budget utilization exceeds 80% THEN the system SHALL switch to more conservative model usage patterns

### Requirement 5: Tournament Compliance and Performance Optimization

**User Story:** As a tournament participant, I want to ensure full compliance with tournament rules while optimizing for log scoring performance, so that I achieve competitive rankings within the tournament framework.

#### Acceptance Criteria

1. WHEN submitting forecasts THEN the system SHALL publish reasoning comments to meet tournament transparency requirements
2. WHEN calibrating predictions THEN the system SHALL avoid overconfidence to minimize log scoring penalties
3. WHEN considering community predictions THEN the system SHALL implement smart anchoring strategies for score preservation
4. WHEN processing tournament questions THEN the system SHALL ensure no human intervention in the forecasting loop
5. IF forecast confidence is low THEN the system SHALL implement conservative strategies to avoid large scoring penalties

### Requirement 6: Monitoring and Performance Tracking

**User Story:** As a tournament operator, I want comprehensive monitoring of API usage, costs, and performance metrics, so that I can optimize strategy and ensure sustainable operation.

#### Acceptance Criteria

1. WHEN running forecasts THEN the system SHALL track and log token usage, API costs, and budget utilization
2. WHEN completing forecasts THEN the system SHALL monitor forecast accuracy and calibration metrics
3. WHEN using multiple APIs THEN the system SHALL track success rates and fallback usage patterns
4. WHEN approaching limits THEN the system SHALL generate alerts for budget, rate limits, and performance thresholds
5. IF performance degrades THEN the system SHALL provide actionable insights for strategy adjustment

## Success Criteria

- Operate within $100 budget for the entire tournament duration
- Maintain competitive forecasting accuracy with optimized resource usage
- Achieve proper calibration to minimize log scoring penalties
- Successfully forecast on 50-100 tournament questions with high-quality reasoning
- Demonstrate sustainable operation with appropriate scheduling frequency
