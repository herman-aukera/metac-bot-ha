# Implementation Plan - Tournament API Optimization and Budget Management

## Overview

Convert the tournament API optimization design into actionable coding tasks that address Ben's feedback about budget management, scheduling frequency, and API key configuration. Each task focuses on specific code modifications to optimize the bot for the $100 budget and tournament scope.

## Implementation Tasks

- [x] 1. API Key Configuration and Budget Management Setup
  - Update environment variable configuration to use the provided OpenRouter API key
  - Implement budget tracking and cost estimation for token usage
  - Create BudgetManager class for real-time cost monitoring
  - Add budget status reporting and alerting mechanisms
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Smart Model Selection and Cost Optimization
  - [x] 2.1 Implement task complexity analyzer for intelligent model selection
    - Create complexity assessment logic based on question type and length
    - Implement model selection strategy (GPT-4o-mini for research, GPT-4o for forecasts)
    - Add cost-per-task estimation and tracking
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 2.2 Create token counting and cost tracking system
    - Implement TokenTracker class for accurate token counting
    - Add real-time cost calculation for each API call
    - Create cost logging and budget utilization monitoring
    - _Requirements: 1.2, 4.4, 6.1_

  - [x] 2.3 Implement budget-aware operation modes
    - Create normal, conservative, and emergency operation modes
    - Add automatic mode switching based on budget utilization
    - Implement graceful degradation when approaching budget limits
    - _Requirements: 1.4, 4.5_

- [x] 3. Tournament Scheduling Optimization
  - [x] 3.1 Update GitHub Actions scheduling frequency
    - Change cron schedule from every 30 minutes to every 4 hours
    - Add configurable scheduling through environment variables
    - Implement deadline-aware scheduling for critical periods
    - _Requirements: 2.1, 2.2, 2.4_

  - [x] 3.2 Implement tournament scope management
    - Update question volume expectations from daily to seasonal
    - Add tournament duration and question count estimation
    - Create sustainable forecasting rate calculations
    - _Requirements: 2.3, 2.5_

- [x] 4. Enhanced Prompt Engineering for Budget Efficiency
  - [x] 4.1 Create optimized research prompt templates
    - Implement structured research prompts with clear output format
    - Add source citation requirements and factual focus
    - Create token-efficient prompt templates for different question types
    - _Requirements: 3.1, 3.3_

  - [x] 4.2 Develop calibrated forecasting prompt templates
    - Create forecasting prompts with explicit calibration instructions
    - Add scenario analysis and uncertainty assessment components
    - Implement overconfidence reduction techniques in prompts
    - _Requirements: 3.2, 5.2_

  - [x] 4.3 Implement adaptive research depth based on question complexity
    - Create research depth assessment logic
    - Add adaptive AskNews usage for 48-hour news windows
    - Implement research quality validation and gap detection
    - _Requirements: 3.4, 3.5_

- [x] 5. Tournament Compliance and Performance Optimization
  - [x] 5.1 Ensure reasoning comment publication compliance
    - Verify publish_reports_to_metaculus is set to True
    - Add reasoning comment validation and formatting
    - Implement tournament transparency requirements
    - _Requirements: 5.1_

  - [x] 5.2 Implement calibration and overconfidence mitigation
    - Add calibration adjustment logic to avoid extreme predictions
    - Implement community prediction anchoring strategies
    - Create log scoring optimization techniques
    - _Requirements: 5.2, 5.3_

  - [x] 5.3 Add tournament rule compliance validation
    - Ensure no human intervention in forecasting loop
    - Validate automated decision-making processes
    - Add compliance monitoring and reporting
    - _Requirements: 5.4, 5.5_

- [x] 6. Monitoring and Performance Tracking System
  - [x] 6.1 Implement comprehensive cost and usage monitoring
    - Create real-time budget utilization dashboard
    - Add token usage and API cost tracking per question
    - Implement budget alert system for threshold breaches
    - _Requirements: 6.1, 6.4_

  - [x] 6.2 Develop performance and accuracy tracking
    - Add forecast accuracy and calibration metrics monitoring
    - Im API success rate and fallback usage tracking
    - Create performance degradation detection and alerting
    - _Requirements: 6.2, 6.3, 6.5_

- [x] 7. Main Application Integration and Configuration
  - [x] 7.1 Update main.py with optimized configuration
    - Integrate BudgetManager and TokenTracker into TemplateForecaster
    - Update LLM configuration to use OpenRouter API key
    - Add cost-aware model selection logic
    - _Requirements: 1.1, 4.1, 4.2_

  - [x] 7.2 Implement environment variable configuration updates
    - Add BUDGET_LIMIT, OPENROUTER_API_KEY, and cost tracking variables
    - Update tournament configuration for seasonal scope
    - Add model selection and scheduling configuration options
    - _Requirements: 1.1, 2.1, 4.3_

  - [x] 7.3 Create enhanced error handling and fallback mechanisms
    - Implement budget exhaustion handling with graceful degradation
    - Add API failure handling with intelligent fallbacks
    - Create emergency mode operation for critical situations
    - _Requirements: 1.4, 1.5_

- [x] 8. GitHub Actions and Deployment Optimization
  - [x] 8.1 Update GitHub Actions workflow configuration
    - Change scheduling frequency from 30 minutes to 4 hours
    - Add budget monitoring and alerting to workflow
    - Update environment variable configuration for new API key
    - _Requirements: 2.1, 6.4_

  - [x] 8.2 Implement deployment monitoring and cost tracking
    - Add budget utilization reporting to GitHub Actions
    - Create cost per run tracking and alerting
    - Implement automatic workflow suspension on budget exhaustion
    - _Requirements: 6.1, 6.4_

- [x] 9. Testing and Validation
  - [x] 9.1 Create budget management and cost tracking tests
    - Implement unit tests for BudgetManager and TokenTracker
    - Add integration tests for cost-aware model selection
    - Create budget simulation tests for tournament duration
    - _Requirements: 1.2, 1.3, 4.4_

  - [x] 9.2 Develop prompt optimization and calibration tests
    - Test prompt efficiency and token usage optimization
    - Validate calibration improvement and overconfidence reduction
    - Create A/B testing framework for prompt performance
    - _Requirements: 3.1, 3.2, 5.2_

  - [x] 9.3 Implement tournament simulation and compliance testing
    - Create dry-run mode testing for complete tournament workflow
    - Test scheduling optimization and question volume management
    - Validate tournament rule compliance and transparency requirements
    - _Requirements: 2.3, 5.1, 5.4_

- [x] 10. Documentation and Configuration Updates
  - [x] 10.1 Update environment configuration templates
    - Update .env.template with new API key and budget variables
    - Add documentation for budget management configuration
    - Create setup guide for tournament optimization
    - _Requirements: 1.1, 6.1_

  - [x] 10.2 Create budget management and monitoring documentation
    - Document cost tracking and budget management features
    - Add troubleshooting guide for budget-related issues
    - Create performance optimization recommendations
    - _Requirements: 6.1, 6.2, 6.5_

## Implementation Notes

- Each task addresses specific issues identified in Ben's feedback
- Budget management is prioritized to ensure $100 limit compliance
- Scheduling optimization reduces frequency to match tournament scope
- All tasks maintain tournament competitiveness while optimizing costs
- Testing ensures reliability and compliance with tournament rules

## Success Criteria

Upon completion of all tasks, the system will:

- Operate efficiently within the $100 budget for the entire tournament
- Use the provided OpenRouter API key as the primary LLM provider
- Schedule forecasting runs every 4 hours instead of every 30 minutes
- Implement smart model selection to optimize cost-performance ratio
- Maintain competitive forecasting accuracy with proper calibration
- Provide comprehensive monitoring of costs, usage, and performance
- Ensure full compliance with tournament rules and transparency requirements
