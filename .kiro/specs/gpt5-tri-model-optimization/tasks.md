# Implementation Plan - OpenRouter Tri-Model Optimization with Anti-Slop Directives

## Overview

Convert the OpenRouter tri-model optimization design into actionable coding tasks that enhance the existing system with state-of-the-art prompt engineering, sophisticated anti-slop directives, and strategic model routing through OpenRouter's unified gateway. Each task builds incrementally to create a tournament-winning forecasting system that maximizes performance within the $100 budget constraint using actual available models.

## Implementation Tasks

- [x] 1. OpenRouter Configuration and Model Setup
  - Configure OpenRouter base URL (<https://openrouter.ai/api/v1>) and attribution headers
  - Update tri-model router to use actual OpenRouter models with correct pricing
  - Implement tier system: gpt-5 ($1.50), gpt-5-mini ($0.25), gpt-5-nano ($0.05)
  - Add free model fallbacks: gpt-oss-20b:free, kimi-k2:free for budget exhaustion
  - Create model availability detection and OpenRouter provider routing
  - _Requirements: 1.1, 1.2, 1.3, 8.1, 8.2_

- [x] 2. Advanced Anti-Slop Prompt Engineering System
  - [x] 2.1 Create enhanced base anti-slop directives with latest prompt engineering techniques
    - Implement Chain-of-Verification (CoVe) internal reasoning directives
    - Add evidence traceability pre-checks and source citation requirements
    - Create uncertainty acknowledgment and calibration instructions
    - Implement structured output formatting with bullet points and word limits
    - _Requirements: 2.1, 2.2, 2.4_

  - [x] 2.2 Develop tier-specific prompt optimizations for each OpenRouter model
    - Create gpt-5-nano prompts optimized for speed and essential validation
    - Develop gpt-5-mini prompts balanced for depth and efficiency in research synthesis
    - Design gpt-5 prompts for comprehensive analysis and maximum reasoning power
    - Implement dynamic prompt adaptation based on OpenRouter model capabilities
    - _Requirements: 5.5, 2.1_

  - [x] 2.3 Implement advanced forecasting prompt templates with calibration techniques
    - Create binary forecasting prompts with scenario analysis and base rate consideration
    - Develop multiple choice prompts with probability distribution guidance
    - Design numeric forecasting prompts with uncertainty quantification
    - Add overconfidence reduction and community prediction anchoring
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 3. Strategic Model Routing Enhancement
  - [x] 3.1 Implement advanced content analysis for optimal model selection
    - Create question complexity scoring algorithm based on multiple factors
    - Add content length analysis and token estimation
    - Implement domain-specific complexity assessment
    - Create urgency and priority scoring for task routing
    - _Requirements: 3.3, 3.4_

  - [x] 3.2 Develop budget-aware routing strategies with operation modes
    - Implement normal mode routing (0-70% budget utilization) with optimal GPT-5 model selection
    - Create conservative mode routing (70-85% budget utilization) with GPT-5 mini/nano preferred selections
    - Design emergency mode routing (85-95% budget utilization) with free models preferred
    - Implement critical mode routing (95-100% budget utilization) with free models only
    - Add OpenRouter provider preferences (sort: "price", max_price limits) for each mode
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 8.3, 8.5_

  - [x] 3.3 Create intelligent fallback and error recovery mechanisms
    - Implement model tier fallback chain with performance tracking
    - Add API provider fallback (OpenRouter → Metaculus Proxy)
    - Create retry logic with exponential backoff and circuit breakers
    - Implement graceful degradation for budget exhaustion scenarios
    - _Requirements: 1.2, 7.4_

- [x] 4. Multi-Stage Validation Pipeline Implementation
  - [x] 4.1 Create research stage with AskNews and gpt-5-mini synthesis
    - Prioritize AskNews API for research (free via METACULUSQ4 promo code)
    - Use gpt-5-mini for research synthesis and analysis with mandatory citations
    - Add 48-hour news focus and structured output formatting
    - Implement free models (gpt-oss-20b:free, kimi-k2:free) as backup when AskNews quota is tight
    - Create research quality validation and gap detection
    - _Requirements: 4.1, 4.2, 5.1_

  - [x] 4.2 Develop validation stage with gpt-5-nano for quality assurance
    - Create validation prompts optimized for gpt-5-nano capabilities
    - Implement evidence traceability verification and hallucination detection
    - Add logical consistency checking and quality scoring
    - Create automated quality issue identification and reporting
    - _Requirements: 4.3, 4.4_

  - [x] 4.3 Implement forecasting stage with gpt-5 and calibration
    - Create final forecasting prompts optimized for gpt-5's maximum reasoning capability
    - Add calibration checks and overconfidence reduction techniques
    - Implement uncertainty quantification and confidence scoring
    - Create forecast quality validation and tournament compliance checks
    - _Requirements: 4.5, 2.4_

- [x] 5. Budget-Aware Operation Manager
  - [x] 5.1 Implement dynamic operation mode detection and switching
    - Create budget utilization monitoring and threshold detection
    - Implement automatic operation mode transitions
    - Add operation mode logging and performance impact tracking
    - Create emergency protocol activation and management
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 5.2 Develop cost optimization strategies for each operation mode
    - Implement model selection adjustments per operation mode
    - Create task prioritization algorithms for budget conservation
    - Add research depth adaptation based on budget constraints
    - Implement graceful feature degradation for emergency modes
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 6. Enhanced Error Handling and Recovery Systems
  - [x] 6.1 Create comprehensive error classification and handling
    - Implement model-specific error detection and classification
    - Add budget exhaustion error handling with graceful degradation
    - Create API failure detection and intelligent retry mechanisms
    - Implement quality validation failure recovery with prompt revision
    - _Requirements: 7.4, 1.2_

  - [x] 6.2 Develop intelligent fallback strategies
    - Create model tier fallback with performance preservation
    - Implement cross-provider API fallback mechanisms
    - Add emergency mode activation for critical failures
    - Create comprehensive error logging and alerting system
    - _Requirements: 7.4, 1.2_

- [x] 7. Performance Monitoring and Analytics Integration
  - [x] 7.1 Implement real-time cost and performance tracking
    - Create model selection effectiveness monitoring
    - Add cost per question tracking with tier breakdown
    - Implement quality score trending and analysis
    - Create tournament competitiveness indicators and alerts
    - _Requirements: 8.1, 8.2, 8.3_

  - [x] 7.2 Develop optimization analytics and recommendations
    - Create cost-effectiveness analysis for model routing decisions
    - Implement performance correlation analysis between cost and quality
    - Add strategic adjustment recommendations based on tournament phase
    - Create budget allocation optimization suggestions
    - _Requirements: 8.4, 8.5_

- [x] 8. Main Application Integration and Enhancement
  - [x] 8.1 Update TemplateForecaster with enhanced tri-model integration
    - Integrate enhanced tri-model router with existing forecasting workflow
    - Update research methods to use multi-stage validation pipeline
    - Modify forecasting methods to use tier-optimized prompts
    - Add comprehensive error handling and fallback mechanisms
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 8.2 Implement seamless budget manager integration
    - Connect enhanced router with existing budget management systems
    - Add operation mode integration with budget alerts
    - Implement cost tracking integration with performance monitoring
    - Create tournament compliance integration with existing systems
    - _Requirements: 7.1, 7.2, 7.3_

- [x] 9. Environment Configuration and OpenRouter Setup
  - [x] 9.1 Update environment variables for OpenRouter model configuration
    - Add OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_HTTP_REFERER, OPENROUTER_APP_TITLE
    - Configure DEFAULT_MODEL=openai/gpt-5, MINI_MODEL=openai/gpt-5-mini, NANO_MODEL=openai/gpt-5-nano
    - Add FREE_FALLBACK_MODELS=openai/gpt-oss-20b:free,moonshotai/kimi-k2:free
    - Update model pricing configuration for accurate OpenRouter cost tracking
    - Add operation mode threshold configuration variables
    - _Requirements: 1.1, 1.3, 8.1, 8.2_

  - [x] 9.2 Implement OpenRouter model availability detection and auto-configuration
    - Create OpenRouter model availability checking on startup
    - Implement automatic fallback chain through OpenRouter models
    - Add OpenRouter provider routing and health monitoring
    - Create configuration validation and error reporting for OpenRouter setup
    - _Requirements: 1.2, 1.3, 8.4_

- [ ] 10. Testing and Validation Implementation
  - [ ] 10.1 Create comprehensive unit tests for enhanced components
    - Test enhanced tri-model router with all routing scenarios
    - Validate anti-slop prompt generation and quality directives
    - Test multi-stage validation pipeline with various inputs
    - Create budget-aware operation manager testing suite
    - _Requirements: All requirements validation_

  - [ ] 10.2 Implement integration tests for complete workflow
    - Test end-to-end question processing with enhanced system
    - Validate budget-aware operation mode switching
    - Test error handling and recovery mechanisms
    - Create tournament simulation tests with full budget scenarios
    - _Requirements: All requirements validation_

  - [ ] 10.3 Develop performance and cost optimization tests
    - Create cost-effectiveness measurement tests
    - Implement quality vs. cost correlation analysis
    - Test model selection optimization under various conditions
    - Validate tournament compliance with enhanced system
    - _Requirements: All requirements validation_

## Implementation Notes

- Each task builds incrementally on the existing tri-model infrastructure
- Anti-slop directives are integrated at the prompt level for maximum effectiveness
- Budget-aware routing ensures optimal cost-performance throughout the tournament
- Multi-stage validation provides quality assurance without sacrificing efficiency
- All enhancements maintain backward compatibility with existing systems

## Success Criteria

Upon completion of all tasks, the enhanced system will:

- Process 5000+ questions within the $100 budget (vs. 300 with single GPT-4o)
- Implement state-of-the-art anti-slop directives for quality assurance
- Provide intelligent model routing based on task complexity and budget constraints
- Maintain tournament compliance with automated operation and transparency
- Deliver superior forecasting performance through optimized prompt engineering
- Offer comprehensive monitoring and analytics for continuous optimization
- Ensure robust error handling and graceful degradation under all conditions

## Technical Architecture Summary

The enhanced system implements a strategic cost-performance triangle via OpenRouter:

1. **Tier 3: openai/gpt-5-nano** ($0.05/1M tokens): Ultra-fast validation, parsing, simple summaries
2. **Tier 2: openai/gpt-5-mini** ($0.25/1M tokens): Research synthesis, intermediate reasoning
3. **Tier 1: openai/gpt-5** ($1.50/1M tokens): Final forecasting, complex analysis
4. **Free Fallbacks: openai/gpt-oss-20b:free, moonshotai/kimi-k2:free** ($0/1M tokens): Budget exhaustion operation

With sophisticated anti-slop quality guards:

- Chain-of-Verification internal reasoning
- Evidence traceability pre-checks
- Source citation requirements
- Uncertainty acknowledgment
- Calibration and overconfidence reduction

And intelligent budget-aware operation via OpenRouter:

- Normal mode (0-70% budget): Optimal GPT-5 model selection with provider routing
- Conservative mode (70-85% budget): GPT-5 mini/nano preferred routing
- Emergency mode (85-95% budget): Free models preferred with GPT-5 nano for critical tasks
- Critical mode (95-100% budget): Free models only operation

With cost-optimized research strategy:

- **Primary**: AskNews API (100% FREE for 4 months via METACULUSQ4 tournament code)
- **Synthesis**: gpt-5-mini for analysis and citation formatting
- **Fallback**: Free models (gpt-oss-20b:free, kimi-k2:free) when AskNews quota exhausted
- **NO expensive APIs**: Perplexity, Claude, or other paid research services eliminated

This implementation creates a tournament-winning forecasting system that maximizes quality per dollar spent while maintaining the highest standards of evidence-based reasoning and tournament compliance through OpenRouter's unified gateway approach with GPT-5 cost optimization.
