# Requirements Document

## Introduction

This document outlines the requirements for fixing 37 failing unit tests in the Metaculus forecasting bot with **28 hours remaining until tournament start**. The failures span multiple categories including agent constructor mismatches, missing methods, configuration issues, and API changes. Given the time constraint, we must prioritize tournament-critical fixes over comprehensive test coverage.

**Context**: 979 tests are passing, indicating core functionality works. The priority is ensuring tournament deployment readiness while maintaining forecasting capability.

## Requirements

### Requirement 1: Agent Constructor Compatibility

**User Story:** As a developer, I want the agent tests to pass so that I can verify the agent functionality works correctly.

#### Acceptance Criteria

1. WHEN ChainOfThoughtAgent tests run THEN they SHALL NOT fail due to unexpected keyword arguments
2. WHEN agent constructors are called THEN they SHALL accept the parameters expected by the tests
3. IF tests expect llm_client and search_client parameters THEN agents SHALL either accept these parameters or tests SHALL be updated to match the actual API

### Requirement 2: Mock Object Attribute Fixes

**User Story:** As a developer, I want calibration improvement tests to pass so that I can verify the calibration logic works correctly.

#### Acceptance Criteria

1. WHEN calibration improvement tests run THEN mock objects SHALL have the required 'result' attribute
2. WHEN tests access mock.result THEN it SHALL return appropriate test data
3. WHEN calibration tests run THEN they SHALL NOT fail with AttributeError for missing attributes

### Requirement 3: Service Method Implementation

**User Story:** As a developer, I want tournament compliance tests to pass so that I can verify compliance validation works correctly.

#### Acceptance Criteria

1. WHEN tournament compliance tests run THEN all expected methods SHALL exist on the validator classes
2. WHEN tests call validate_reasoning_transparency THEN the method SHALL exist and return appropriate results
3. WHEN tests call compliance validation methods THEN they SHALL NOT fail with AttributeError

### Requirement 4: Error Handling System Fixes

**User Story:** As a developer, I want error handling tests to pass so that I can verify the error recovery system works correctly.

#### Acceptance Criteria

1. WHEN error handling tests run THEN retry delay calculations SHALL match expected values
2. WHEN error recovery is tested THEN the correct recovery strategies SHALL be returned
3. WHEN error statistics are calculated THEN they SHALL match expected counts

### Requirement 5: Configuration and API Consistency

**User Story:** As a developer, I want configuration tests to pass so that I can verify the bot configuration works correctly.

#### Acceptance Criteria

1. WHEN bot configuration tests run THEN default values SHALL match expected test assertions
2. WHEN prediction objects are created THEN they SHALL accept the parameters expected by tests
3. WHEN API interfaces are tested THEN they SHALL match the actual implementation signatures

### Requirement 6: Prompt and Agent Integration Fixes

**User Story:** As a developer, I want prompt optimization tests to pass so that I can verify the prompt system works correctly.

#### Acceptance Criteria

1. WHEN prompt optimization tests run THEN all expected methods SHALL exist on prompt classes
2. WHEN agent integration tests run THEN they SHALL NOT fail with StopAsyncIteration errors
3. WHEN prompt generation is tested THEN the methods SHALL return appropriate prompt structures

### Requirement 7: Tournament Deployment Priority

**User Story:** As a tournament participant, I want the bot to be deployable within 28 hours so that I can compete in the forecasting tournament.

#### Acceptance Criteria

1. WHEN deployment is attempted THEN tournament-critical functionality SHALL work regardless of test status
2. WHEN core forecasting workflow is tested THEN it SHALL pass integration tests
3. IF some unit tests fail THEN deployment SHALL still proceed if core functionality is verified
4. WHEN tournament compliance is checked THEN all required validation methods SHALL exist and return appropriate results

### Requirement 8: Emergency Deployment Capability

**User Story:** As a tournament participant, I want the ability to deploy with known test failures so that I don't miss the tournament deadline.

#### Acceptance Criteria

1. WHEN critical tests pass THEN deployment SHALL proceed even if non-critical tests fail
2. WHEN tournament compliance methods are missing THEN stub implementations SHALL be provided
3. WHEN constructor mismatches occur THEN backward-compatible fixes SHALL be implemented
4. WHEN configuration tests fail THEN tournament-appropriate defaults SHALL be used

### Requirement 9: Test Failure Triage

**User Story:** As a developer under time pressure, I want to focus on the most critical test failures so that I can maximize tournament readiness.

#### Acceptance Criteria

1. WHEN test failures are analyzed THEN they SHALL be categorized by tournament impact (critical/medium/low)
2. WHEN fixes are prioritized THEN tournament-blocking issues SHALL be addressed first
3. WHEN time is limited THEN non-critical test failures MAY be deferred post-tournament
4. WHEN core functionality is verified THEN deployment SHALL be considered ready

### Requirement 10: Network Timeout and CI/CD Resilience

**User Story:** As a tournament participant, I want my bot deployment to be resilient to network timeouts so that infrastructure issues don't prevent tournament participation.

#### Acceptance Criteria

1. WHEN GitHub Actions encounters network timeouts THEN the workflow SHALL retry with alternative installation methods
2. WHEN Poetry installation fails THEN the system SHALL fallback to pip installation
3. WHEN network connectivity is poor THEN installation steps SHALL have appropriate timeout limits
4. WHEN CI/CD fails THEN manual deployment options SHALL be available as backup

### Requirement 11: Emergency Deployment Fallbacks

**User Story:** As a tournament participant, I want multiple deployment options so that I can still compete if the primary deployment method fails.

#### Acceptance Criteria

1. WHEN GitHub Actions fails THEN manual deployment instructions SHALL be available
2. WHEN Poetry fails THEN pip-based installation SHALL work as fallback
3. WHEN dependencies can't be installed THEN core functionality SHALL still be testable locally
4. WHEN all else fails THEN the bot SHALL be deployable on any Linux server with basic Python

### Requirement 12: Time-Critical Bug Prioritization

**User Story:** As a developer with 27 hours until tournament start, I want to fix only the bugs that actually prevent tournament participation.

#### Acceptance Criteria

1. WHEN bugs are triaged THEN tournament-blocking issues SHALL be identified first
2. WHEN constructor mismatches occur THEN they SHALL be fixed with minimal code changes
3. WHEN missing methods are found THEN stub implementations SHALL be provided for tournament compliance
4. WHEN configuration issues arise THEN tournament-appropriate defaults SHALL be used immediately
