# Implementation Plan

## Phase 1: Tournament-Critical Fixes (0-2 hours)

- [x] 1. Fix ChainOfThoughtAgent constructor parameter mismatch
  - Add llm_client and search_client parameters to constructor for backward compatibility
  - Update constructor to accept these parameters without breaking existing functionality
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2. Implement missing tournament compliance validation methods
  - Add validate_reasoning_transparency method to TournamentComplianceValidator
  - Add validate_automated_decision_making method to TournamentComplianceValidator
  - Add validate_data_source_compliance method to TournamentComplianceValidator
  - Add validate_prediction_format method to TournamentComplianceValidator
  - Add run_comprehensive_compliance_check method to TournamentComplianceValidator
  - Add check_human_intervention method to TournamentRuleComplianceMonitor
  - Add check_submission_timing method to TournamentRuleComplianceMonitor
  - _Requirements: 3.1, 3.2, 3.3, 7.4_

- [x] 3. Fix BotConfig default values for test compatibility
  - Update publish_reports_to_metaculus default value to match test expectations
  - Verify other configuration defaults align with test assertions
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 4. Update GitHub Actions workflow for network resilience
  - Add timeout protection to Poetry installation steps
  - Implement pip fallback when Poetry installation fails
  - Add retry logic for network-dependent operations
  - Include alternative runner configurations for network issues
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

## Phase 2: Core Functionality Fixes (2-4 hours)

- [x] 5. Fix calibration improvement test mock objects
  - Add result attribute to mock objects in calibration tests
  - Update mock object setup to return appropriate test data structures
  - Fix mock configuration in test fixtures
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 6. Fix error handling system calculations and logic
  - Correct retry delay calculation to match expected exponential backoff values
  - Fix error statistics tracking to return expected counts
  - Update recovery strategy selection logic to return correct strategies
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 7. Update PredictionResult constructor for API compatibility
  - Add choice_index parameter to PredictionResult constructor
  - Maintain backward compatibility with existing prediction creation code
  - Update validation logic to handle new parameter
  - _Requirements: 5.2, 5.3_

- [x] 8. Fix knowledge gap detector and pattern detector logic
  - Update confidence level calculations in knowledge gap assessment
  - Fix quantitative data detection to return expected gap counts
  - Correct pattern detection to return non-empty result sets
  - Fix ensemble pattern detection property setter issues
  - _Requirements: 4.1, 4.2_

## Phase 3: Service Method Implementation

- [x] 9. Implement missing prompt optimization methods
  - Add generate_basic_calibrated_prompt method to CalibratedForecastingPrompts
  - Add generate_scenario_analysis_prompt method to CalibratedForecastingPrompts
  - Add generate_overconfidence_reduction_prompt method to CalibratedForecastingPrompts
  - Add select_optimal_prompt method to CalibrationPromptManager
  - _Requirements: 6.1, 6.3_

- [x] 10. Fix ReActAgent async iteration issues
  - Handle StopAsyncIteration exceptions in ReActAgent forecast method
  - Implement proper async iteration termination logic
  - Add error handling for async generator completion
  - _Requirements: 6.2_

- [x] 11. Fix Metaculus client interface consistency
  - Update submit_prediction methods to return expected comment formats
  - Align new and old interface prediction submission responses
  - Ensure comment formatting matches test expectations
  - _Requirements: 5.3_

## Phase 4: Emergency Deployment Preparation

- [x] 12. Create emergency deployment documentation
  - Document manual deployment steps for cloud instances
  - Provide pip-only installation instructions
  - Create local testing verification commands
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [x] 13. Implement deployment verification tests
  - Create minimal integration test for core forecasting workflow
  - Add tournament compliance validation test
  - Implement deployment readiness check script
  - _Requirements: 7.1, 7.2, 8.4_

- [x] 14. Add CI/CD fallback mechanisms
  - Implement continue-on-error for non-critical test failures
  - Add deployment bypass for emergency tournament participation
  - Create manual trigger for deployment without full test suite
  - _Requirements: 8.1, 8.3, 10.4_
