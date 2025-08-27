# Tournament Testing Framework

This directory contains comprehensive testing framework for tournament simulation, competitive pressure testing, and agent performance validation.

## Overview

The tournament testing framework provides end-to-end validation of the AI forecasting bot's performance under realistic tournament conditions, including:

- **Tournament Simulation**: Complete tournament scenarios with time pressure, resource constraints, and competitive dynamics
- **Competitive Pressure Testing**: Validation of system behavior under various pressure conditions
- **Recovery and Resilience Testing**: Testing system recovery from failures and degraded conditions
- **Agent Performance Benchmarking**: Individual agent accuracy, calibration, and performance validation
- **Calibration Validation**: Comprehensive calibration accuracy testing and bias detection

## Test Structure

### Core Test Files

- `test_tournament_simulation.py`: End-to-end tournament simulation testing
- `test_competitive_pressure.py`: Competitive pressure and stress testing
- `test_recovery_resilience.py`: Recovery and resilience testing under failure conditions
- `test_agent_performance.py`: Individual agent performance benchmarking
- `test_calibration_validation.py`: Calibration accuracy validation and bias detection

### Configuration

- `conftest.py`: Tournament-specific fixtures and configuration
- `README.md`: This documentation file

## Key Features

### Tournament Simulation (`test_tournament_simulation.py`)

- **Complete Tournament Scenarios**: Simulates full tournament conditions with multiple questions, time constraints, and competitive pressure
- **Performance Benchmarking**: Measures accuracy, calibration, Brier scores, and execution time
- **Resource Usage Tracking**: Monitors API calls, memory usage, and computational resources
- **Error Analysis**: Comprehensive error tracking and analysis

**Key Classes:**
- `TournamentSimulator`: Main simulation engine
- `TournamentScenario`: Defines tournament parameters and constraints
- `TournamentResult`: Captures comprehensive results and metrics

### Competitive Pressure Testing (`test_competitive_pressure.py`)

- **Time Pressure**: Tests performance under tight time constraints
- **Resource Pressure**: Validates behavior with limited computational resources
- **Accuracy Pressure**: Tests high-stakes scenarios with accuracy requirements
- **Combined Pressure**: Tests system resilience under multiple pressure factors

**Key Classes:**
- `CompetitivePressureTester`: Pressure testing engine
- `CompetitivePressureTest`: Defines pressure test scenarios

### Recovery and Resilience Testing (`test_recovery_resilience.py`)

- **Failure Injection**: Simulates various failure types (API, network, memory, timeout)
- **Recovery Pattern Analysis**: Tests immediate, gradual, and delayed recovery patterns
- **Circuit Breaker Testing**: Validates circuit breaker behavior during failures
- **Cascading Failure Recovery**: Tests recovery from multiple simultaneous failures

**Key Classes:**
- `ResilienceTester`: Main resilience testing engine
- `FailureInjector`: Configurable failure injection system
- `FailureScenario`: Defines failure scenarios and recovery expectations

### Agent Performance Testing (`test_agent_performance.py`)

- **Individual Agent Benchmarking**: Comprehensive performance metrics for each agent type
- **Ensemble Optimization**: Tests ensemble aggregation methods and optimization
- **Calibration Analysis**: Detailed calibration analysis with reliability diagrams
- **Timing Analysis**: Response time and performance benchmarking

**Key Classes:**
- `AgentPerformanceTester`: Individual agent testing framework
- `EnsembleOptimizationTester`: Ensemble performance testing
- `PerformanceBenchmark`: Defines performance thresholds and expectations

### Calibration Validation (`test_calibration_validation.py`)

- **Calibration Accuracy**: Validates prediction calibration using reliability diagrams
- **Bias Detection**: Detects overconfidence, underconfidence, anchoring, availability, and confirmation biases
- **Sharpness Analysis**: Measures prediction discrimination ability
- **Resolution Analysis**: Tests ability to distinguish between different outcomes

**Key Classes:**
- `CalibrationValidator`: Main calibration validation engine
- `BiasTestScenario`: Defines bias testing scenarios
- `CalibrationBin`: Represents calibration analysis bins

## Running Tests

### Run All Tournament Tests
```bash
pytest tests/tournament/ -v
```

### Run Specific Test Categories
```bash
# Tournament simulation tests
pytest tests/tournament/test_tournament_simulation.py -v

# Competitive pressure tests
pytest -m pressure tests/tournament/ -v

# Resilience tests
pytest -m resilience tests/tournament/ -v

# Performance tests
pytest -m performance tests/tournament/ -v

# Calibration tests
pytest -m calibration tests/tournament/ -v

# Bias detection tests
pytest -m bias tests/tournament/ -v
```

### Run with Coverage
```bash
pytest tests/tournament/ --cov=src --cov-report=html
```

## Test Markers

The framework uses pytest markers to categorize tests:

- `@pytest.mark.tournament`: All tournament-related tests
- `@pytest.mark.pressure`: Competitive pressure tests
- `@pytest.mark.resilience`: Recovery and resilience tests
- `@pytest.mark.performance`: Agent performance tests
- `@pytest.mark.calibration`: Calibration validation tests
- `@pytest.mark.bias`: Bias detection tests

## Performance Benchmarks

### Default Benchmarks

- **Accuracy**: ≥ 65% for individual agents, ≥ 70% for ensembles
- **Brier Score**: ≤ 0.35 for individual agents, ≤ 0.30 for ensembles
- **Calibration Score**: ≥ 0.70 (Expected Calibration Error ≤ 0.30)
- **Response Time**: ≤ 10 seconds per question
- **Completion Rate**: ≥ 90% under normal conditions, ≥ 60% under pressure
- **Error Rate**: ≤ 10% under normal conditions, ≤ 25% under pressure

### Tournament-Specific Benchmarks

- **Tournament Completion**: ≥ 80% of questions completed within time limits
- **Competitive Performance**: Top 25% performance in simulated tournaments
- **Recovery Time**: ≤ 15 seconds recovery from failures
- **Pressure Degradation**: ≤ 30% performance degradation under high pressure

## Failure Scenarios

### Supported Failure Types

1. **API Failures**: Service unavailable, rate limiting, authentication errors
2. **Network Timeouts**: Connection timeouts, slow responses
3. **Memory Exhaustion**: Out of memory conditions
4. **Partial Failures**: Intermittent service degradation
5. **Cascading Failures**: Multiple simultaneous failure types

### Recovery Patterns

1. **Immediate Recovery**: Instant recovery after failure period
2. **Gradual Recovery**: Slowly improving success rate
3. **Delayed Recovery**: Full failure followed by complete recovery

## Bias Detection

### Supported Bias Types

1. **Overconfidence Bias**: Excessive confidence in predictions
2. **Underconfidence Bias**: Insufficient confidence in accurate predictions
3. **Anchoring Bias**: Over-reliance on initial information or round numbers
4. **Availability Bias**: Overweighting recent or memorable events
5. **Confirmation Bias**: Seeking information that confirms prior beliefs

### Detection Methods

- **Statistical Analysis**: Confidence-accuracy correlation analysis
- **Pattern Recognition**: Clustering around anchor points
- **Reasoning Analysis**: Text analysis of reasoning patterns
- **Calibration Curves**: Reliability diagram analysis

## Integration with CI/CD

The tournament testing framework is designed to integrate with continuous integration pipelines:

```yaml
# Example GitHub Actions configuration
- name: Run Tournament Tests
  run: |
    pytest tests/tournament/ --junitxml=tournament-results.xml
    pytest tests/tournament/ --cov=src --cov-report=xml
```

## Extending the Framework

### Adding New Test Scenarios

1. Create new scenario classes inheriting from base scenario types
2. Define scenario parameters and expected outcomes
3. Implement scenario-specific validation logic
4. Add appropriate pytest markers

### Adding New Bias Detection

1. Implement detection logic in `CalibrationValidator`
2. Create test scenarios in `BiasTestScenario`
3. Add mitigation recommendations
4. Include validation tests

### Adding New Failure Types

1. Extend `FailureInjector` with new failure types
2. Define recovery patterns and expectations
3. Create test scenarios and validation logic
4. Document expected behavior and thresholds

## Troubleshooting

### Common Issues

1. **Test Timeouts**: Increase timeout values in test configuration
2. **Mock Failures**: Verify mock setup and side effects
3. **Assertion Errors**: Check benchmark thresholds and expected values
4. **Resource Issues**: Monitor memory and CPU usage during tests

### Debug Mode

Enable debug logging for detailed test execution information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Requirements Traceability

This testing framework addresses the following requirements:

- **Requirement 10.3**: Tournament simulation and competitive testing
- **Requirement 10.4**: Recovery and resilience testing
- **Requirement 10.5**: Performance benchmarking and quality assurance
- **Requirement 10.1**: Individual agent accuracy validation
- **Requirement 10.2**: Ensemble optimization testing
- **Requirement 10.3**: Reasoning quality assessment and bias detection

## Future Enhancements

- **Real Tournament Integration**: Connect to actual Metaculus tournaments
- **Advanced Bias Detection**: Machine learning-based bias detection
- **Performance Optimization**: Automated performance tuning recommendations
- **Comparative Analysis**: Cross-agent performance comparison and ranking
- **Historical Analysis**: Long-term performance trend analysis
