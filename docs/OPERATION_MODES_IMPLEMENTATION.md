# Budget-Aware Operation Modes Implementation

## Overview

This document describes the implementation of budget-aware operation modes for the tournament API optimization system. The operation modes provide automatic budget management with graceful degradation when approaching budget limits.

## Implementation Summary

### Core Components

#### 1. Operation Modes (`src/infrastructure/config/operation_modes.py`)

**Three Operation Modes:**
- **NORMAL**: Full functionality with optimal model selection
- **CONSERVATIVE**: Reduced functionality to conserve budget (triggered at 80% utilization)
- **EMERGENCY**: Minimal functionality to preserve remaining budget (triggered at 95% utilization)

**Key Features:**
- Automatic mode switching based on budget utilization
- Question priority filtering (emergency mode only processes high/critical priority)
- Model selection constraints (emergency mode forces cheapest models)
- Processing limits adjustment (batch sizes, retries, timeouts)
- Mode transition history tracking

#### 2. Enhanced LLM Config Integration

Updated `src/infrastructure/config/enhanced_llm_config.py` to:
- Automatically check and update operation modes before task execution
- Use operation mode manager for model selection
- Apply processing limits from current operation mode
- Provide question processing checks

### Operation Mode Configurations

| Mode         | Budget Threshold | Max Questions/Batch | Research Model | Forecast Model | Max Retries | Timeout | Complexity Analysis | Skip Low Priority |
| ------------ | ---------------- | ------------------- | -------------- | -------------- | ----------- | ------- | ------------------- | ----------------- |
| Normal       | 0%               | 10                  | gpt-4o-mini    | gpt-4o         | 3           | 90s     | ✓                   | ✗                 |
| Conservative | 80%              | 5                   | gpt-4o-mini    | gpt-4o-mini    | 2           | 60s     | ✓                   | ✓                 |
| Emergency    | 95%              | 2                   | gpt-4o-mini    | gpt-4o-mini    | 1           | 45s     | ✗                   | ✓                 |

### Graceful Degradation Features

#### Question Processing Filters
- **Normal Mode**: Processes all questions regardless of priority
- **Conservative Mode**: Skips low priority questions
- **Emergency Mode**: Only processes high/critical priority questions

#### Model Selection Constraints
- **Normal Mode**: Uses complexity-based model selection
- **Conservative Mode**: Limits expensive models for research tasks
- **Emergency Mode**: Forces cheapest model (gpt-4o-mini) for all tasks

#### Processing Limits
- Batch sizes reduced in higher constraint modes
- Retry attempts limited to conserve budget
- Timeouts reduced to prevent resource waste
- Complexity analysis disabled in emergency mode

### Integration Points

#### Budget Manager Integration
- Monitors budget utilization in real-time
- Triggers automatic mode transitions
- Provides budget status for decision making

#### Task Complexity Analyzer Integration
- Uses complexity assessment for model selection (when enabled)
- Respects operation mode constraints
- Disabled in emergency mode to save processing

#### Enhanced LLM Config Integration
- Automatic mode checking before task execution
- Model selection through operation mode manager
- Processing limit application
- Question processing validation

## API Usage

### Basic Usage

```python
from src.infrastructure.config.operation_modes import operation_mode_manager

# Check current mode
current_mode = operation_mode_manager.get_current_mode()

# Check if question can be processed
can_process, reason = operation_mode_manager.can_process_question("high")

# Get model for task
model = operation_mode_manager.get_model_for_task("forecast", complexity_assessment)

# Get processing limits
limits = operation_mode_manager.get_processing_limits()
```

### Enhanced LLM Config Usage

```python
from src.infrastructure.config.enhanced_llm_config import enhanced_llm_config

# Get LLM with automatic mode management
llm = enhanced_llm_config.get_llm_for_task("research")

# Check question processing
can_process, reason = enhanced_llm_config.can_process_question("normal")
```

## Testing

### Unit Tests
- `tests/unit/infrastructure/test_operation_modes.py`: Core functionality tests
- 15 test cases covering all major features
- Mode transitions, question processing, model selection

### Integration Tests
- `tests/integration/test_operation_modes_integration.py`: Integration with enhanced LLM config
- 4 test cases covering integration scenarios
- Mode changes during task requests, configuration logging

### Demo Script
- `examples/operation_modes_demo.py`: Interactive demonstration
- Shows all operation modes in action
- Demonstrates graceful degradation strategies

## Key Benefits

1. **Automatic Budget Management**: No manual intervention required
2. **Graceful Degradation**: System remains functional as budget depletes
3. **Priority-Based Processing**: Critical questions processed even in emergency mode
4. **Cost Optimization**: Automatic model downgrading to conserve budget
5. **Transparent Operation**: Full logging and history tracking
6. **Flexible Configuration**: Easy to adjust thresholds and constraints

## Requirements Satisfied

This implementation satisfies the following requirements from the specification:

- **1.4**: Budget-aware operation modes with automatic switching
- **4.5**: Graceful degradation when approaching budget limits

The system now provides comprehensive budget management with three distinct operation modes that automatically adjust system behavior based on budget utilization, ensuring optimal resource usage while maintaining functionality.
