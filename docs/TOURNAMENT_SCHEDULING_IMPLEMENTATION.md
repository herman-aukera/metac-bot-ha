# Tournament Scheduling Implementation

## Overview

This document describes the implementation of optimized tournament scheduling for the AI Forecasting Bot, addressing the requirements to change from 30-minute to 4-hour scheduling frequency with configurable options for deadline-aware scheduling.

## Changes Made

### 1. Updated GitHub Actions Workflows

#### Main Tournament Workflow (`.github/workflows/run_bot_on_tournament.yaml`)
- **Changed scheduling**: From every 30 minutes to every 4 hours (`0 */4 * * *`)
- **Added workflow_dispatch inputs**:
  - `scheduling_frequency_hours`: Override default frequency
  - `tournament_mode`: Choose between normal, critical, final_24h modes
- **Enhanced environment variables**: Support for input parameters and repository variables
- **Added scheduling display**: Shows current configuration in workflow logs

#### Deadline-Aware Workflow (`.github/workflows/tournament_deadline_aware.yaml`)
- **Flexible scheduling**: Supports normal (4h), critical (2h), and final_24h (1h) modes
- **Manual control**: Workflow dispatch with custom frequency options
- **Dynamic configuration**: Automatically adjusts based on tournament phase

#### Legacy Workflow (`workflows/github-actions.yml`)
- **Updated cron schedule**: Changed from `*/30 * * * *` to `0 */4 * * *`
- **Added workflow_dispatch**: Manual trigger with frequency configuration
- **Environment variables**: Added scheduling configuration support

### 2. Environment Configuration

#### Template Files Updated
- **`.env.template`**: Contains all scheduling variables with default values
- **`.env.example`**: Includes scheduling configuration examples

#### Scheduling Variables Added
```bash
SCHEDULING_FREQUENCY_HOURS=4
DEADLINE_AWARE_SCHEDULING=true
CRITICAL_PERIOD_FREQUENCY_HOURS=2
FINAL_24H_FREQUENCY_HOURS=1
TOURNAMENT_SCOPE=seasonal
```

### 3. Configuration Scripts

#### GitHub Secrets Setup (`scripts/setup-github-secrets.sh`)
- **Updated scheduling information**: Reflects new 4-hour frequency
- **Added repository variables section**: Instructions for scheduling customization

#### New Configuration Script (`scripts/configure-tournament-scheduling.sh`)
- **Comprehensive guide**: Step-by-step scheduling configuration
- **Multiple operation modes**: Normal, active monitoring, conservative
- **Deadline-aware setup**: Instructions for automatic frequency adjustment
- **Manual control guidance**: How to override scheduling for individual runs

## Scheduling Modes

### 1. Normal Operation (Default)
- **Frequency**: Every 4 hours
- **Budget impact**: Optimized for $100 budget
- **Use case**: Standard tournament operation

### 2. Critical Period
- **Frequency**: Every 2 hours
- **Budget impact**: Moderate increase
- **Use case**: Important tournament phases

### 3. Final 24 Hours
- **Frequency**: Every 1 hour
- **Budget impact**: High usage
- **Use case**: Last day before question deadlines

## Configuration Options

### Repository Variables (GitHub Settings)
Configure these in: Settings → Secrets and variables → Actions → Variables

| Variable                          | Default  | Description                           |
| --------------------------------- | -------- | ------------------------------------- |
| `SCHEDULING_FREQUENCY_HOURS`      | 4        | Default scheduling frequency          |
| `DEADLINE_AWARE_SCHEDULING`       | true     | Enable automatic deadline adjustments |
| `CRITICAL_PERIOD_FREQUENCY_HOURS` | 2        | Frequency during critical periods     |
| `FINAL_24H_FREQUENCY_HOURS`       | 1        | Frequency in final 24 hours           |
| `TOURNAMENT_SCOPE`                | seasonal | Tournament operation scope            |

### Workflow Dispatch Inputs
Override settings for individual runs:
- **Scheduling frequency**: Custom hours (1-24)
- **Tournament mode**: normal, critical, final_24h

## Benefits

### 1. Budget Efficiency
- **Reduced API calls**: 4-hour frequency vs 30-minute saves ~87% of calls
- **Smart resource allocation**: Higher frequency only when needed
- **Cost predictability**: Better budget planning for tournament duration

### 2. Competitive Advantage
- **Deadline awareness**: Automatic frequency increase near deadlines
- **Flexible response**: Manual overrides for urgent situations
- **Market responsiveness**: Balanced between efficiency and timeliness

### 3. Operational Control
- **Multiple workflows**: Choose appropriate scheduling for different scenarios
- **Easy configuration**: Repository variables for team-wide settings
- **Manual overrides**: Individual run customization

## Usage Instructions

### 1. Standard Setup
1. Configure required secrets (METACULUS_TOKEN, OPENROUTER_API_KEY)
2. Set repository variables for custom scheduling (optional)
3. Enable workflows - they will run automatically every 4 hours

### 2. Custom Scheduling
1. Go to Actions → Select workflow
2. Click "Run workflow"
3. Choose frequency and mode
4. Run with custom settings

### 3. Deadline Management
1. Use `tournament_deadline_aware.yaml` workflow
2. Enable different cron schedules based on tournament phase
3. Monitor budget usage and adjust as needed

## Monitoring

### Workflow Logs
Each run displays:
- Current scheduling configuration
- Tournament mode and frequency
- Budget and resource usage

### Budget Tracking
- Monitor API costs per run
- Track budget utilization over time
- Adjust frequency based on remaining budget

## Requirements Satisfied

✅ **Requirement 2.1**: Changed cron schedule from every 30 minutes to every 4 hours
✅ **Requirement 2.2**: Added configurable scheduling through environment variables
✅ **Requirement 2.4**: Implemented deadline-aware scheduling for critical periods

## Next Steps

1. **Test workflows**: Run manual workflows to verify configuration
2. **Monitor performance**: Track budget usage and forecast quality
3. **Adjust as needed**: Fine-tune frequency based on tournament dynamics
4. **Document results**: Record optimal settings for future tournaments

This implementation provides a robust, flexible scheduling system that balances budget efficiency with competitive performance in the tournament environment.
