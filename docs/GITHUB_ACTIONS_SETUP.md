# GitHub Actions Setup Guide

This guide explains how to set up GitHub Actions workflows for the AI Forecasting Bot with proper secret management.

## 🔐 Required GitHub Secrets

The bot requires certain API keys to function. These should be configured as GitHub repository secrets.

### Setting up GitHub Secrets

1. Go to your GitHub repository
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"** for each required secret

### Required Secrets

These secrets **must** be configured for the bot to work:

| Secret Name          | Description                                   | Value                     |
| -------------------- | --------------------------------------------- | ------------------------- |
| `METACULUS_TOKEN`    | Metaculus API token for accessing tournaments | [Your Metaculus token]    |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM access             | [Your OpenRouter API key] |

### Optional Secrets

These secrets are optional. If not configured, the bot will use fallback/dummy values:

| Secret Name          | Description                  | Status                    |
| -------------------- | ---------------------------- | ------------------------- |
| `PERPLEXITY_API_KEY` | Enhanced search capabilities | Optional - using fallback |
| `EXA_API_KEY`        | Web search functionality     | Optional - using fallback |
| `OPENAI_API_KEY`     | OpenAI GPT models            | Optional - using fallback |
| `ANTHROPIC_API_KEY`  | Claude models                | Optional - using fallback |
| `ASKNEWS_CLIENT_ID`  | News search functionality    | Optional - using fallback |
| `ASKNEWS_SECRET`     | News search functionality    | Optional - using fallback |

## 🚀 Available Workflows

### 1. Tournament Forecasting (`run_bot_on_tournament.yaml`)

- **Trigger**: Every 4 hours (automatic) + manual dispatch
- **Purpose**: Forecast on new AI tournament questions with optimized scheduling
- **Features**: Budget-aware scheduling, concurrency control, monitoring
- **Timeout**: 30 minutes with automatic cancellation of previous runs
- **Scheduling**: Optimized for $100 budget and seasonal tournament scope

### 2. Deadline-Aware Tournament Forecasting (`tournament_deadline_aware.yaml`)

- **Trigger**: Configurable scheduling + manual dispatch with parameters
- **Purpose**: Advanced scheduling based on tournament deadlines
- **Features**:
  - Normal mode: Every 4 hours
  - Critical period: Every 2 hours (72 hours before deadline)
  - Final 24 hours: Every hour
- **Manual Controls**: Choose scheduling mode and custom frequency

### 2. Quarterly Cup Forecasting (`run_bot_on_quarterly_cup.yaml`)

- **Trigger**: Every 2 days at midnight (automatic) + manual dispatch
- **Purpose**: Forecast on Quarterly Cup questions
- **Features**: Extended timeout, specialized for quarterly tournaments
- **Timeout**: 45 minutes for comprehensive processing

### 3. Deployment Testing (`test_deployment.yaml`)

- **Trigger**: Push to main/develop branches + pull requests + manual dispatch
- **Purpose**: Test automated deployment and monitoring systems
- **Features**: API validation, configuration testing, dry run verification
- **Timeout**: 15 minutes for quick feedback

## 🧪 Testing the Setup

### Manual Testing

1. Go to the **Actions** tab in your GitHub repository
2. Select either workflow:
   - "Forecast on new AI tournament questions"
   - "Forecast on Quarterly Cup"
3. Click **"Run workflow"** to test manually

### Automatic Execution

Once secrets are configured, workflows will run automatically:

- Tournament forecasting: Every 30 minutes (optimized for responsiveness)
- Quarterly Cup: Every 2 days at midnight (extended processing time)
- Deployment testing: On every push/PR (ensures system reliability)

## ⏰ Tournament Scheduling Configuration

### Environment Variables for Scheduling

The bot now supports configurable scheduling through environment variables:

| Variable                          | Default  | Description                                   |
| --------------------------------- | -------- | --------------------------------------------- |
| `SCHEDULING_FREQUENCY_HOURS`      | 4        | Base frequency for tournament runs (hours)    |
| `DEADLINE_AWARE_SCHEDULING`       | true     | Enable deadline-aware frequency adjustment    |
| `CRITICAL_PERIOD_FREQUENCY_HOURS` | 2        | Frequency during critical period (72h before) |
| `FINAL_24H_FREQUENCY_HOURS`       | 1        | Frequency in final 24 hours before deadline   |
| `TOURNAMENT_SCOPE`                | seasonal | Tournament scope (seasonal vs daily)          |

### GitHub Repository Variables

You can configure scheduling through GitHub repository variables:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click the **Variables** tab
3. Add repository variables for scheduling control

### Scheduling Modes

#### Normal Tournament Operation
- **Frequency**: Every 4 hours
- **Use case**: Standard tournament operation within budget
- **Cron**: `0 */4 * * *`

#### Critical Period (72 hours before deadline)
- **Frequency**: Every 2 hours
- **Use case**: Increased activity as deadlines approach
- **Cron**: `0 */2 * * *`

#### Final 24 Hours
- **Frequency**: Every hour
- **Use case**: Maximum responsiveness for final submissions
- **Cron**: `0 * * * *`

### Manual Workflow Control

Use the deadline-aware workflow for manual control:

1. Go to **Actions** → **Tournament Deadline-Aware Forecasting**
2. Click **Run workflow**
3. Choose scheduling mode:
   - `normal`: 4-hour frequency
   - `critical`: 2-hour frequency
   - `final_24h`: 1-hour frequency
4. Optionally set custom frequency in hours

## 🔧 Local Development

### Setup Local Environment

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your API keys:

   ```bash
   # Required API Keys
   METACULUS_TOKEN=your_metaculus_token_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here

   # Optional API Keys (using dummy values)
   PERPLEXITY_API_KEY=dummy_perplexity_key
   EXA_API_KEY=dummy_exa_key
   # ... etc
   ```

### Run Locally

Use the provided script to run the bot locally:

```bash
# Run tournament mode
./scripts/run-local.sh

# Run quarterly cup mode
./scripts/run-local.sh quarterly_cup
```

## 🛡️ Security Best Practices

### ✅ What's Secure

- API keys are stored as GitHub secrets (encrypted)
- `.env` file is in `.gitignore` (won't be committed)
- Fallback values for optional services
- API key validation before running

### ❌ Never Do This

- Don't commit API keys to the repository
- Don't share your `.env` file
- Don't put secrets in code or configuration files

## 🔄 Adding More API Keys Later

When you get credits for additional services:

1. Add the new secret in GitHub repository settings
2. The workflow will automatically use the real key instead of the fallback
3. No code changes needed!

## 📊 Monitoring

### Workflow Status

- Check the **Actions** tab for workflow execution status
- Each run shows logs and any errors
- Failed runs will show which API keys are missing

### Bot Performance

- Logs are available in each workflow run
- Monitor forecast accuracy and API usage
- Set up alerts for failed runs

## 🆘 Troubleshooting

### Common Issues

1. **"Missing required API keys"**
   - Check that `METACULUS_TOKEN` and `OPENROUTER_API_KEY` are set in GitHub secrets
   - Verify the secret names match exactly

2. **"Workflow not running automatically"**
   - Check that the workflow files are in `.github/workflows/`
   - Verify the cron schedule syntax
   - Ensure the repository has Actions enabled

3. **"API rate limits exceeded"**
   - The bot has built-in rate limiting
   - Consider adjusting the schedule frequency
   - Monitor API usage in the logs

### Getting Help

- Check workflow logs in the Actions tab
- Review the API key validation step
- Ensure all required dependencies are installed

## 🎯 Next Steps

1. **Set up the GitHub secrets** using the values provided above
2. **Test the workflows** manually to ensure they work
3. **Monitor the automatic runs** to verify everything is working
4. **Add more API keys** as you get credits for additional services

Your GitHub Actions workflows are now ready to run! 🚀
