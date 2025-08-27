#!/bin/bash

# Script to help configure tournament scheduling for the AI Forecasting Bot
# This script provides guidance on setting up GitHub repository variables

echo "⏰ Tournament Scheduling Configuration"
echo "====================================="
echo ""

echo "This script helps you configure tournament scheduling variables in GitHub."
echo "These variables control how frequently the bot runs forecasts."
echo ""

echo "📋 SCHEDULING CONFIGURATION OPTIONS:"
echo "-----------------------------------"
echo ""

echo "1. NORMAL OPERATION (Recommended for budget efficiency):"
echo "   SCHEDULING_FREQUENCY_HOURS: 4"
echo "   - Runs every 4 hours"
echo "   - Optimized for $100 budget"
echo "   - Suitable for seasonal tournaments"
echo ""

echo "2. ACTIVE MONITORING (For competitive periods):"
echo "   SCHEDULING_FREQUENCY_HOURS: 2"
echo "   - Runs every 2 hours"
echo "   - Higher budget usage"
echo "   - Better responsiveness to market changes"
echo ""

echo "3. CONSERVATIVE MODE (For budget preservation):"
echo "   SCHEDULING_FREQUENCY_HOURS: 6"
echo "   - Runs every 6 hours"
echo "   - Minimal budget usage"
echo "   - Suitable for long tournaments"
echo ""

echo "📅 DEADLINE-AWARE SCHEDULING:"
echo "-----------------------------"
echo "Enable automatic frequency adjustment near question deadlines:"
echo ""
echo "DEADLINE_AWARE_SCHEDULING: true"
echo "CRITICAL_PERIOD_FREQUENCY_HOURS: 2  # Last few days"
echo "FINAL_24H_FREQUENCY_HOURS: 1        # Final 24 hours"
echo ""

echo "🎯 TOURNAMENT SCOPE CONFIGURATION:"
echo "---------------------------------"
echo "TOURNAMENT_SCOPE: seasonal"
echo "- Optimized for Fall 2025 AI Benchmark Tournament"
echo "- Expects 50-100 questions total (not per day)"
echo "- Budget planning for full tournament duration"
echo ""

echo "⚙️  HOW TO CONFIGURE:"
echo "--------------------"
echo "1. Go to your GitHub repository"
echo "2. Navigate to: Settings → Secrets and variables → Actions"
echo "3. Click on the 'Variables' tab"
echo "4. Click 'New repository variable'"
echo "5. Add each variable name and value from your chosen configuration above"
echo ""

echo "🔧 MANUAL WORKFLOW CONTROL:"
echo "---------------------------"
echo "You can also override scheduling for individual runs:"
echo "1. Go to Actions → 'Forecast on new AI tournament questions'"
echo "2. Click 'Run workflow'"
echo "3. Set custom frequency and tournament mode"
echo "4. Available modes: normal, critical, final_24h"
echo ""

echo "📊 MONITORING YOUR CONFIGURATION:"
echo "--------------------------------"
echo "Each workflow run will display your current scheduling configuration."
echo "Check the workflow logs to verify your settings are applied correctly."
echo ""

echo "💡 RECOMMENDATIONS:"
echo "------------------"
echo "• Start with 4-hour frequency for budget efficiency"
echo "• Enable deadline-aware scheduling for competitive advantage"
echo "• Use manual runs for testing or urgent updates"
echo "• Monitor budget usage and adjust frequency as needed"
echo ""

echo "✅ Configuration complete! Your tournament scheduling is now optimized."
