#!/bin/bash
# Quick script to manually trigger the tournament workflow
# Usage: ./trigger_reforecast.sh

set -e

echo "🎯 Triggering tournament workflow for re-forecasting..."
echo ""
echo "Option 1: GitHub CLI (if authenticated)"
echo "  gh auth login  # First time only"
echo "  gh workflow run run_bot_on_tournament.yaml --ref main"
echo ""
echo "Option 2: GitHub Actions UI (RECOMMENDED)"
echo "  1. Go to: https://github.com/herman-aukera/metac-bot-ha/actions"
echo "  2. Click 'Run Bot on Tournament' (left sidebar)"
echo "  3. Click 'Run workflow' button (right side)"
echo "  4. Select branch: main"
echo "  5. Click green 'Run workflow' button"
echo ""
echo "📊 Monitor run:"
echo "  https://github.com/herman-aukera/metac-bot-ha/actions"
echo ""
echo "⚠️  REMEMBER: After successful run, restore SKIP_PREVIOUSLY_FORECASTED=true"
echo "  in .github/workflows/run_bot_on_tournament.yaml (line 162)"
echo ""

# Try to open Actions page in browser
if command -v open &> /dev/null; then
    read -p "Open GitHub Actions page in browser? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "https://github.com/herman-aukera/metac-bot-ha/actions/workflows/run_bot_on_tournament.yaml"
        echo "✅ Opened in browser"
    fi
fi
