#!/bin/bash

# Automated fix script for GitHub workflow linting issues
# Fixes the most common shellcheck issues: SC2086 (unquoted variables)

set -e

echo "🔧 Starting automated linting fixes..."

# List of workflow files to process
WORKFLOW_FILES=(
  ".github/workflows/budget_monitoring.yaml"
  ".github/workflows/ci-cd.yml"
  ".github/workflows/cost_tracking_integration.yaml"
  ".github/workflows/emergency-deployment.yaml"
  ".github/workflows/network-resilience-config.yaml"
  ".github/workflows/run_bot_on_minibench.yaml"
  ".github/workflows/run_bot_on_quarterly_cup.yaml"
  ".github/workflows/run_bot_on_tournament.yaml"
  ".github/workflows/test_bot_resilient.yaml"
  ".github/workflows/test_deployment.yaml"
  ".github/workflows/tournament-ready-cicd.yml"
  ".github/workflows/tournament_deadline_aware.yaml"
  ".github/workflows/workflow_management.yaml"
)

# Function to fix common SC2086 patterns
fix_sc2086_patterns() {
    local file="$1"
    echo "  📝 Fixing SC2086 patterns in $(basename "$file")"

    # Create backup
    cp "$file" "$file.backup"

    # Fix $GITHUB_ENV references
    sed -i.tmp 's/>> $GITHUB_ENV/>> "$GITHUB_ENV"/g' "$file"
    sed -i.tmp 's/>> $GITHUB_OUTPUT/>> "$GITHUB_OUTPUT"/g' "$file"

    # Fix exit code patterns
    sed -i.tmp 's/exit $EXIT_CODE/exit "$EXIT_CODE"/g' "$file"
    sed -i.tmp 's/exit $exit_code/exit "$exit_code"/g' "$file"

    # Fix common variable echo patterns
    sed -i.tmp 's/\$PY_RUN /"\$PY_RUN" /g' "$file"
    sed -i.tmp 's/echo $\([A-Z_][A-Z_0-9]*\)/echo "$\1"/g' "$file"

    # Fix conditional patterns with variables
    sed -i.tmp 's/\[ $\([A-Z_a-z][A-Z_a-z0-9]*\) /[ "$\1" /g' "$file"

    # Fix assignment patterns
    sed -i.tmp 's/echo "\([^"]*\)=\$\([A-Z_][A-Z_0-9]*\)"/echo "\1=$\2"/g' "$file"
    sed -i.tmp 's/echo "\([^"]*\)=\$\([A-Z_][A-Z_0-9]*\)" >>/echo "\1=$\2" >>/g' "$file"

    # Cleanup temp file
    rm -f "$file.tmp"
}

# Function to fix style patterns SC2129 (redirections)
fix_sc2129_patterns() {
    local file="$1"
    echo "  📝 Checking SC2129 patterns in $(basename "$file")"

    # This is complex to automate, so we'll just note files that need manual review
    if grep -q '^[[:space:]]*echo.*>>' "$file"; then
        echo "    ⚠️ Manual review needed: Multiple echo >> redirections found"
    fi
}

# Process each workflow file
for file in "${WORKFLOW_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "🔍 Processing $file"
        fix_sc2086_patterns "$file"
        fix_sc2129_patterns "$file"
        echo "  ✅ Completed $file"
    else
        echo "  ⚠️ File not found: $file"
    fi
done

echo ""
echo "🧪 Verifying fixes with actionlint..."

# Run actionlint to check improvements
if [[ -x "./bin/actionlint" ]]; then
    BEFORE_COUNT=$(./bin/actionlint -color 2>&1 | grep -c "SC2086" || echo "0")
    echo "🔍 Remaining SC2086 issues: $BEFORE_COUNT"

    if [[ "$BEFORE_COUNT" -eq 0 ]]; then
        echo "🎉 All SC2086 issues fixed!"
    else
        echo "📋 Remaining SC2086 issues:"
        ./bin/actionlint -color 2>&1 | grep "SC2086" | head -5
    fi

    # Check for other issue types
    TOTAL_ISSUES=$(./bin/actionlint -color 2>&1 | grep -E "(error|warning|info|style)" | wc -l || echo "0")
    echo "📊 Total remaining linting issues: $TOTAL_ISSUES"

else
    echo "⚠️ actionlint not found - cannot verify fixes"
fi

echo ""
echo "🎯 Summary:"
echo "✅ Automated fixes applied for SC2086 (unquoted variables)"
echo "⚠️ Manual review still needed for:"
echo "  - SC2129: Optimize redirection patterns"
echo "  - SC2034: Remove unused variables"
echo "  - SC2016: Fix quote expansion issues"
echo ""
echo "💡 Backup files created with .backup extension"
