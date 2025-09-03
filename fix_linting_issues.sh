#!/bin/bash

# Script to fix common shellcheck linting issues across GitHub workflows

set -e

echo "🔧 Fixing linting issues in GitHub workflows..."

# Function to fix variable quoting issues (SC2086)
fix_quoting_issues() {
    local file="$1"
    echo "  📝 Fixing quoting issues in $file"

    # Fix common unquoted variable patterns
    sed -i.bak -E 's/\$([A-Z_]+)( )/"\$\1"\2/g' "$file"
    sed -i.bak -E 's/\$\{([A-Z_a-z0-9._]+)\}([^A-Za-z0-9_])/"\${\1}"\2/g' "$file"

    # Fix specific patterns found in the workflows
    sed -i.bak 's/exit $EXIT_CODE/exit "$EXIT_CODE"/g' "$file"
    sed -i.bak 's/-v $METACULUS_TOKEN/-v "$METACULUS_TOKEN"/g' "$file"
    sed -i.bak 's/-v $OPENROUTER_API_KEY/-v "$OPENROUTER_API_KEY"/g' "$file"
    sed -i.bak 's/echo $SPENT/echo "$SPENT"/g' "$file"
    sed -i.bak 's/echo $BUDGET/echo "$BUDGET"/g' "$file"
    sed -i.bak 's/echo $REMAINING/echo "$REMAINING"/g' "$file"
    sed -i.bak 's/echo $PERCENTAGE/echo "$PERCENTAGE"/g' "$file"
    sed -i.bak 's/echo $THRESHOLD/echo "$THRESHOLD"/g' "$file"
    sed -i.bak 's/echo $COST/echo "$COST"/g' "$file"
    sed -i.bak 's/echo $COST_ESTIMATE/echo "$COST_ESTIMATE"/g' "$file"
    sed -i.bak 's/echo $TOURNAMENT_ID/echo "$TOURNAMENT_ID"/g' "$file"
    sed -i.bak 's/echo $TOURNAMENT_SLUG/echo "$TOURNAMENT_SLUG"/g' "$file"
    sed -i.bak 's/echo $WORKFLOW_STATUS/echo "$WORKFLOW_STATUS"/g' "$file"
    sed -i.bak 's/echo $NETWORK_CONFIG/echo "$NETWORK_CONFIG"/g' "$file"
    sed -i.bak 's/echo $RETRY_CONFIG/echo "$RETRY_CONFIG"/g' "$file"
    sed -i.bak 's/echo $TIMEOUT_CONFIG/echo "$TIMEOUT_CONFIG"/g' "$file"

    # Remove backup file
    rm -f "$file.bak"
}

# Function to fix output redirection patterns (SC2129)
fix_redirection_patterns() {
    local file="$1"
    echo "  📝 Fixing redirection patterns in $file"

    # This is more complex and needs manual review for each case
    # For now, we'll add a comment noting the improvement opportunity
    if grep -q "SC2129" <<< "$(shellcheck "$file" 2>/dev/null || true)"; then
        echo "  ⚠️ Note: $file has redirection patterns that could be optimized"
    fi
}

# Function to fix unused variable warnings (SC2034)
fix_unused_variables() {
    local file="$1"
    echo "  📝 Checking unused variables in $file"

    # Look for SPENT variable that appears unused but is actually used in echo
    if grep -q "SPENT=.*BUDGET" "$file" && ! grep -q 'echo.*SPENT' "$file"; then
        echo "  ⚠️ Note: $file may have unused SPENT variable - needs manual review"
    fi
}

# Process all workflow files
for workflow in .github/workflows/*.{yaml,yml}; do
    if [[ -f "$workflow" ]]; then
        echo "🔍 Processing $workflow"
        fix_quoting_issues "$workflow"
        fix_redirection_patterns "$workflow"
        fix_unused_variables "$workflow"
    fi
done

echo "✅ Linting fixes applied!"
echo "📋 Running actionlint to verify improvements..."

# Run actionlint to show remaining issues
if command -v ./bin/actionlint >/dev/null 2>&1; then
    ./bin/actionlint -color || echo "Still some issues remaining - may need manual fixes"
else
    echo "⚠️ actionlint not found - install it to verify fixes"
fi

echo "🎯 Manual review needed for:"
echo "  - SC2129: Consider using { cmd1; cmd2; } >> file instead of individual redirects"
echo "  - SC2034: Unused variables (may be false positives)"
echo "  - SC2016: Single vs double quotes for variable expansion"
