#!/bin/bash

# Script to verify that the migration captured all important files
# Run this after migration to double-check nothing important was missed

set -e

ORIGINAL_DIR="$1"
NEW_REPO_DIR="$2"

if [ -z "$ORIGINAL_DIR" ] || [ -z "$NEW_REPO_DIR" ]; then
    echo "Usage: $0 <original_dir> <new_repo_dir>"
    echo "Example: $0 ../metac-agent-ha ../metac-agent-agent"
    exit 1
fi

if [ ! -d "$ORIGINAL_DIR" ] || [ ! -d "$NEW_REPO_DIR" ]; then
    echo "❌ One or both directories don't exist"
    exit 1
fi

echo "🔍 Verifying migration from $ORIGINAL_DIR to $NEW_REPO_DIR"
echo ""

# Function to check if important files exist
check_important_files() {
    local dir="$1"
    local label="$2"

    echo "📋 Checking important files in $label:"

    # Core Python files
    important_files=(
        "main.py"
        "main_agent.py"
        "pyproject.toml"
        "poetry.lock"
        "Dockerfile"
        "Makefile"
    )

    for file in "${important_files[@]}"; do
        if [ -f "$dir/$file" ]; then
            echo "  ✅ $file"
        else
            echo "  ❌ $file (MISSING)"
        fi
    done

    # Important directories
    important_dirs=(
        "src"
        "tests"
        "configs"
        "scripts"
        "infrastructure"
        ".github"
        ".kiro"
    )

    echo ""
    echo "📁 Checking important directories in $label:"
    for dir_name in "${important_dirs[@]}"; do
        if [ -d "$dir/$dir_name" ]; then
            file_count=$(find "$dir/$dir_name" -type f | wc -l)
            echo "  ✅ $dir_name/ ($file_count files)"
        else
            echo "  ❌ $dir_name/ (MISSING)"
        fi
    done
}

# Check both directories
check_important_files "$ORIGINAL_DIR" "ORIGINAL"
echo ""
check_important_files "$NEW_REPO_DIR" "NEW REPO"

echo ""
echo "📊 File count comparison:"
original_py_count=$(find "$ORIGINAL_DIR" -name "*.py" | grep -v __pycache__ | wc -l)
new_py_count=$(find "$NEW_REPO_DIR" -name "*.py" | grep -v __pycache__ | wc -l)

original_test_count=$(find "$ORIGINAL_DIR/tests" -name "*.py" 2>/dev/null | wc -l)
new_test_count=$(find "$NEW_REPO_DIR/tests" -name "*.py" 2>/dev/null | wc -l)

echo "  Python files: $original_py_count (original) vs $new_py_count (new)"
echo "  Test files: $original_test_count (original) vs $new_test_count (new)"

# Check for critical tournament optimization files
echo ""
echo "🏆 Checking tournament optimization specific files:"
tournament_files=(
    "src/application/use_cases/process_tournament_question.py"
    "src/application/services/forecasting_pipeline.py"
    "src/application/services/tournament_service.py"
    "src/application/services/integration_service.py"
    "src/presentation/cli.py"
    "src/presentation/rest_api.py"
    "tests/integration/test_complete_tournament_orchestration.py"
)

for file in "${tournament_files[@]}"; do
    if [ -f "$NEW_REPO_DIR/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (CRITICAL - MISSING!)"
    fi
done

# Check for any Python files that might have been missed
echo ""
echo "🔍 Looking for Python files that might have been missed..."
missed_files=()

while IFS= read -r -d '' file; do
    rel_path="${file#$ORIGINAL_DIR/}"
    if [ ! -f "$NEW_REPO_DIR/$rel_path" ] && [[ "$rel_path" != *"__pycache__"* ]] && [[ "$rel_path" != *".git/"* ]]; then
        missed_files+=("$rel_path")
    fi
done < <(find "$ORIGINAL_DIR" -name "*.py" -print0)

if [ ${#missed_files[@]} -eq 0 ]; then
    echo "  ✅ All Python files appear to be copied"
else
    echo "  ⚠️  Found ${#missed_files[@]} potentially missed Python files:"
    for file in "${missed_files[@]}"; do
        echo "    - $file"
    done
fi

# Summary
echo ""
echo "📋 MIGRATION SUMMARY:"
if [ $new_py_count -ge $((original_py_count - 5)) ]; then
    echo "  ✅ Python file count looks good ($new_py_count/$original_py_count)"
else
    echo "  ⚠️  Significant difference in Python file count ($new_py_count/$original_py_count)"
fi

if [ $new_test_count -ge $((original_test_count - 2)) ]; then
    echo "  ✅ Test file count looks good ($new_test_count/$original_test_count)"
else
    echo "  ⚠️  Significant difference in test file count ($new_test_count/$original_test_count)"
fi

# Check if all tournament optimization files are present
missing_tournament_files=0
for file in "${tournament_files[@]}"; do
    if [ ! -f "$NEW_REPO_DIR/$file" ]; then
        missing_tournament_files=$((missing_tournament_files + 1))
    fi
done

if [ $missing_tournament_files -eq 0 ]; then
    echo "  ✅ All critical tournament optimization files present"
else
    echo "  ❌ $missing_tournament_files critical tournament optimization files missing!"
fi

echo ""
if [ $missing_tournament_files -eq 0 ] && [ ${#missed_files[@]} -lt 5 ]; then
    echo "🎉 Migration looks successful! You should be good to proceed."
else
    echo "⚠️  Migration may have issues. Review the missing files above."
fi
