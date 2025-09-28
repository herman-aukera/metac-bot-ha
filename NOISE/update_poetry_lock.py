#!/usr/bin/env python3
"""
Update poetry.lock file to match current pyproject.toml
"""
import subprocess
import sys
import os

def update_poetry_lock():
    """Update the poetry.lock file."""
    try:
        # Try to update the lock file
        result = subprocess.run([
            sys.executable, "-m", "poetry", "lock", "--no-update"
        ], capture_output=True, text=True, cwd=os.getcwd())

        if result.returncode == 0:
            print("✅ Poetry lock file updated successfully")
            return True
        else:
            print(f"❌ Poetry lock update failed: {result.stderr}")

            # Try alternative approach
            print("🔄 Trying alternative approach...")
            result2 = subprocess.run([
                "poetry", "lock", "--no-update"
            ], capture_output=True, text=True, cwd=os.getcwd())

            if result2.returncode == 0:
                print("✅ Poetry lock file updated successfully (alternative)")
                return True
            else:
                print(f"❌ Alternative approach failed: {result2.stderr}")
                return False

    except Exception as e:
        print(f"💥 Error updating poetry lock: {e}")
        return False

if __name__ == "__main__":
    success = update_poetry_lock()
    sys.exit(0 if success else 1)
