#!/usr/bin/env python3
"""
Core Pipeline Test Script
Tests only the essential components that would cause CI/CD failures.
"""

import subprocess
import sys
import yaml
from pathlib import Path

def test_yaml_syntax():
    """Test YAML syntax for all workflow files."""
    print("🔍 Testing YAML Syntax...")
    workflow_dir = Path(".github/workflows")

    if not workflow_dir.exists():
        print("❌ .github/workflows directory not found")
        return False

    yaml_files = list(workflow_dir.glob("*.yaml")) + list(workflow_dir.glob("*.yml"))

    if not yaml_files:
        print("❌ No YAML workflow files found")
        return False

    all_valid = True
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                yaml.safe_load(f)
            print(f"  ✅ {yaml_file.name}")
        except yaml.YAMLError as e:
            print(f"  ❌ {yaml_file.name}: {e}")
            all_valid = False
        except Exception as e:
            print(f"  ❌ {yaml_file.name}: {e}")
            all_valid = False

    return all_valid

def test_poetry_config():
    """Test Poetry configuration."""
    print("🔍 Testing Poetry Configuration...")

    try:
        result = subprocess.run(
            ["poetry", "check"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print("  ✅ Poetry configuration is valid")
            return True
        else:
            print(f"  ❌ Poetry check failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("  ❌ Poetry check timed out")
        return False
    except Exception as e:
        print(f"  ❌ Poetry check error: {e}")
        return False

def test_python_environment():
    """Test Python environment setup."""
    print("🔍 Testing Python Environment...")

    try:
        result = subprocess.run(
            ["poetry", "run", "python", "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✅ Python environment: {version}")
            return True
        else:
            print(f"  ❌ Python environment test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Python environment error: {e}")
        return False

def test_dependencies():
    """Test that key dependencies are available."""
    print("🔍 Testing Key Dependencies...")

    dependencies = [
        ("pytest", "pytest --version"),
        ("black", "black --version"),
        ("flake8", "flake8 --version"),
        ("mypy", "mypy --version"),
    ]

    all_available = True
    for dep_name, cmd in dependencies:
        try:
            result = subprocess.run(
                ["poetry", "run"] + cmd.split(),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print(f"  ✅ {dep_name} available")
            else:
                print(f"  ❌ {dep_name} not available")
                all_available = False
        except Exception as e:
            print(f"  ❌ {dep_name} error: {e}")
            all_available = False

    return all_available

def test_project_structure():
    """Test that required project structure exists."""
    print("🔍 Testing Project Structure...")

    required_paths = [
        "pyproject.toml",
        "poetry.lock",
        "src/",
        "tests/",
        ".github/workflows/",
    ]

    all_exist = True
    for path in required_paths:
        if Path(path).exists():
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path} missing")
            all_exist = False

    return all_exist

def main():
    """Run core pipeline tests."""
    print("🚀 Core Pipeline Tests")
    print("=" * 50)

    tests = [
        ("YAML Syntax", test_yaml_syntax),
        ("Poetry Configuration", test_poetry_config),
        ("Python Environment", test_python_environment),
        ("Key Dependencies", test_dependencies),
        ("Project Structure", test_project_structure),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} - ERROR: {e}")

    print("\n" + "=" * 50)
    print("📊 CORE PIPELINE TEST RESULTS")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\n🎉 All core pipeline tests passed!")
        print("The CI/CD pipeline should work correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} core tests failed.")
        print("These issues may cause CI/CD pipeline failures.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
