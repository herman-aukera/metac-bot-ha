#!/usr/bin/env python3
"""
Repository cleanup script - move unused/experimental files to NOISE folder.
This implements the decluttering that was requested but not completed.
"""

import os
import shutil
from pathlib import Path

def move_to_noise(file_path: str, reason: str):
    """Move file to NOISE folder with logging."""
    src = Path(file_path)
    if not src.exists():
        print(f"Skip (not found): {file_path}")
        return

    noise_dir = Path("NOISE")
    noise_dir.mkdir(exist_ok=True)

    # Preserve directory structure in NOISE
    if src.is_file():
        rel_path = src.relative_to(".")
        dst = noise_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"Moved: {file_path} → NOISE/{rel_path} ({reason})")
    else:
        dst = noise_dir / src.name
        shutil.move(str(src), str(dst))
        print(f"Moved: {file_path} → NOISE/{src.name} ({reason})")

def cleanup_repo():
    """Clean up repository by moving unused files to NOISE."""

    print("=== REPOSITORY CLEANUP ===")
    print("Moving unused/experimental files to NOISE folder...\n")

    # 1. Development/testing scripts that are no longer needed
    cleanup_files = [
        ("debug_missing_questions.py", "debug script"),
        ("reset_circuit_breaker.py", "utility script"),
        ("openrouter_setup_guide.md", "outdated setup guide"),
        ("PIPELINE_STATUS_RESOLVED.md", "resolved status doc"),
        ("USAGE_MAP_COMPREHENSIVE.md", "replaced by README"),
        ("USAGE_MAP.md", "replaced by README"),
        ("TOURNAMENT_READINESS_REPORT.json", "old report"),
        ("deployment_readiness_report.json", "old report"),
        ("seasonal_test.log", "old test log"),
        ("minibench_test.log", "old test log"),
        ("run_pid.txt", "temporary file"),
        ("configs/", "unused config alternatives"),
        ("workflows/", "experimental workflows"),
        ("temp/", "temporary files"),
    ]

    # 2. Old documentation that's been superseded
    docs_to_move = [
        "docs/README_UPDATED.md",
        "docs/IMPLEMENTATION_STATUS.md",
        "docs/IMPLEMENTATION_STATUS_FINAL.md",
        "docs/ROBUST_PYTHON_EXECUTION.md",
        "docs/EMERGENCY_DEPLOYMENT.md",
    ]

    # 3. Example files that clutter the repo
    if Path("examples").exists():
        example_files = list(Path("examples").glob("*_demo.py"))
        for f in example_files:
            cleanup_files.append((str(f), "demo file"))

    # 4. Old monitoring files
    monitoring_files = [
        "monitoring/nginx",
        "htmlcov/",
    ]

    # Execute cleanup
    all_files = cleanup_files + [(f, "old docs") for f in docs_to_move] + [(f, "monitoring") for f in monitoring_files]

    moved_count = 0
    for file_path, reason in all_files:
        if Path(file_path).exists():
            move_to_noise(file_path, reason)
            moved_count += 1

    print(f"\n=== CLEANUP COMPLETE ===")
    print(f"Moved {moved_count} files/directories to NOISE folder")
    print("Repository is now cleaner and focused on essential files.")

    # 5. Update .gitignore to ignore NOISE
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        content = gitignore_path.read_text()
        if "NOISE/" not in content:
            with open(".gitignore", "a") as f:
                f.write("\n# Cleanup folder\nNOISE/\n")
            print("Added NOISE/ to .gitignore")

if __name__ == "__main__":
    cleanup_repo()
