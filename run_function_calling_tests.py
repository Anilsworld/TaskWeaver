#!/usr/bin/env python
"""
Quick test runner for workflow function calling tests.

Usage:
    python run_function_calling_tests.py
    
Or with pytest directly:
    pytest tests/unit_tests/test_workflow_function_calling.py -v
"""

import subprocess
import sys

def run_tests():
    """Run the workflow function calling tests."""
    print("=" * 80)
    print("🧪 Running Workflow Function Calling Tests")
    print("=" * 80)
    print()
    
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit_tests/test_workflow_function_calling.py",
                "-v",
                "--tb=short",
                "--color=yes"
            ],
            cwd="C:\\xTrac-AI-Apps\\xtrac-app-api\\TaskWeaver",
            capture_output=False,
            text=True
        )
        
        print()
        print("=" * 80)
        if result.returncode == 0:
            print("✅ ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED!")
        print("=" * 80)
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
