#!/usr/bin/env python
"""Clear the embedding debug log before a new test run"""
import os

log_file = os.path.join(os.path.dirname(__file__), 'project', 'embedding_debug.log')

if os.path.exists(log_file):
    os.remove(log_file)
    print(f"✅ Cleared: {log_file}")
else:
    print(f"ℹ️  No log file found: {log_file}")

print("\n🚀 Ready for new workflow generation test!")
print("📝 Results will be written to: project/embedding_debug.log")
