#!/usr/bin/env python
"""Clear the embedding debug log before a new test run"""
import os

log_file = os.path.join(os.path.dirname(__file__), 'project', 'embedding_debug.log')

if os.path.exists(log_file):
    os.remove(log_file)
    print(f"✅ Cleared: {log_file}")
else:
    print(f"ℹ️  No log file found: {log_file}")

print("\n" + "="*80)
print("🚀 Ready for new workflow generation test!")
print("="*80)
print("\n📝 Results will be written to: project/embedding_debug.log")
print("\n💡 After generating workflow, run:")
print("   python view_embedding_debug.py")
print("\n🎯 What to check:")
print("   - Query should focus on task (e.g., 'search for flights')")
print("   - Query should NOT have meta-language (e.g., 'generate workflow')")
print("   - Matched actions should be domain-relevant (e.g., SKYSCANNER, not WRIKE)")
print("="*80)
