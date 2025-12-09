#!/usr/bin/env python
"""View the embedding debug log"""
import os

log_file = os.path.join(os.path.dirname(__file__), 'project', 'embedding_debug.log')

if not os.path.exists(log_file):
    print(f"❌ No log file found: {log_file}")
    print("\n💡 Run a workflow generation test first to generate the log.")
    exit(1)

print(f"📖 Reading: {log_file}\n")
print("="*80)

with open(log_file, 'r', encoding='utf-8') as f:
    content = f.read()
    print(content)

print("="*80)
print(f"\n✅ End of log ({os.path.getsize(log_file)} bytes)")
