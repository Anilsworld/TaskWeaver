#!/usr/bin/env python
"""Test the Composio plugin in TaskWeaver."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from taskweaver.app.app import TaskWeaverApp

print('\n' + '='*80)
print('TESTING COMPOSIO PLUGIN IN TASKWEAVER')
print('='*80)

# Initialize TaskWeaver app
print('\n[1/3] Initializing TaskWeaver...')
app = TaskWeaverApp(app_dir="./project")
print('   ✅ TaskWeaver initialized')

# Create a session
print('\n[2/3] Creating session...')
session = app.get_session()
print('   ✅ Session created')

# Test the plugin
print('\n[3/3] Testing Composio plugin...')
test_queries = [
    "use composio to find tools for sending email",
    "search composio for calendar actions",
]

for i, query in enumerate(test_queries, 1):
    print(f'\n   Test {i}: "{query}"')
    try:
        response = session.send_message(query)
        
        # Check if plugin was used
        if 'composio' in str(response).lower():
            print(f'      ✅ Plugin recognized')
        else:
            print(f'      ⚠️  Plugin may not have been used')
        
        # Show response snippet
        response_text = str(response)[:200]
        print(f'      Response: {response_text}...')
    except Exception as e:
        print(f'      ❌ Error: {e}')

print('\n' + '='*80)
print('TEST COMPLETE')
print('='*80 + '\n')

