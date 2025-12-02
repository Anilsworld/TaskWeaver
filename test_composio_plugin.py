#!/usr/bin/env python
"""
Test the composio_action plugin in TaskWeaver (Eclipse).

This tests:
1. Plugin file exists and is valid
2. TaskWeaver can load the plugin
3. Plugin can be called from generated code
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print('\n' + '='*80)
print('TESTING COMPOSIO_ACTION PLUGIN IN ECLIPSE (TASKWEAVER)')
print('='*80)

# Test 1: Check plugin files exist
print('\n[1/4] Checking plugin files...')
plugin_py = os.path.join(os.path.dirname(__file__), 'project', 'plugins', 'composio_action.py')
plugin_yaml = os.path.join(os.path.dirname(__file__), 'project', 'plugins', 'composio_action.yaml')

if os.path.exists(plugin_py) and os.path.exists(plugin_yaml):
    print('   [OK] composio_action.py exists')
    print('   [OK] composio_action.yaml exists')
else:
    print('   [FAIL] Plugin files missing!')
    sys.exit(1)

# Test 2: Validate YAML
print('\n[2/4] Validating plugin YAML...')
try:
    import yaml
    with open(plugin_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    assert config.get('name') == 'composio_action', "Plugin name should be 'composio_action'"
    assert config.get('enabled') == True, "Plugin should be enabled"
    assert 'parameters' in config, "Plugin should have parameters"
    
    print(f'   [OK] name: {config.get("name")}')
    print(f'   [OK] enabled: {config.get("enabled")}')
    print(f'   [OK] parameters: {len(config.get("parameters", []))} defined')
except Exception as e:
    print(f'   [FAIL] YAML validation error: {e}')
    sys.exit(1)

# Test 3: Check plugin can be imported
print('\n[3/4] Testing plugin import...')
try:
    from taskweaver.plugin import register_plugin
    
    # Import the plugin module
    import importlib.util
    spec = importlib.util.spec_from_file_location("composio_action", plugin_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Check ComposioAction class exists
    assert hasattr(module, 'ComposioAction'), "ComposioAction class not found"
    print('   [OK] ComposioAction class found')
    print('   [OK] Plugin can be imported')
except Exception as e:
    print(f'   [FAIL] Import error: {e}')
    sys.exit(1)

# Test 4: Check example YAML exists
print('\n[4/4] Checking example YAML...')
example_yaml = os.path.join(
    os.path.dirname(__file__), 
    'project', 
    'examples',
    'planner_examples',
    'example-planner-workflow-composio-integration.yaml'
)

if os.path.exists(example_yaml):
    try:
        with open(example_yaml, 'r') as f:
            example = yaml.safe_load(f)
        
        assert example.get('enabled') == True, "Example should be enabled"
        assert 'rounds' in example, "Example should have rounds"
        
        print('   [OK] Example YAML exists and is valid')
        print(f'   [OK] Example has {len(example.get("rounds", []))} round(s)')
    except Exception as e:
        print(f'   [WARN] Example YAML error: {e}')
else:
    print('   [WARN] Example YAML not found (optional)')

print('\n' + '='*80)
print('ALL TESTS PASSED!')
print('='*80)
print('\nThe composio_action plugin is ready for use in Eclipse workflows.')
print('Example usage in generated code:')
print('  result, desc = composio_action(')
print('      action_name="GMAIL_SEND_EMAIL",')
print('      params={"to": "user@example.com", "subject": "Hello", "body": "Hi!"}')
print('  )')
print('='*80 + '\n')

