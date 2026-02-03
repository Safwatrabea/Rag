#!/usr/bin/env python3
"""
Quick test script to verify incremental ingestion state management.
"""

import os
import json
from pathlib import Path

STATE_FILE = "ingestion_state.json"

def test_state_management():
    """Test the state management functions."""
    print("🧪 Testing Incremental Ingestion State Management\n")
    
    # Clean up any existing state file
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
        print("🧹 Cleaned up existing state file")
    
    # Test 1: Load empty state
    print("\n📋 Test 1: Loading empty state...")
    from ingest import load_ingestion_state
    state = load_ingestion_state()
    assert state == {}, "Should return empty dict for missing state file"
    print("✅ PASS: Empty state loaded correctly")
    
    # Test 2: Save state
    print("\n📋 Test 2: Saving state...")
    from ingest import save_ingestion_state
    test_state = {
        "data/test1.pdf": 1234567890.123,
        "data/test2.docx": 1234567891.456
    }
    save_ingestion_state(test_state)
    assert os.path.exists(STATE_FILE), "State file should be created"
    print("✅ PASS: State saved successfully")
    
    # Test 3: Load saved state
    print("\n📋 Test 3: Loading saved state...")
    loaded_state = load_ingestion_state()
    assert loaded_state == test_state, "Loaded state should match saved state"
    print("✅ PASS: State loaded correctly")
    
    # Test 4: should_process_file logic
    print("\n📋 Test 4: Testing should_process_file logic...")
    from ingest import should_process_file
    
    # Create a test file
    test_file = Path("test_file.txt")
    test_file.write_text("test content")
    
    # Case 1: New file (not in state)
    empty_state = {}
    should_process = should_process_file(test_file, empty_state)
    assert should_process == True, "New file should be processed"
    print("  ✅ New file: PROCESS")
    
    # Case 2: Unchanged file
    current_mtime = os.path.getmtime(test_file)
    state_with_file = {str(test_file): current_mtime}
    should_process = should_process_file(test_file, state_with_file)
    assert should_process == False, "Unchanged file should be skipped"
    print("  ✅ Unchanged file: SKIP")
    
    # Case 3: Updated file (simulate by setting old timestamp in state)
    old_state = {str(test_file): current_mtime - 100}  # 100 seconds ago
    should_process = should_process_file(test_file, old_state)
    assert should_process == True, "Updated file should be processed"
    print("  ✅ Updated file: PROCESS")
    
    # Clean up test file
    test_file.unlink()
    print("✅ PASS: All should_process_file tests passed")
    
    # Test 5: JSON structure validation
    print("\n📋 Test 5: Validating JSON structure...")
    with open(STATE_FILE, 'r') as f:
        json_content = json.load(f)
    assert isinstance(json_content, dict), "State should be a dictionary"
    for key, value in json_content.items():
        assert isinstance(key, str), "Keys should be strings"
        assert isinstance(value, (int, float)), "Values should be numbers"
    print("✅ PASS: JSON structure is valid")
    
    print("\n" + "="*60)
    print("🎉 All tests passed! Incremental ingestion is ready to use.")
    print("="*60)
    
    # Clean up
    print("\n🧹 Cleaning up test artifacts...")
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    print("✅ Cleanup complete")

if __name__ == "__main__":
    test_state_management()
