#!/usr/bin/env python3
"""
Test script to verify the General Helper UI/UX overhaul is working correctly.
"""

import sqlite3
from database import get_user_sessions, save_message, generate_session_id

DB_NAME = "chat_history.db"

def test_database_schema():
    """Test that chat_type column exists and has correct values."""
    print("\n" + "="*60)
    print("Testing Database Schema")
    print("="*60)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Test 1: Check chat_type column exists
    cursor.execute("PRAGMA table_info(conversations)")
    columns = [col[1] for col in cursor.fetchall()]
    assert 'chat_type' in columns, "❌ chat_type column not found!"
    print("✅ Test 1 PASSED: chat_type column exists")
    
    # Test 2: Check chat_type values are valid
    cursor.execute("SELECT DISTINCT chat_type FROM conversations")
    chat_types = [row[0] for row in cursor.fetchall()]
    for ct in chat_types:
        assert ct in ['rag', 'general', None], f"❌ Invalid chat_type: {ct}"
    print(f"✅ Test 2 PASSED: Valid chat_types found: {chat_types}")
    
    # Test 3: Check index exists
    cursor.execute("PRAGMA index_list(conversations)")
    indexes = [idx[1] for idx in cursor.fetchall()]
    assert 'idx_user_chat_type' in indexes, "❌ Performance index not found!"
    print("✅ Test 3 PASSED: Performance index idx_user_chat_type exists")
    
    conn.close()
    print("="*60)
    print("✅ All Database Tests Passed!")
    print("="*60)


def test_save_and_retrieve():
    """Test saving and retrieving messages with different chat types."""
    print("\n" + "="*60)
    print("Testing Save and Retrieve Functions")
    print("="*60)
    
    test_user = "test_user"
    
    # Test 1: Save RAG message
    rag_session = generate_session_id()
    save_message(test_user, rag_session, 'user', 'Test RAG message', chat_type='rag')
    print("✅ Test 1 PASSED: Saved RAG message")
    
    # Test 2: Save General message
    general_session = generate_session_id()
    save_message(test_user, general_session, 'user', 'Test General message', chat_type='general')
    print("✅ Test 2 PASSED: Saved General message")
    
    # Test 3: Retrieve RAG sessions only
    rag_sessions = get_user_sessions(test_user, chat_type='rag')
    rag_session_ids = [s['session_id'] for s in rag_sessions]
    assert rag_session in rag_session_ids, "❌ RAG session not found in RAG filter!"
    assert general_session not in rag_session_ids, "❌ General session found in RAG filter!"
    print(f"✅ Test 3 PASSED: RAG filter works ({len(rag_sessions)} RAG sessions)")
    
    # Test 4: Retrieve General sessions only
    general_sessions = get_user_sessions(test_user, chat_type='general')
    general_session_ids = [s['session_id'] for s in general_sessions]
    assert general_session in general_session_ids, "❌ General session not found in General filter!"
    assert rag_session not in general_session_ids, "❌ RAG session found in General filter!"
    print(f"✅ Test 4 PASSED: General filter works ({len(general_sessions)} General sessions)")
    
    # Test 5: Retrieve all sessions (no filter)
    all_sessions = get_user_sessions(test_user, chat_type=None)
    all_session_ids = [s['session_id'] for s in all_sessions]
    assert rag_session in all_session_ids, "❌ RAG session not in all sessions!"
    assert general_session in all_session_ids, "❌ General session not in all sessions!"
    print(f"✅ Test 5 PASSED: No filter returns all sessions ({len(all_sessions)} total)")
    
    # Cleanup test data
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE user_username = ?", (test_user,))
    conn.commit()
    conn.close()
    print("🧹 Cleaned up test data")
    
    print("="*60)
    print("✅ All Save/Retrieve Tests Passed!")
    print("="*60)


def test_statistics():
    """Show statistics about the current database."""
    print("\n" + "="*60)
    print("Database Statistics")
    print("="*60)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Total messages
    cursor.execute("SELECT COUNT(*) FROM conversations")
    total = cursor.fetchone()[0]
    print(f"📊 Total messages: {total}")
    
    # Messages by chat type
    cursor.execute("SELECT chat_type, COUNT(*) FROM conversations GROUP BY chat_type")
    for chat_type, count in cursor.fetchall():
        ct_name = "Business Data" if chat_type == 'rag' else "General Assistant"
        print(f"   - {ct_name} ({chat_type}): {count}")
    
    # Unique sessions
    cursor.execute("SELECT COUNT(DISTINCT session_id) FROM conversations")
    sessions = cursor.fetchone()[0]
    print(f"📊 Total unique sessions: {sessions}")
    
    # Sessions by chat type
    cursor.execute("SELECT chat_type, COUNT(DISTINCT session_id) FROM conversations GROUP BY chat_type")
    for chat_type, count in cursor.fetchall():
        ct_name = "Business Data" if chat_type == 'rag' else "General Assistant"
        print(f"   - {ct_name} sessions: {count}")
    
    conn.close()
    print("="*60)


if __name__ == "__main__":
    print("\n🔬 Starting General Helper Upgrade Tests...\n")
    
    try:
        test_database_schema()
        test_save_and_retrieve()
        test_statistics()
        
        print("\n" + "🎉"*30)
        print("ALL TESTS PASSED! General Helper upgrade is working correctly!")
        print("🎉"*30 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
