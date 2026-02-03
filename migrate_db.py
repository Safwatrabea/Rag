#!/usr/bin/env python3
"""
Database migration script to add chat_type column to existing conversations.
This script is safe to run multiple times - it will check if migration is needed.
"""

import sqlite3
from datetime import datetime

DB_NAME = "chat_history.db"

def migrate_database():
    """Add chat_type column and set existing records to 'rag'."""
    print("=" * 60)
    print("Database Migration: Adding chat_type Column")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # Check if chat_type column exists
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'chat_type' in columns:
            print("✅ Migration already completed. chat_type column exists.")
            
            # Count records by type
            cursor.execute("SELECT chat_type, COUNT(*) FROM conversations GROUP BY chat_type")
            results = cursor.fetchall()
            for chat_type, count in results:
                print(f"   - {chat_type}: {count} messages")
        else:
            print("🔧 Adding chat_type column...")
            
            # Add the column with default value
            cursor.execute("""
                ALTER TABLE conversations 
                ADD COLUMN chat_type TEXT DEFAULT 'rag' CHECK(chat_type IN ('rag', 'general'))
            """)
            
            # Update existing records to 'rag'
            cursor.execute("""
                UPDATE conversations 
                SET chat_type = 'rag' 
                WHERE chat_type IS NULL
            """)
            
            conn.commit()
            
            # Count migrated records
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total = cursor.fetchone()[0]
            
            print(f"✅ Migration successful!")
            print(f"   - Migrated {total} existing messages to chat_type='rag'")
            print(f"   - New messages will be tagged as 'rag' or 'general'")
            
        # Create index for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_chat_type
            ON conversations(user_username, chat_type, timestamp DESC)
        """)
        conn.commit()
        print("✅ Created performance index on chat_type")
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
        conn.rollback()
    finally:
        conn.close()
    
    print("=" * 60)
    print("Migration Complete!")
    print("=" * 60)

if __name__ == "__main__":
    migrate_database()
