"""
Database module for persistent chat history using SQLite.
"""
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

DB_NAME = "chat_history.db"


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize the database and create tables if they don't exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_username TEXT NOT NULL,
                session_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL
            )
        """)
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_session 
            ON conversations(user_username, session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_timestamp 
            ON conversations(user_username, timestamp DESC)
        """)
        conn.commit()


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


def save_message(user_username: str, session_id: str, role: str, content: str) -> int:
    """
    Save a single message to the database.
    
    Args:
        user_username: The username of the logged-in user
        session_id: The unique session/chat ID
        role: Either 'user' or 'assistant'
        content: The message content
    
    Returns:
        The ID of the inserted message
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (user_username, session_id, role, content)
            VALUES (?, ?, ?, ?)
        """, (user_username, session_id, role, content))
        conn.commit()
        return cursor.lastrowid


def get_user_sessions(user_username: str) -> List[Dict]:
    """
    Get all chat sessions for a user (lightweight - only session metadata).
    Returns sessions with their first message preview and timestamp.
    
    Args:
        user_username: The username of the logged-in user
    
    Returns:
        List of session dictionaries with id, title, and timestamp
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Get unique sessions with their first message and latest timestamp
        cursor.execute("""
            WITH SessionInfo AS (
                SELECT 
                    session_id,
                    MIN(timestamp) as first_timestamp,
                    MAX(timestamp) as last_timestamp
                FROM conversations
                WHERE user_username = ?
                GROUP BY session_id
            ),
            FirstMessages AS (
                SELECT 
                    c.session_id,
                    c.content as first_message
                FROM conversations c
                INNER JOIN SessionInfo si ON c.session_id = si.session_id
                WHERE c.user_username = ?
                  AND c.role = 'user'
                  AND c.timestamp = (
                      SELECT MIN(timestamp) 
                      FROM conversations 
                      WHERE session_id = c.session_id 
                        AND user_username = ?
                        AND role = 'user'
                  )
            )
            SELECT 
                si.session_id,
                COALESCE(fm.first_message, 'New Chat') as first_message,
                si.first_timestamp,
                si.last_timestamp
            FROM SessionInfo si
            LEFT JOIN FirstMessages fm ON si.session_id = fm.session_id
            ORDER BY si.last_timestamp DESC
        """, (user_username, user_username, user_username))
        
        sessions = []
        for row in cursor.fetchall():
            # Create a title from the first message (truncate if too long)
            first_msg = row['first_message']
            title = first_msg[:40] + "..." if len(first_msg) > 40 else first_msg
            
            # Format the timestamp nicely
            try:
                ts = datetime.fromisoformat(row['first_timestamp'])
                date_str = ts.strftime("%Y-%m-%d %H:%M")
            except:
                date_str = row['first_timestamp'][:16] if row['first_timestamp'] else "Unknown"
            
            sessions.append({
                'session_id': row['session_id'],
                'title': title,
                'date': date_str,
                'last_activity': row['last_timestamp']
            })
        
        return sessions


def get_session_messages(user_username: str, session_id: str) -> List[Dict]:
    """
    Get all messages for a specific session (only when session is selected).
    
    Args:
        user_username: The username of the logged-in user (for security)
        session_id: The session ID to fetch messages for
    
    Returns:
        List of message dictionaries with role and content
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content, timestamp
            FROM conversations
            WHERE user_username = ? AND session_id = ?
            ORDER BY timestamp ASC
        """, (user_username, session_id))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'role': row['role'],
                'content': row['content'],
                'timestamp': row['timestamp']
            })
        
        return messages


def delete_session(user_username: str, session_id: str) -> bool:
    """
    Delete all messages from a specific session.
    
    Args:
        user_username: The username of the logged-in user (for security)
        session_id: The session ID to delete
    
    Returns:
        True if deletion was successful
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM conversations
            WHERE user_username = ? AND session_id = ?
        """, (user_username, session_id))
        conn.commit()
        return cursor.rowcount > 0


def get_session_count(user_username: str) -> int:
    """
    Get the count of sessions for a user.
    
    Args:
        user_username: The username of the logged-in user
    
    Returns:
        Number of unique sessions
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(DISTINCT session_id) as count
            FROM conversations
            WHERE user_username = ?
        """, (user_username,))
        row = cursor.fetchone()
        return row['count'] if row else 0


# Initialize database on module import
init_database()
