import streamlit as st
import os
import yaml
import urllib.parse
import html
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from rag import MarketExpert
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from database import (
    generate_session_id, 
    save_message, 
    get_user_sessions, 
    get_session_messages,
    delete_session
)

# Load env variables
load_dotenv()

st.set_page_config(page_title="Mashroo3k Brain", page_icon="🧠")

# --- Authentication Logic ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Render the login widget
# The login() method returns the name, authentication_status, and username
authenticator.login('main')

if st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')
elif st.session_state['authentication_status']:
    name = st.session_state['name']
    username = st.session_state['username']
    # --- Main Application (Only visible if logged in) ---
    
    # Initialize session state for chat
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = generate_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar - User Info & Chat History
    with st.sidebar:
        st.write(f"Welcome, **{name}**")
        authenticator.logout('Logout', 'main')
        st.divider()
        
        # --- New Chat Button (Primary Action) ---
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.current_session_id = generate_session_id()
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # --- Your History Section ---
        st.subheader("🕒 Your History")
        
        # Fetch user's sessions (lightweight - only metadata)
        user_sessions = get_user_sessions(username)
        
        if user_sessions:
            for session in user_sessions:
                session_id = session['session_id']
                is_current = session_id == st.session_state.current_session_id
                
                # Create a container for each session
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    # Display session as clickable button
                    button_label = f"{'🔵 ' if is_current else ''}{session['title']}"
                    if st.button(
                        button_label, 
                        key=f"session_{session_id}",
                        use_container_width=True,
                        disabled=is_current
                    ):
                        # Load this session
                        st.session_state.current_session_id = session_id
                        # Fetch messages for this session
                        session_messages = get_session_messages(username, session_id)
                        st.session_state.messages = []
                        for msg in session_messages:
                            if msg['role'] == 'user':
                                st.session_state.messages.append(HumanMessage(content=msg['content']))
                            else:
                                st.session_state.messages.append(AIMessage(content=msg['content']))
                        st.rerun()
                
                with col2:
                    # Delete button (only for non-current sessions)
                    if not is_current:
                        if st.button("🗑️", key=f"del_{session_id}", help="Delete this chat"):
                            delete_session(username, session_id)
                            st.rerun()
                
                # Show date below the session
                st.caption(f"   📅 {session['date']}")
        else:
            st.caption("No previous chats yet. Start a new conversation!")

    st.title("🧠 Mashroo3k Brain")
    st.subheader("Mashroo3k Brain, Database and Knowledge Bank")

    # Global CSS for source badges
    st.markdown("""
        <style>
        .source-badge {
            text-decoration: none; 
            padding: 6px 14px; 
            background-color: rgba(255, 75, 75, 0.08); 
            color: #ff4b4b !important; 
            border: 1px solid rgba(255, 75, 75, 0.15); 
            border-radius: 20px; 
            font-size: 0.85rem; 
            font-weight: 500; 
            display: inline-flex; 
            align-items: center; 
            gap: 8px;
            margin: 4px;
            transition: all 0.2s ease;
        }
        .source-badge:hover {
            background-color: rgba(255, 75, 75, 0.15);
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border-color: rgba(255, 75, 75, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)

    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OpenAI API Key missing! Please set it in .env file.")
        st.stop()

    # Initialize RAG System (Cached to avoid reloading on every rerun)
    @st.cache_resource
    def get_expert():
        return MarketExpert()

    try:
        expert = get_expert()
    except Exception as e:
        st.error(f"Error initializing Knowledge Bank: {e}")
        st.stop()

    # Chat Interface
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content, unsafe_allow_html=True)

    user_input = st.chat_input("Ask about Saudi market stats, reports, or feasibility studies...")

    if user_input:
        # Add user message to history
        st.session_state.messages.append(HumanMessage(content=user_input))
        # Save user message to database
        save_message(username, st.session_state.current_session_id, 'user', user_input)
        
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing market data..."):
                try:
                    # Pass history excluding the latest user message to avoid duplication in prompt
                    history_for_chain = st.session_state.messages[:-1]
                    
                    # Use the new process_query method with Query Router
                    response = expert.process_query(user_input, history_for_chain)
                    answer = response["answer"]
                    context_docs = response.get("context", [])
                    query_type = response.get("query_type", "KNOWLEDGE_SEARCH")
                    
                    st.markdown(answer)
                    
                    # Only show sources if there are documents AND it was a knowledge search
                    if query_type == "KNOWLEDGE_SEARCH" and context_docs:
                        unique_sources = set()
                        for doc in context_docs:
                            source_path = doc.metadata.get("source", "Unknown")
                            if source_path != "Unknown":
                                unique_sources.add(os.path.basename(source_path))
                        
                        if unique_sources:
                            st.markdown("---")
                            st.markdown("### 📄 Source Verification")
                            
                            # Display sources as clickable badges
                            sources_html = '<div style="display: flex; flex-wrap: wrap; margin-bottom: 15px;">'
                            for filename in sorted(unique_sources):
                                encoded_filename = urllib.parse.quote(filename)
                                file_url = f"/app/static/data/{encoded_filename}"
                                sources_html += f'<a href="{file_url}" target="_blank" class="source-badge">📄 Open Source: {html.escape(filename)}</a>'
                            sources_html += '</div>'
                            
                            st.markdown(sources_html, unsafe_allow_html=True)
                            
                            # Append sources to the answer for persistence
                            answer += "\n\n---\n### 📄 Source Verification\n" + sources_html

                            with st.expander("View Source Snippets"):
                                for doc in context_docs:
                                    source = os.path.basename(doc.metadata.get("source", "Unknown"))
                                    st.caption(f"📍 Source: {source}")
                                    st.text(doc.page_content[:300] + "...")

                    # Add assistant message to history
                    st.session_state.messages.append(AIMessage(content=answer))
                    # Save assistant message to database
                    save_message(username, st.session_state.current_session_id, 'assistant', answer)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
