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

st.set_page_config(page_title="Mashroo3k Brain", page_icon="🧠", layout="wide")

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
    
    # Initialize session state for General Helper (SEPARATE session ID)
    if "general_session_id" not in st.session_state:
        st.session_state.general_session_id = generate_session_id()
    
    # Initialize session state for Report Writer
    if "suggested_files" not in st.session_state:
        st.session_state.suggested_files = []
    if "all_files" not in st.session_state:
        st.session_state.all_files = []
    if "report_content" not in st.session_state:
        st.session_state.report_content = ""
    if "selected_sources" not in st.session_state:
        st.session_state.selected_sources = []
    
    # Initialize session state for General Helper
    if "general_messages" not in st.session_state:
        st.session_state.general_messages = []
    
    # Initialize chat type filter for sidebar
    if "sidebar_chat_filter" not in st.session_state:
        st.session_state.sidebar_chat_filter = "rag"  # Default to Business Data
    
    # Sidebar - User Info & Chat History
    with st.sidebar:
        st.write(f"Welcome, **{name}**")
        authenticator.logout('Logout', 'main')
        st.divider()
        
        # --- Chat Type Selector (Segmented Control) ---
        st.markdown("### 📂 Chat Type")
        chat_type_option = st.radio(
            "Select chat type:",
            options=["💼 Business Data", "🤖 General Assistant"],
            index=0 if st.session_state.sidebar_chat_filter == "rag" else 1,
            horizontal=False,
            label_visibility="collapsed"
        )
        
        # Update filter based on selection
        if chat_type_option == "💼 Business Data":
            st.session_state.sidebar_chat_filter = "rag"
        else:
            st.session_state.sidebar_chat_filter = "general"
        
        st.divider()
        
        # --- New Chat Button (Primary Action) ---
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            if st.session_state.sidebar_chat_filter == "rag":
                st.session_state.current_session_id = generate_session_id()
                st.session_state.messages = []
            else:
                st.session_state.general_session_id = generate_session_id()
                st.session_state.general_messages = []
            st.rerun()
        
        st.divider()
        
        # --- Your History Section ---
        st.subheader("🕒 Your History")
        
        # Fetch user's sessions filtered by chat type
        user_sessions = get_user_sessions(username, chat_type=st.session_state.sidebar_chat_filter)
        
        if user_sessions:
            for session in user_sessions:
                session_id = session['session_id']
                
                # Determine if this is the current session based on chat type
                if st.session_state.sidebar_chat_filter == "rag":
                    is_current = session_id == st.session_state.current_session_id
                else:
                    is_current = session_id == st.session_state.general_session_id
                
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
                        # Load this session based on chat type
                        if st.session_state.sidebar_chat_filter == "rag":
                            st.session_state.current_session_id = session_id
                            # Fetch messages for this session
                            session_messages = get_session_messages(username, session_id)
                            st.session_state.messages = []
                            for msg in session_messages:
                                if msg['role'] == 'user':
                                    st.session_state.messages.append(HumanMessage(content=msg['content']))
                                else:
                                    st.session_state.messages.append(AIMessage(content=msg['content']))
                        else:  # general
                            st.session_state.general_session_id = session_id
                            # Fetch messages for general chat session
                            session_messages = get_session_messages(username, session_id)
                            st.session_state.general_messages = []
                            for msg in session_messages:
                                if msg['role'] == 'user':
                                    st.session_state.general_messages.append(HumanMessage(content=msg['content']))
                                else:
                                    st.session_state.general_messages.append(AIMessage(content=msg['content']))
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
        .report-output {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #e9ecef;
        }
        </style>
    """, unsafe_allow_html=True)

    # Check for API Key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OpenAI API Key missing! Please set it in .env file.")
        st.stop()

    # Initialize RAG System (Cached to avoid reloading on every rerun)
    # Version hash forces cache refresh when rag.py changes
    @st.cache_resource(hash_funcs={type: lambda _: None}, ttl=3600)
    def get_expert(_cache_version="v7"):
        return MarketExpert()

    try:
        expert = get_expert()
    except Exception as e:
        st.error(f"Error initializing Knowledge Bank: {e}")
        st.stop()

    # ========================================
    # TABS: Data Chat | Report Writer | General Helper
    # ========================================
    tab_chat, tab_writer, tab_general = st.tabs(["💬 Data Chat", "📝 Report Writer", "🤖 General Helper"])
    
    # ========================================
    # TAB 1: CHAT MODE
    # ========================================
    with tab_chat:
        # Chat Interface - Display History
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message.content, unsafe_allow_html=True)

        # Standard Sticky Chat Input
        if prompt := st.chat_input("Ask about Saudi market stats, reports, or feasibility studies..."):
            # 1. Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 2. Add to history & DB
            st.session_state.messages.append(HumanMessage(content=prompt))
            save_message(username, st.session_state.current_session_id, 'user', prompt, chat_type='rag')

            # 3. Generate & Stream Response
            with st.chat_message("assistant"):
                try:
                    # Pass history excluding the latest user message
                    history_for_chain = st.session_state.messages[:-1]
                    
                    # Use the STREAMING version for typewriter effect
                    response = expert.process_query_streaming(prompt, history_for_chain)
                    
                    # Get metadata immediately
                    context_docs = response.get("context", [])
                    query_type = response.get("query_type", "KNOWLEDGE_SEARCH")
                    answer_generator = response.get("answer_generator")
                    
                    # Stream the answer
                    full_answer = st.write_stream(answer_generator)
                    
                    # Only show sources if there are documents AND it was a knowledge search
                    if query_type == "KNOWLEDGE_SEARCH" and context_docs:
                        unique_sources = set()
                        for doc in context_docs:
                            source_path = doc.metadata.get("source", "Unknown")
                            if source_path != "Unknown" and source_path != "🌐 Web Search":
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
                            full_answer += "\n\n---\n### 📄 Source Verification\n" + sources_html

                            with st.expander("View Source Snippets"):
                                for doc in context_docs:
                                    source = os.path.basename(doc.metadata.get("source", "Unknown"))
                                    st.caption(f"📍 Source: {source}")
                                    st.text(doc.page_content[:300] + "...")

                    # 4. Add assistant message to history & DB
                    st.session_state.messages.append(AIMessage(content=full_answer))
                    save_message(username, st.session_state.current_session_id, 'assistant', full_answer, chat_type='rag')
                    
                    # 5. Rerun to refresh UI and position input at bottom
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # ========================================
    # TAB 2: REPORT WRITER MODE
    # ========================================
    with tab_writer:
        st.markdown("### 📝 Report Writer")
        st.markdown("Generate structured reports from your knowledge base with smart source selection.")
        
        st.divider()
        
        # --- Step 1: Topic Input ---
        col_topic, col_find = st.columns([4, 1])
        
        with col_topic:
            topic_input = st.text_input(
                "📋 Report Topic",
                placeholder="e.g., Feasibility Study for Coffee Shop in Riyadh",
                key="report_topic"
            )
        
        with col_find:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacer for alignment
            find_clicked = st.button("🔍 Find Files", use_container_width=True, type="secondary")
        
        # --- Handle Find Files Button ---
        if find_clicked and topic_input:
            with st.spinner("🔍 Finding relevant files..."):
                suggested = expert.suggest_files(topic_input, k=5)
                st.session_state.suggested_files = suggested
                
                # Also load all files for the multiselect
                if not st.session_state.all_files:
                    st.session_state.all_files = expert.get_all_indexed_files()
                
                # THE FIX: Auto-populate the multiselect widget
                # Filter to ensure only valid files are selected
                valid_files = [f for f in suggested if f in st.session_state.all_files]
                st.session_state.selected_sources = valid_files
                
            if suggested:
                st.success(f"✅ Found {len(suggested)} relevant files!")
                st.rerun()  # Refresh UI to show selected files
            else:
                st.warning("⚠️ No files found matching this topic. Try different keywords.")
        
        # --- Step 2: File Selection ---
        st.markdown("#### 📁 Select Source Files")
        
        # Load all files if not loaded
        if not st.session_state.all_files:
            try:
                st.session_state.all_files = expert.get_all_indexed_files()
            except:
                st.session_state.all_files = []
        
        # Multiselect with smart defaults (controlled via session_state.selected_sources)
        selected_files = st.multiselect(
            "Choose files to use as sources:",
            options=st.session_state.all_files,
            help="Click 'Find Files' to auto-suggest relevant files, or manually select.",
            key="selected_sources"
        )
        
        if selected_files:
            st.info(f"📄 {len(selected_files)} file(s) selected")
        
        st.divider()
        
        # --- Step 3: Generate Report ---
        col_gen, col_clear = st.columns([3, 1])
        
        with col_gen:
            generate_clicked = st.button(
                "✨ Generate Report",
                use_container_width=True,
                type="primary",
                disabled=not topic_input or not selected_files
            )
        
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.report_content = ""
                st.session_state.suggested_files = []
                st.rerun()
        
        # --- Generate Report ---
        if generate_clicked:
            st.markdown("---")
            st.markdown("### 📄 Generated Report")
            
            report_placeholder = st.empty()
            
            with report_placeholder.container():
                # Stream the report
                report_generator = expert.generate_report_streaming(topic_input, selected_files)
                full_report = st.write_stream(report_generator)
                
                # Store for download
                st.session_state.report_content = full_report
        
        # --- Show Previous Report (if exists) ---
        elif st.session_state.report_content:
            st.markdown("---")
            st.markdown("### 📄 Generated Report")
            st.markdown(st.session_state.report_content)
        
        # --- Download Button ---
        if st.session_state.report_content:
            st.divider()
            col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
            with col_dl2:
                st.download_button(
                    label="📥 Download Report (Markdown)",
                    data=st.session_state.report_content,
                    file_name=f"report_{topic_input[:30].replace(' ', '_')}.md" if topic_input else "report.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    # ========================================
    # TAB 3: GENERAL HELPER (ChatGPT-Style UX)
    # ========================================
    with tab_general:
        # Display General Helper chat history
        for message in st.session_state.general_messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message.content)
        
        # Standard Sticky Chat Input
        if prompt := st.chat_input("Message General Assistant...", key="general_chat_input"):
            # 1. Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 2. Add to history & DB
            st.session_state.general_messages.append(HumanMessage(content=prompt))
            save_message(username, st.session_state.general_session_id, 'user', prompt, chat_type='general')
            
            # 3. Generate & Stream Response
            with st.chat_message("assistant"):
                try:
                    # Use General Helper streaming (no RAG)
                    # Pass history excluding the latest user message
                    response_generator = expert.general_chat_streaming(
                        prompt, 
                        st.session_state.general_messages[:-1]
                    )
                    
                    # Stream the response
                    full_response = st.write_stream(response_generator)
                    
                    # 4. Add to history & DB
                    st.session_state.general_messages.append(AIMessage(content=full_response))
                    save_message(username, st.session_state.general_session_id, 'assistant', full_response, chat_type='general')
                    
                    # 5. Rerun to refresh UI and position input at bottom
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
