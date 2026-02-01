import streamlit as st
import os
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from rag import MarketExpert
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

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
    
    # Sidebar - User Info & Logout
    with st.sidebar:
        st.write(f"Welcome, **{name}**")
        authenticator.logout('Logout', 'main')
        st.divider() # Visual separation
        
        st.header("📚 Knowledge Base")
        st.write("Loaded Documents:")
        if os.path.exists("data"):
            files = os.listdir("data")
            if files:
                for f in files:
                    st.caption(f"📄 {f}")
            else:
                st.caption("No documents found in data/")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("🧠 Mashroo3k Brain")
    st.subheader("Mashroo3k Brain, Database and Knowledge Bank")

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
        rag_chain = expert.get_chain()
    except Exception as e:
        st.error(f"Error initializing Knowledge Bank: {e}")
        st.stop()

    # Chat Interface
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)

    user_input = st.chat_input("Ask about Saudi market stats, reports, or feasibility studies...")

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing market data..."):
                try:
                    response = rag_chain.invoke({
                        "input": user_input,
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Check for sources
                    if "context" in response and response["context"]:
                        with st.expander("View Sources"):
                            for doc in response["context"]:
                                source = doc.metadata.get("source", "Unknown")
                                st.caption(f"📍 Source: {source}")
                                # Clean up source path for display if needed
                                st.text(doc.page_content[:200] + "...")

                    # Add assistant message to history
                    st.session_state.chat_history.append(AIMessage(content=answer))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
