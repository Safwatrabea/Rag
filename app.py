import streamlit as st
import os
from rag import MarketExpert
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load env variables
load_dotenv()

st.set_page_config(page_title="Saudi Market Knowledge Bank", page_icon="🏦")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🏦 Saudi Market Knowledge Bank")
st.subheader("AI Assistant for Market Reports & Statistics")

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

# Sidebar
with st.sidebar:
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
                            st.text(doc.page_content[:200] + "...")

                # Add assistant message to history
                st.session_state.chat_history.append(AIMessage(content=answer))
            except Exception as e:
                st.error(f"An error occurred: {e}")
