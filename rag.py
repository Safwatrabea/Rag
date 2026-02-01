import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "saudi_market_knowledge"

class MarketExpert:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        qdrant_mode = os.getenv("QDRANT_MODE", "local").lower()
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        if qdrant_mode == "server":
            print(f"Connecting to Qdrant Server at {qdrant_url}...")
            self.client = QdrantClient(url=qdrant_url)
        else:
            print(f"Using Local Qdrant at {QDRANT_PATH}...")
            self.client = QdrantClient(path=QDRANT_PATH)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def get_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # Answer question prompt
        system_prompt = (
            "You are a Senior Strategy Consultant & Market Researcher specializing in the Saudi Market "
            "(Feasibility Studies, Consumer Behavior, Marketing KPIs).\n\n"
            "Constraint #1 (The Evidence): specific project data (e.g., 'Expected ROI', 'Construction Costs', "
            "'Target Audience Demographics', 'Client Name') MUST come strictly from the provided {context}. "
            "Do not invent client data.\n\n"
            "Constraint #2 (The Advisory Value): You ARE allowed to use your internal knowledge to:\n"
            "   * Define Frameworks: Explain concepts mentioned in the text (e.g., define 'TAM/SAM/SOM', 'SWOT Analysis', "
            "'Conversion Rate', 'CAGR').\n"
            "   * Contextualize: If a report mentions a specific Saudi sector (e.g., Coffee Shops), you can briefly "
            "mention general market trends in that sector to add value, BUT clearly distinguish it from the report's "
            "specific findings.\n"
            "   * Structure: Organize answers like a professional consultancy report (Executive Summary -> Key Findings -> Recommendations).\n\n"
            "Fallback Mechanism: If the {context} misses the specific answer, reply: "
            "'عذراً، هذه التفاصيل غير مذكورة في دراسات الجدوى أو التقارير التسويقية المتاحة حالياً.'\n\n"
            "Tone: Insightful, Professional, Advisory. Arabic language priority.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain
