import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from qdrant_client import QdrantClient


load_dotenv()

QDRANT_PATH = "qdrant_db"
COLLECTION_NAME = "saudi_market_knowledge"


class QueryRouter:
    """
    Routes user queries to either GENERAL_CHAT or KNOWLEDGE_SEARCH.
    Prevents unnecessary vector searches for casual conversation.
    Uses heuristic pre-checks + LLM classification for accuracy.
    """
    
    # Heuristic indicators that FORCE search mode (bypass LLM classification)
    FORCE_SEARCH_MARKERS = ['?', '؟']  # Question marks (English and Arabic)
    MIN_WORDS_FOR_SEARCH = 3  # If more than 3 words, likely a real question
    
    # Arabic slang/terms that indicate a substantive question (not greeting)
    BUSINESS_SLANG_TERMS = [
        # Egyptian Business Slang
        'البيزنس', 'بيزنس', 'غايب', 'ماشي', 'ازاي', 'إزاي', 'ليه', 'ليش', 'عايز', 'عاوز',
        'فلوس', 'مصاري', 'تكلفة', 'سعر', 'كام', 'قد ايه', 'قديش',
        # Saudi Business Terms
        'ايش', 'وش', 'كيف', 'علوم', 'السوق', 'المشروع', 'الدراسة', 'جدوى',
        'ربح', 'خسارة', 'رأس مال', 'استثمار', 'عميل', 'زبون',
        # Common question words
        'هل', 'ما', 'ماذا', 'أين', 'متى', 'لماذا', 'كم', 'من',
        # Business terms
        'roi', 'feasibility', 'market', 'analysis', 'study', 'report', 'cost', 'revenue'
    ]
    
    ROUTER_PROMPT = """You are a query classifier for a Saudi Market Knowledge Base system.
Analyze the user's message and classify it into one of two categories:

1. GENERAL_CHAT: ONLY for pure greetings, thanks, or farewells with NO information request.
   Examples: "Hello", "Hi", "Thanks!", "شكراً", "مرحبا", "السلام عليكم", "Bye", "Ok", "تمام"
   
2. KNOWLEDGE_SEARCH: ANY message that asks a question, requests information, or discusses business topics.
   This includes:
   - Direct questions (with or without question marks)
   - Egyptian slang: "البيزنس صاحبه غايب؟", "الموضوع ماشي ازاي؟", "عايز اعرف"
   - Saudi dialect: "ايش الوضع؟", "علوم السوق", "كيف التكلفة؟"
   - Business discussions: ROI, feasibility, costs, revenue, market analysis
   - Follow-up questions in a conversation
   
IMPORTANT RULES:
- If the message contains a question mark (? or ؟), classify as KNOWLEDGE_SEARCH
- If the message has more than 3 words, bias towards KNOWLEDGE_SEARCH
- If the message contains business/market terms in any dialect, classify as KNOWLEDGE_SEARCH
- When in doubt, classify as KNOWLEDGE_SEARCH (better to search than miss a question)

Respond with ONLY one word: either "GENERAL_CHAT" or "KNOWLEDGE_SEARCH"

User message: {input}"""

    CHAT_RESPONSES = [
        "مرحباً! كيف يمكنني مساعدتك اليوم في بيانات السوق أو دراسات الجدوى؟",
        "أهلاً وسهلاً! هل لديك أسئلة حول السوق السعودي؟",
        "تحت أمرك! اسألني عن أي دراسة جدوى أو تحليل سوقي.",
    ]
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.router_prompt = ChatPromptTemplate.from_template(self.ROUTER_PROMPT)
    
    def _has_question_marker(self, text: str) -> bool:
        """Check if text contains question marks."""
        return any(marker in text for marker in self.FORCE_SEARCH_MARKERS)
    
    def _is_long_enough(self, text: str) -> bool:
        """Check if text has more than MIN_WORDS_FOR_SEARCH words."""
        # Split on whitespace and filter empty strings
        words = [w for w in text.split() if w.strip()]
        return len(words) > self.MIN_WORDS_FOR_SEARCH
    
    def _contains_business_terms(self, text: str) -> bool:
        """Check if text contains known business/slang terms."""
        text_lower = text.lower()
        return any(term in text_lower for term in self.BUSINESS_SLANG_TERMS)
    
    def classify(self, user_input: str) -> str:
        """
        Classify the user input as GENERAL_CHAT or KNOWLEDGE_SEARCH.
        Uses heuristic pre-checks before LLM classification.
        Returns the classification string.
        """
        # HEURISTIC PRE-CHECKS: Force SEARCH mode if any condition is met
        if self._has_question_marker(user_input):
            return "KNOWLEDGE_SEARCH"
        
        if self._is_long_enough(user_input):
            return "KNOWLEDGE_SEARCH"
        
        if self._contains_business_terms(user_input):
            return "KNOWLEDGE_SEARCH"
        
        # If heuristics don't trigger, use LLM classification
        try:
            chain = self.router_prompt | self.llm
            result = chain.invoke({"input": user_input})
            classification = result.content.strip().upper()
            
            # Ensure we return a valid classification
            if "GENERAL_CHAT" in classification:
                return "GENERAL_CHAT"
            else:
                # Default to knowledge search if unclear
                return "KNOWLEDGE_SEARCH"
        except Exception as e:
            print(f"Router error: {e}. Defaulting to KNOWLEDGE_SEARCH.")
            return "KNOWLEDGE_SEARCH"
    
    def get_chat_response(self) -> str:
        """Return a friendly chat response for general conversation."""
        import random
        return random.choice(self.CHAT_RESPONSES)


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
        
        # Initialize the Query Router
        self.router = QueryRouter(self.llm)

    def get_chain(self):
        """Returns the RAG chain for knowledge search queries."""
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
            
            "LANGUAGE UNDERSTANDING:\n"
            "- You understand Egyptian Arabic slang (e.g., 'البيزنس', 'ماشي ازاي', 'عايز اعرف')\n"
            "- You understand Saudi Arabic dialect (e.g., 'ايش الوضع', 'علوم السوق', 'كيف التكلفة')\n"
            "- You understand formal Arabic and English business terminology\n\n"
            
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
            
            "CRITICAL FALLBACK RULES:\n"
            "- If the {context} does not contain the specific answer the user asked for, you MUST respond with:\n"
            "  'بحثت في الملفات المتاحة ولم أجد معلومة محددة عن [mention the topic they asked about]. "
            "هل تقصد ملفاً أو تقريراً محدداً؟ أو هل تريد معلومات عامة عن هذا الموضوع؟'\n"
            "- NEVER respond with a generic greeting like 'Welcome! How can I help?' when the user has asked a specific question.\n"
            "- If documents are retrieved but irrelevant, acknowledge you searched but didn't find matching data.\n"
            "- Always stay in 'helpful assistant' mode, never switch to 'greeting' mode mid-conversation.\n\n"
            
            "CONTEXT AWARENESS:\n"
            "- Pay attention to the chat_history - if there are previous messages, you are CONTINUING a conversation.\n"
            "- In an ongoing conversation, NEVER start with greetings or 'How can I help you?'\n"
            "- Reference previous context when relevant.\n\n"
            
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
    
    def process_query(self, user_input: str, chat_history: list) -> dict:
        """
        Main entry point for processing user queries.
        Routes to either chat response or knowledge search based on query type.
        
        Returns:
            dict with keys:
                - "answer": The response text
                - "context": List of source documents (empty for chat)
                - "query_type": "GENERAL_CHAT" or "KNOWLEDGE_SEARCH"
        """
        # Step 1: Classify the query
        query_type = self.router.classify(user_input)
        
        if query_type == "GENERAL_CHAT":
            # Skip retrieval, return friendly chat response
            return {
                "answer": self.router.get_chat_response(),
                "context": [],  # No sources for chat
                "query_type": "GENERAL_CHAT"
            }
        else:
            # Proceed with full RAG pipeline
            rag_chain = self.get_chain()
            response = rag_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            return {
                "answer": response["answer"],
                "context": response.get("context", []),
                "query_type": "KNOWLEDGE_SEARCH"
            }
