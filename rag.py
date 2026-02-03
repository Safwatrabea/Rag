import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.tools import DuckDuckGoSearchRun
# For production, uncomment and use Tavily instead:
# from langchain_community.tools.tavily_search import TavilySearchResults
from qdrant_client import QdrantClient

# Groq for fast General Helper chat
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ langchain-groq not installed. General Helper will use OpenAI.")

# FlashRank for fast reranking
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    print("⚠️ FlashRank not installed. Reranking disabled. Install with: pip install flashrank")

# Enable logging for MultiQueryRetriever to see generated queries
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)


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
    """
    Hybrid RAG System with Smart Fallback:
    1. ALWAYS searches internal PDFs first (VectorStore)
    2. Falls back to Web Search ONLY if no internal docs found
    3. Keeps source_documents populated for UI compatibility
    
    This uses deterministic retrieval logic instead of an Agent for reliability.
    """
    
    # Unified System Prompt for generation
    SYSTEM_PROMPT = """You are a Senior Strategy Consultant & Market Researcher specializing in the Saudi Market 
(Feasibility Studies, Consumer Behavior, Marketing KPIs).

LANGUAGE UNDERSTANDING:
- You understand Egyptian Arabic slang (e.g., 'البيزنس', 'ماشي ازاي', 'عايز اعرف')
- You understand Saudi Arabic dialect (e.g., 'ايش الوضع', 'علوم السوق', 'كيف التكلفة')
- You understand formal Arabic and English business terminology

RESPONSE RULES:
1. Answer the question based ONLY on the provided Context below.
2. If the context is from INTERNAL REPORTS (📄), cite the report name clearly.
3. If the context is from WEB SEARCH (🌐), mention it's external/current data and include the source.
4. If the context doesn't contain the answer, say: "لم أجد معلومات محددة عن هذا الموضوع في المصادر المتاحة."
5. Respond in the SAME LANGUAGE the user used (Arabic → Arabic, English → English).
6. If data year is not exact, mention: "وفقاً لآخر البيانات المتاحة..."

CITATION FORMAT:
- For internal documents: "📄 المصدر: [document_name]"
- For web search: "🌐 المصدر: [source description]"

Tone: Insightful, Professional, Advisory. Arabic language priority.

=== CONTEXT ===
{context}
=== END CONTEXT ==="""

    # Keywords that indicate web search is more appropriate
    WEB_SEARCH_KEYWORDS = [
        # Time-sensitive terms
        'اليوم', 'الآن', 'حالياً', 'أحدث', 'الحالي', 'today', 'current', 'latest', 'now',
        '2024', '2025', '2026',
        # General knowledge terms
        'عدد سكان', 'population', 'gdp', 'الناتج المحلي',
        # Prices
        'سعر', 'أسعار', 'price', 'prices', 'cost of',
        # Commodities
        'حديد', 'اسمنت', 'نفط', 'بترول', 'steel', 'cement', 'oil', 'petroleum',
        # Regulations
        'قانون', 'لائحة', 'نظام', 'regulation', 'law', 'policy'
    ]

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        qdrant_mode = os.getenv("QDRANT_MODE", "local").lower()
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

        if qdrant_mode == "server":
            print(f"🔗 Connecting to Qdrant Server at {qdrant_url}...")
            self.client = QdrantClient(url=qdrant_url)
        else:
            print(f"📁 Using Local Qdrant at {QDRANT_PATH}...")
            self.client = QdrantClient(path=QDRANT_PATH)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )
        
        # LLM for generation and translation (non-streaming for internal use)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        
        # Streaming LLM for typewriter effect in UI (chat)
        self.llm_streaming = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)
        
        # Low-temperature streaming LLM for Report Writer (high factual adherence)
        self.llm_writer = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, streaming=True)
        
        # Initialize Groq LLM for General Helper (lightning fast!)
        self._setup_groq_llm()
        
        # Initialize the Query Router (for chat vs search classification)
        self.router = QueryRouter(self.llm)
        
        # Initialize Web Search Tool
        self.web_search = DuckDuckGoSearchRun()
        # For production with Tavily (better quality):
        # from langchain_community.tools.tavily_search import TavilySearchResults
        # self.web_search = TavilySearchResults(max_results=5)
        
        # Create Multi-Query Retriever for better document recall
        self._setup_multi_query_retriever()
        
        # Setup FlashRank Reranker for filtering noise
        self._setup_reranker()
        
        print("✅ Hybrid RAG System initialized (MultiQuery + Reranker + VectorStore + Web Fallback)")

    def _setup_reranker(self):
        """
        Setup FlashRank Cross-Encoder Reranker.
        This filters out irrelevant chunks (TOC, headers) before sending to LLM.
        """
        if FLASHRANK_AVAILABLE:
            try:
                # Use the default small model for speed
                # Options: "ms-marco-TinyBERT-L-2-v2" (fastest), "ms-marco-MiniLM-L-12-v2" (balanced)
                self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./flashrank_cache")
                print("🎯 FlashRank Reranker initialized (ms-marco-MiniLM-L-12-v2)")
            except Exception as e:
                print(f"⚠️ FlashRank initialization failed: {e}. Reranking disabled.")
                self.reranker = None
        else:
            self.reranker = None

    def _setup_groq_llm(self):
        """
        Setup Groq LLMs with dual-model strategy:
        1. Llama 3.3 70B (Versatile) for RAG & Report Writer - Deep reasoning, stable
        2. Allam 1 13B (Instruct) for General Chat - Saudi-native, fast, conversational
        
        Fallbacks:
        - If Allam is unavailable, use llama-3.1-8b-instant for General Chat
        - If Groq entirely unavailable, use OpenAI
        """
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        self.llm_groq = None       # General Helper (Allam or Llama 3.1)
        self.llm_groq_rag = None   # RAG & Report Writer (Llama 3.3 70B)
        
        if GROQ_AVAILABLE and groq_api_key:
            try:
                # ===== 1. RAG & REPORT WRITER MODEL (Heavy Lifting) =====
                # Use Llama 3.3 70B for deep reasoning and stable reports
                try:
                    self.llm_groq_rag = ChatGroq(
                        api_key=groq_api_key,
                        model_name="llama-3.3-70b-versatile",
                        temperature=0.5,  # Balanced for reports
                        max_tokens=8000,  # High context for detailed reports
                        streaming=True,
                    )
                    print("✅ Groq RAG/Writer LLM initialized (Llama 3.3 70B - Deep Reasoning)")
                except Exception as e:
                    print(f"⚠️ Llama 3.3 70B initialization failed: {e}")
                    print("   Will use OpenAI fallback for reports")
                
                # ===== 2. GENERAL CHAT MODEL (Saudi-Native Custom) =====
                # Try custom Orpheus Arabic Saudi model first (User-specific on Groq)
                try:
                    print("🌙 Attempting to initialize Orpheus Arabic Saudi (Custom Saudi-native model)...")
                    self.llm_groq = ChatGroq(
                        api_key=groq_api_key,
                        model_name="canopylabs/orpheus-arabic-saudi",  # Custom model ID
                        temperature=0.7,  # More creative/conversational
                        streaming=True,
                    )
                    print("✅ Groq General Chat LLM initialized (Orpheus Arabic Saudi - Custom)")
                    
                except Exception as e:
                    # Fallback to Llama 3.1 8B if custom model not accessible
                    print(f"⚠️ Custom Orpheus model initialization failed: {e}")
                    print("🔄 Falling back to Llama 3.1 8B Instant for General Chat...")
                    try:
                        self.llm_groq = ChatGroq(
                            api_key=groq_api_key,
                            model_name="llama-3.1-8b-instant",
                            temperature=0.7,
                            streaming=True,
                        )
                        print("✅ Groq General Chat LLM initialized (Llama 3.1 8B Instant - Fallback)")
                    except Exception as e2:
                        print(f"⚠️ Llama 3.1 fallback also failed: {e2}")
                        print("   Will use OpenAI GPT-4o-mini for General Chat")
                
            except Exception as e:
                print(f"⚠️ Groq initialization failed: {e}. Using OpenAI fallback.")
        else:
            if not GROQ_AVAILABLE:
                print("⚠️ langchain-groq not installed. Using OpenAI fallback.")
            elif not groq_api_key:
                print("⚠️ GROQ_API_KEY not set. Using OpenAI fallback.")

    def _setup_multi_query_retriever(self):
        """
        Setup Multi-Query Retriever that generates multiple query variations
        to capture different semantic aspects of the user's question.
        """
        # Base retriever from vector store
        # Retrieve more docs (10) for reranking to filter down to best 5
        base_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Custom prompt for generating query variations
        # Optimized for Arabic and English business queries
        query_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant helping to generate alternative search queries.
Your task is to generate 3 different versions of the user's question to retrieve relevant documents.

RULES:
1. If the original query is in Arabic, generate variations in BOTH Arabic AND English.
2. Think about different ways to phrase the same question.
3. Consider synonyms and related business terms.
4. Each variation should capture a different aspect of the question.

Examples:
- Original: "تكلفة مزرعة دجاج" 
  → "رأس المال المطلوب لمشروع دواجن"
  → "Poultry farm initial investment cost"
  → "تكاليف التشغيل لمزارع الطيور"

- Original: "Chicken farm cost"
  → "Poultry farm initial capital requirements"
  → "Chicken feed and operational expenses"
  → "Equipment and machinery prices for poultry farming"

Return ONLY the 3 alternative queries, one per line. No numbering, no explanations."""),
            ("human", "Generate 3 search query variations for: {question}")
        ])
        
        # Create the multi-query retriever
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
            prompt=query_gen_prompt,
            include_original=True  # Also search with the original query
        )
        
        print("🔄 Multi-Query Retriever configured (generates 3 query variations)")

    def _is_web_search_query(self, query: str) -> bool:
        """Check if the query is better suited for web search based on keywords."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.WEB_SEARCH_KEYWORDS)

    def _translate_to_english(self, arabic_text: str) -> str:
        """Translate Arabic query to English for better web search results."""
        try:
            translation_prompt = ChatPromptTemplate.from_messages([
                ("system", "Translate the following Arabic text to English. Return ONLY the English translation, nothing else."),
                ("human", "{text}")
            ])
            chain = translation_prompt | self.llm
            result = chain.invoke({"text": arabic_text})
            return result.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return arabic_text  # Fallback to original

    def _is_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters."""
        arabic_pattern = any('\u0600' <= char <= '\u06FF' for char in text)
        return arabic_pattern

    def _search_vectorstore(self, query: str, k: int = 5) -> List[Any]:
        """
        Step 1: Search internal PDFs using Multi-Query Retrieval + Reranking.
        
        Pipeline:
        1. Generate multiple query variations (MultiQuery)
        2. Retrieve top 10-15 candidate documents
        3. Deduplicate
        4. Rerank using FlashRank Cross-Encoder
        5. Return top k highest-scored documents
        
        Returns list of Document objects with metadata.
        """
        try:
            print(f"\n🔍 Multi-Query + Rerank Search starting for: {query[:50]}...")
            print("📝 Generating query variations...")
            
            # Step 1: Use Multi-Query Retriever for better recall
            docs = self.multi_query_retriever.invoke(query)
            
            # Step 2: Remove duplicates based on page_content
            seen_content = set()
            unique_docs = []
            for doc in docs:
                content_hash = hash(doc.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            print(f"📚 Retrieved {len(docs)} total docs, {len(unique_docs)} unique after dedup")
            
            # Step 3: Rerank using FlashRank (if available)
            if self.reranker and len(unique_docs) > 0:
                print("🎯 Reranking documents with FlashRank...")
                reranked_docs = self._rerank_documents(query, unique_docs, top_k=k)
                
                if reranked_docs:
                    unique_docs = reranked_docs
                    print(f"✅ Reranked to top {len(unique_docs)} most relevant documents")
                else:
                    print("⚠️ Reranking failed, using original order")
                    unique_docs = unique_docs[:k]
            else:
                # No reranker, just limit to k
                unique_docs = unique_docs[:k]
            
            # Log the final sources
            print("📄 Final documents:")
            for i, doc in enumerate(unique_docs, 1):
                source = doc.metadata.get("source", "Unknown")
                filename = os.path.basename(source) if source else "Unknown"
                score = doc.metadata.get("rerank_score", "N/A")
                print(f"   {i}. {filename} (score: {score})")
            
            return unique_docs
            
        except Exception as e:
            print(f"❌ Multi-Query + Rerank search error: {e}")
            # Fallback to simple similarity search
            print("⚠️ Falling back to simple similarity search...")
            try:
                docs = self.vector_store.similarity_search(query, k=k)
                print(f"📚 Fallback found {len(docs)} documents")
                return docs
            except Exception as e2:
                print(f"❌ Fallback search also failed: {e2}")
                return []

    def _rerank_documents(self, query: str, docs: List[Any], top_k: int = 5) -> List[Any]:
        """
        Rerank documents using FlashRank Cross-Encoder.
        Returns documents sorted by relevance score (highest first).
        """
        if not self.reranker or not docs:
            return docs[:top_k]
        
        try:
            # Prepare passages for reranking
            passages = []
            for i, doc in enumerate(docs):
                passages.append({
                    "id": i,
                    "text": doc.page_content[:1000],  # Limit text length for speed
                    "meta": doc.metadata
                })
            
            # Create rerank request
            rerank_request = RerankRequest(query=query, passages=passages)
            
            # Rerank
            results = self.reranker.rerank(rerank_request)
            
            # Sort by score and get top k
            reranked_docs = []
            for result in results[:top_k]:
                original_doc = docs[result["id"]]
                # Add rerank score to metadata for logging
                original_doc.metadata["rerank_score"] = round(result["score"], 4)
                reranked_docs.append(original_doc)
            
            return reranked_docs
            
        except Exception as e:
            print(f"❌ Reranking error: {e}")
            return docs[:top_k]

    def _search_web(self, query: str) -> Dict[str, Any]:
        """
        Step 2: Web search fallback.
        Translates Arabic queries to English for better results.
        Returns dict with content and source info.
        """
        try:
            # Translate Arabic queries to English for better web results
            search_query = query
            if self._is_arabic(query):
                print(f"🌐 Translating Arabic query for web search...")
                search_query = self._translate_to_english(query)
                print(f"🌐 Translated query: {search_query}")
            
            # Enhance query with context
            enhanced_query = f"{search_query} Saudi Arabia 2024 2025"
            print(f"🌐 Searching web: {enhanced_query[:60]}...")
            
            results = self.web_search.run(enhanced_query)
            
            if results:
                print(f"🌐 Web search returned results")
                return {
                    "content": results,
                    "source": "🌐 Web Search",
                    "source_type": "web"
                }
            else:
                print(f"🌐 Web search returned no results")
                return {"content": "", "source": "", "source_type": "none"}
                
        except Exception as e:
            print(f"❌ Web search error: {e}")
            return {"content": f"Web search error: {str(e)}", "source": "", "source_type": "error"}

    def _format_context(self, docs: List[Any], web_result: Dict[str, Any] = None) -> str:
        """Format retrieved documents and/or web results into context string."""
        context_parts = []
        
        # Add internal documents
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            filename = os.path.basename(source) if source else "Unknown Document"
            context_parts.append(f"📄 Document {i} [{filename}]:\n{doc.page_content}")
        
        # Add web results if available
        if web_result and web_result.get("content"):
            context_parts.append(f"🌐 Web Search Results:\n{web_result['content']}")
        
        if not context_parts:
            return "No relevant information found in available sources."
        
        return "\n\n---\n\n".join(context_parts)

    def _generate_answer(self, query: str, context: str, chat_history: list) -> str:
        """Generate answer using LLM with the provided context (non-streaming)."""
        try:
            # Build the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}")
            ])
            
            # Convert chat history
            formatted_history = self._convert_chat_history(chat_history)
            
            # Create chain and invoke
            chain = prompt | self.llm
            result = chain.invoke({
                "context": context,
                "chat_history": formatted_history,
                "input": query
            })
            
            return result.content
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return f"عذراً، حدث خطأ أثناء إنشاء الإجابة: {str(e)}"

    def _generate_answer_streaming(self, query: str, context: str, chat_history: list):
        """
        Generate answer using streaming LLM (yields chunks for typewriter effect).
        This is a generator that yields string chunks.
        """
        try:
            # Build the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}")
            ])
            
            # Convert chat history
            formatted_history = self._convert_chat_history(chat_history)
            
            # Create chain with streaming LLM
            chain = prompt | self.llm_streaming
            
            # Stream the response
            for chunk in chain.stream({
                "context": context,
                "chat_history": formatted_history,
                "input": query
            }):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            print(f"❌ Streaming generation error: {e}")
            yield f"عذراً، حدث خطأ أثناء إنشاء الإجابة: {str(e)}"

    def _convert_chat_history(self, chat_history: list) -> List:
        """Convert chat history to LangChain message format."""
        messages = []
        for msg in chat_history:
            if isinstance(msg, dict):
                role = msg.get("role", "human")
                content = msg.get("content", "")
                if role == "human" or role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai":
                    messages.append(AIMessage(content=content))
            elif isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role == "human":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))
            elif isinstance(msg, (HumanMessage, AIMessage)):
                messages.append(msg)
        return messages

    def _create_web_source_document(self, web_result: Dict[str, Any]) -> Any:
        """Create a pseudo-document object for web results to maintain UI compatibility."""
        from langchain_core.documents import Document
        return Document(
            page_content=web_result.get("content", ""),
            metadata={
                "source": "🌐 Web Search",
                "source_type": "web"
            }
        )

    def process_query(self, user_input: str, chat_history: list) -> dict:
        """
        Main entry point for processing user queries.
        Uses DETERMINISTIC retrieval logic:
        
        1. ALWAYS search VectorStore first (with original Arabic query)
        2. If no docs found AND query seems web-appropriate, search web
        3. Generate answer with proper source_documents for UI
        
        Returns:
            dict with keys:
                - "answer": The response text
                - "context": List of source documents (populated for UI!)
                - "query_type": "GENERAL_CHAT" or "KNOWLEDGE_SEARCH"
                - "source_type": "internal", "web", "both", or "none"
        """
        print(f"\n{'='*60}")
        print(f"📝 Processing query: {user_input[:80]}...")
        print(f"{'='*60}")
        
        # Step 0: Classify the query (chat vs search)
        query_type = self.router.classify(user_input)
        
        if query_type == "GENERAL_CHAT":
            print("💬 Query classified as GENERAL_CHAT")
            return {
                "answer": self.router.get_chat_response(),
                "context": [],
                "query_type": "GENERAL_CHAT",
                "source_type": "none"
            }
        
        print("🔎 Query classified as KNOWLEDGE_SEARCH")
        
        # Step 1: ALWAYS search VectorStore first (keep original query for Arabic PDF matching)
        docs = self._search_vectorstore(user_input, k=5)
        
        source_type = "none"
        web_result = None
        
        # Step 2: Check if we need web fallback
        if len(docs) == 0:
            print("📭 No internal documents found. Checking if web search is appropriate...")
            
            # Always try web search if no internal docs
            web_result = self._search_web(user_input)
            
            if web_result.get("content"):
                source_type = "web"
                # Create pseudo-document for UI compatibility
                docs = [self._create_web_source_document(web_result)]
        else:
            source_type = "internal"
            
            # Even if we have docs, check if query wants CURRENT data
            if self._is_web_search_query(user_input):
                print("🔄 Query contains time-sensitive keywords. Also searching web...")
                web_result = self._search_web(user_input)
                if web_result.get("content"):
                    source_type = "both"
        
        # Step 3: Format context and generate answer
        context = self._format_context(
            docs if source_type != "web" else [],
            web_result if source_type in ["web", "both"] else None
        )
        
        print(f"📊 Source type: {source_type}")
        print(f"📄 Context length: {len(context)} chars")
        
        # Step 4: Generate the answer
        answer = self._generate_answer(user_input, context, chat_history)
        
        # Return with source_documents populated for UI
        return {
            "answer": answer,
            "context": docs,  # POPULATED for UI buttons!
            "query_type": "KNOWLEDGE_SEARCH",
            "source_type": source_type
        }

    def process_query_streaming(self, user_input: str, chat_history: list) -> dict:
        """
        Streaming version of process_query.
        Returns context/metadata immediately, plus a generator for the answer.
        
        This enables:
        1. Get source documents FIRST (for UI preparation)
        2. Stream the answer with typewriter effect
        3. Display sources AFTER streaming completes
        
        Returns:
            dict with keys:
                - "answer_generator": Generator that yields answer chunks
                - "context": List of source documents (populated immediately!)
                - "query_type": "GENERAL_CHAT" or "KNOWLEDGE_SEARCH"
                - "source_type": "internal", "web", "both", or "none"
        """
        print(f"\n{'='*60}")
        print(f"📝 Processing query (STREAMING): {user_input[:80]}...")
        print(f"{'='*60}")
        
        # Step 0: Classify the query (chat vs search)
        query_type = self.router.classify(user_input)
        
        if query_type == "GENERAL_CHAT":
            print("💬 Query classified as GENERAL_CHAT")
            # For chat, just return a simple generator with the response
            chat_response = self.router.get_chat_response()
            def chat_generator():
                yield chat_response
            return {
                "answer_generator": chat_generator(),
                "context": [],
                "query_type": "GENERAL_CHAT",
                "source_type": "none"
            }
        
        print("🔎 Query classified as KNOWLEDGE_SEARCH")
        
        # Step 1: ALWAYS search VectorStore first
        docs = self._search_vectorstore(user_input, k=5)
        
        source_type = "none"
        web_result = None
        
        # Step 2: Check if we need web fallback
        if len(docs) == 0:
            print("📭 No internal documents found. Checking if web search is appropriate...")
            web_result = self._search_web(user_input)
            
            if web_result.get("content"):
                source_type = "web"
                docs = [self._create_web_source_document(web_result)]
        else:
            source_type = "internal"
            
            if self._is_web_search_query(user_input):
                print("🔄 Query contains time-sensitive keywords. Also searching web...")
                web_result = self._search_web(user_input)
                if web_result.get("content"):
                    source_type = "both"
        
        # Step 3: Format context
        context = self._format_context(
            docs if source_type != "web" else [],
            web_result if source_type in ["web", "both"] else None
        )
        
        print(f"📊 Source type: {source_type}")
        print(f"📄 Context length: {len(context)} chars")
        print("📡 Starting streaming response...")
        
        # Step 4: Create streaming generator for the answer
        answer_generator = self._generate_answer_streaming(user_input, context, chat_history)
        
        # Return immediately with docs and generator
        return {
            "answer_generator": answer_generator,
            "context": docs,  # POPULATED for UI buttons!
            "query_type": "KNOWLEDGE_SEARCH",
            "source_type": source_type
        }

    # ========================================
    # REPORT WRITER MODE
    # ========================================
    
    WRITER_SYSTEM_PROMPT = """# الدور: مستشار دراسات جدوى متخصص

أنت خبير في كتابة دراسات الجدوى والتقارير التفصيلية للسوق السعودي.
مهمتك: كتابة تقرير شامل ومُفصّل وغني بالبيانات عن الموضوع المطلوب.

---

## ⚠️ قواعد صارمة (يجب الالتزام بها):

### القاعدة 1: ممنوع التلخيص!
- لا تختصر أي معلومة
- إذا وجدت 10 نقاط في المصادر، اذكر الـ 10 نقاط كاملة
- التوسع والتفصيل مطلوب، وليس الاختصار

### القاعدة 2: البيانات أولاً!
- استخرج واعرض كل رقم، نسبة مئوية، وقيمة مالية موجودة في المصادر
- إذا وجدت جدولاً في المصادر، أعد إنشائه بالكامل في التقرير
- الأرقام المحددة أهم من العبارات العامة
- مثال: بدلاً من "تكلفة عالية" ← اكتب "تكلفة 2,500,000 ريال سعودي"

### القاعدة 3: هيكل متداخل ومفصل!
- استخدم الترقيم المتداخل: 1. ثم 1.1 ثم 1.1.1
- كل قسم رئيسي يجب أن يحتوي على أقسام فرعية
- أضف عمقاً للتحليل

### القاعدة 4: الطول والتفصيل!
- التقرير يجب أن يكون طويلاً ومفصلاً
- إذا كانت نقطة ما تحتوي على تفاصيل في المصادر، اذكرها جميعاً
- لا تكتفِ بذكر العناوين، بل اشرح كل عنوان بالتفصيل

---

## 📋 الهيكل المطلوب للتقرير:

### 1. الملخص التنفيذي (Executive Summary)
- نظرة عامة شاملة عن المشروع/الموضوع
- أهم المؤشرات والأرقام الرئيسية
- النتيجة النهائية والتوصية

### 2. تحليل السوق (Market Analysis)
2.1 حجم السوق والطلب
2.2 المنافسين والحصص السوقية  
2.3 الاتجاهات والتوقعات المستقبلية
2.4 الفرص والتحديات

### 3. التحليل المالي (Financial Analysis)
3.1 التكاليف الرأسمالية (CAPEX)
    - تفصيل كل بند
3.2 التكاليف التشغيلية (OPEX)
    - تفصيل شهري/سنوي
3.3 الإيرادات المتوقعة
    - السيناريوهات المختلفة
3.4 مؤشرات الربحية
    - ROI, IRR, فترة الاسترداد
    - نقطة التعادل

### 4. المتطلبات والموارد
4.1 الموقع والمساحة
4.2 المعدات والتجهيزات
4.3 الموارد البشرية
4.4 التراخيص والمتطلبات القانونية

### 5. تحليل المخاطر (Risk Analysis)
5.1 المخاطر التشغيلية
5.2 المخاطر المالية
5.3 المخاطر السوقية
5.4 خطط التخفيف

### 6. التوصيات والخطوات التالية
6.1 التوصية النهائية
6.2 خطة التنفيذ المقترحة
6.3 عوامل النجاح الحرجة

---

## 📊 تنسيق البيانات:

- استخدم الجداول لعرض البيانات المالية:
| البند | القيمة | الملاحظات |
|-------|--------|----------|

- استخدم القوائم المرقمة للخطوات
- استخدم النقاط للتفاصيل
- اذكر المصدر لكل معلومة رئيسية: [المصدر: اسم الملف]

---

# المصادر المتاحة (استخرج منها كل التفاصيل):
{context}

# الموضوع المطلوب:
{topic}

---

⚠️ تذكير: اكتب تقريراً مفصلاً وشاملاً. لا تختصر. كل رقم مهم. كل تفصيل مطلوب.
"""

    def suggest_files(self, topic: str, k: int = 5) -> List[str]:
        """
        Smart file suggestion based on topic.
        Returns list of unique filenames most relevant to the topic.
        
        Args:
            topic: The report topic to search for
            k: Number of top documents to retrieve
            
        Returns:
            List of unique filenames (not full paths)
        """
        try:
            print(f"\n🔍 Finding relevant files for topic: {topic[:50]}...")
            
            # Quick similarity search (no reranking for speed)
            docs = self.vector_store.similarity_search(topic, k=k * 2)  # Get more for variety
            
            # Extract unique filenames
            unique_files = []
            seen = set()
            
            for doc in docs:
                source = doc.metadata.get("source", "")
                if source and source != "🌐 Web Search":
                    filename = os.path.basename(source)
                    if filename not in seen:
                        seen.add(filename)
                        unique_files.append(filename)
                        
                        if len(unique_files) >= k:
                            break
            
            print(f"📄 Found {len(unique_files)} relevant files: {unique_files}")
            return unique_files
            
        except Exception as e:
            print(f"❌ Error suggesting files: {e}")
            return []

    def get_all_indexed_files(self) -> List[str]:
        """
        Get all unique filenames indexed in the vector store.
        Useful for populating the multiselect options.
        """
        try:
            # Scroll through Qdrant to get all unique sources
            unique_files = set()
            offset = None
            
            while True:
                results = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, offset = results
                if not points:
                    break
                
                for point in points:
                    if point.payload:
                        source = point.payload.get("metadata", {}).get("source", "")
                        if source and source != "🌐 Web Search":
                            unique_files.add(os.path.basename(source))
                
                if offset is None:
                    break
            
            return sorted(list(unique_files))
            
        except Exception as e:
            print(f"❌ Error getting indexed files: {e}")
            return []

    def _search_with_file_filter(self, query: str, filenames: List[str], k: int = 15) -> List[Any]:
        """
        Search VectorStore with metadata filtering to only search within specific files.
        
        Args:
            query: The search query
            filenames: List of filenames to filter by
            k: Number of documents to retrieve
            
        Returns:
            List of Document objects
        """
        try:
            if not filenames:
                print("⚠️ No files specified for filtering, searching all documents")
                return self.vector_store.similarity_search(query, k=k)
            
            print(f"🔍 Searching within files: {filenames}")
            
            # Search with much higher k and filter manually (need enough to filter by filename)
            # Multiplier of 4 ensures we get enough matches even with file filtering
            all_docs = self.vector_store.similarity_search(query, k=min(k * 4, 200))
            
            # Filter by filename
            filtered_docs = []
            for doc in all_docs:
                source = doc.metadata.get("source", "")
                filename = os.path.basename(source) if source else ""
                
                if filename in filenames:
                    filtered_docs.append(doc)
                    
                    if len(filtered_docs) >= k:
                        break
            
            print(f"📚 Found {len(filtered_docs)} chunks from selected files")
            return filtered_docs
            
        except Exception as e:
            print(f"❌ File-filtered search error: {e}")
            return []

    def generate_report_streaming(self, topic: str, selected_files: List[str]):
        """
        Generate a DETAILED, DATA-HEAVY report with streaming output.
        Uses high-context retrieval (50 chunks) for maximum detail.
        
        Args:
            topic: The report topic
            selected_files: List of filenames to use as sources
            
        Yields:
            String chunks of the report
        """
        try:
            print(f"\n{'='*60}")
            print(f"📝 Generating DETAILED Report: {topic[:50]}...")
            print(f"📄 Using sources: {selected_files}")
            print(f"{'='*60}")
            
            # Step 1: Retrieve HIGH-CONTEXT chunks (50 chunks for maximum detail)
            docs = self._search_with_file_filter(topic, selected_files, k=50)
            
            if not docs:
                yield "⚠️ لم أجد معلومات كافية في الملفات المحددة. يرجى اختيار ملفات أخرى."
                return
            
            # Step 2: Format context with source attribution
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = os.path.basename(doc.metadata.get("source", "Unknown"))
                context_parts.append(f"[المصدر {i}: {source}]\n{doc.page_content}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            print(f"📊 Context prepared: {len(docs)} chunks, {len(context)} chars")
            
            # Step 3: Build the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.WRITER_SYSTEM_PROMPT),
                ("human", "اكتب تقريراً شاملاً عن الموضوع التالي: {topic}")
            ])
            
            # Step 4: Stream the response
            # PRIORITIZE Llama 3.3 70B (Groq) for reports if available (Superior Reasoning + Speed)
            llm_to_use = self.llm_groq_rag if self.llm_groq_rag else self.llm_writer
            model_name = "Llama 3.3 70B (Groq)" if self.llm_groq_rag else "GPT-4o-Mini"
            
            print(f"🧠 Generating report using: {model_name}")
            
            chain = prompt | llm_to_use
            
            # Stream the response
            for chunk in chain.stream({
                "context": context,
                "topic": topic
            }):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            print(f"❌ Report generation error: {e}")
            yield f"❌ حدث خطأ أثناء إنشاء التقرير: {str(e)}"

    def generate_report(self, topic: str, selected_files: List[str]) -> str:
        """
        Non-streaming version of report generation.
        Returns the complete report text.
        """
        chunks = []
        for chunk in self.generate_report_streaming(topic, selected_files):
            chunks.append(chunk)
        return "".join(chunks)

    # ========================================
    # GENERAL HELPER (No RAG - Direct LLM)
    # ========================================
    
    GENERAL_HELPER_PROMPT = """أنت مساعد ذكي متعدد المهام. يمكنك المساعدة في:
- كتابة الإيميلات والرسائل الرسمية
- الترجمة بين العربية والإنجليزية
- كتابة المحتوى والنصوص
- الإجابة على الأسئلة العامة
- التلخيص والتحرير
- العصف الذهني والأفكار

قواعد:
1. كن مهنياً ومساعداً
2. اكتب بنفس لغة المستخدم (إذا كتب بالعربية، رد بالعربية)
3. إذا لم تكن متأكداً من شيء، اسأل للتوضيح
4. قدم إجابات واضحة ومنظمة
"""

    def general_chat_streaming(self, user_input: str, chat_history: list):
        """
        General purpose chat - NO RAG, direct LLM response.
        Uses gpt-4o-mini for cost optimization.
        
        Args:
            user_input: User's message
            chat_history: Previous conversation messages
            
        Yields:
            String chunks of the response
        """
        try:
            # Build the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.GENERAL_HELPER_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}")
            ])
            
            # Convert chat history
            formatted_history = self._convert_chat_history(chat_history)
            
            # Use Groq LLM for lightning-fast responses (falls back to OpenAI if unavailable)
            llm = self.llm_groq if self.llm_groq else self.llm_streaming
            chain = prompt | llm
            
            for chunk in chain.stream({
                "chat_history": formatted_history,
                "input": user_input
            }):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            print(f"❌ General chat error: {e}")
            yield f"عذراً، حدث خطأ: {str(e)}"

    def general_chat(self, user_input: str, chat_history: list) -> str:
        """
        Non-streaming version of general chat.
        """
        chunks = []
        for chunk in self.general_chat_streaming(user_input, chat_history):
            chunks.append(chunk)
        return "".join(chunks)

    # Legacy method for backwards compatibility
    def get_chain(self):
        """
        Legacy method - Returns self for compatibility.
        The new process_query method handles everything.
        """
        return self
    
    def invoke(self, inputs: dict) -> dict:
        """
        Legacy invoke method for compatibility with old chain-style calls.
        """
        user_input = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])
        result = self.process_query(user_input, chat_history)
        return {
            "answer": result["answer"],
            "context": result["context"]
        }
