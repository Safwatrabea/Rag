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
        
        # Streaming LLM for typewriter effect in UI
        self.llm_streaming = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, streaming=True)
        
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
