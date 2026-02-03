import os
import json
import asyncio
import hashlib
import nest_asyncio
from pathlib import Path
from typing import List, Set, Dict
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    Docx2txtLoader, 
    CSVLoader, 
    UnstructuredExcelLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

# Allow nested asyncio (needed for LlamaParse in Jupyter/scripts)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

DATA_DIR = "data"
COLLECTION_NAME = "saudi_market_knowledge"
QDRANT_PATH = "qdrant_db"
STATE_FILE = "ingestion_state.json"  # Local state tracking

# Concurrency settings
MAX_CONCURRENT_PARSES = 5  # Process 5 files at a time (rate limit protection)

# Check if LlamaParse is available
try:
    from llama_parse import LlamaParse
    LLAMA_PARSE_AVAILABLE = True
except ImportError:
    LLAMA_PARSE_AVAILABLE = False
    print("⚠️ LlamaParse not installed. Using fallback PyPDFLoader for PDFs.")


def load_ingestion_state() -> Dict[str, float]:
    """
    Load the ingestion state from JSON file.
    Returns a dictionary mapping filename -> last_modified_timestamp.
    """
    if not os.path.exists(STATE_FILE):
        return {}
    
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Could not load state file: {e}")
        return {}


def save_ingestion_state(state: Dict[str, float]):
    """
    Save the ingestion state to JSON file.
    """
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save state file: {e}")


def should_process_file(file_path: Path, state: Dict[str, float]) -> bool:
    """
    Check if a file should be processed based on modification time.
    
    Args:
        file_path: Path to the file
        state: Current ingestion state
    
    Returns:
        True if file should be processed, False if it should be skipped
    """
    filename = str(file_path)
    current_mtime = os.path.getmtime(file_path)
    
    # If file is not in state, it's new -> process it
    if filename not in state:
        return True
    
    # If file timestamp is newer than stored timestamp -> process it (updated)
    if current_mtime > state[filename]:
        return True
    
    # File hasn't changed -> skip it
    return False


def get_file_hash(file_path: Path) -> str:
    """Generate a hash for a file based on its path and modification time."""
    stat = file_path.stat()
    hash_input = f"{file_path.name}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def get_indexed_file_hashes(client: QdrantClient, collection_name: str) -> Set[str]:
    """
    Get all unique file hashes already indexed in Qdrant.
    This enables smart skip for incremental ingestion.
    """
    try:
        if not client.collection_exists(collection_name):
            return set()
        
        # Scroll through all points to get unique file hashes
        indexed_hashes = set()
        offset = None
        
        while True:
            results = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, offset = results
            if not points:
                break
                
            for point in points:
                if point.payload and "file_hash" in point.payload:
                    indexed_hashes.add(point.payload["file_hash"])
            
            if offset is None:
                break
        
        return indexed_hashes
        
    except Exception as e:
        print(f"⚠️ Could not retrieve indexed hashes: {e}")
        return set()


async def parse_single_pdf_async(
    parser: "LlamaParse",
    pdf_path: Path,
    semaphore: asyncio.Semaphore,
    file_hash: str
) -> List[Document]:
    """
    Parse a single PDF file asynchronously with semaphore for rate limiting.
    """
    async with semaphore:
        try:
            # Use async version of load_data
            llama_docs = await parser.aload_data(str(pdf_path))
            
            # Convert LlamaIndex Documents -> LangChain Documents
            langchain_docs = []
            for i, llama_doc in enumerate(llama_docs):
                langchain_doc = Document(
                    page_content=llama_doc.text,
                    metadata={
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "page": i + 1,
                        "parser": "LlamaParse",
                        "format": "markdown",
                        "file_hash": file_hash  # For smart skip
                    }
                )
                langchain_docs.append(langchain_doc)
            
            return langchain_docs
            
        except Exception as e:
            print(f"\n   ❌ Error parsing {pdf_path.name}: {e}")
            return []


async def parse_pdfs_async(data_dir: str, ingestion_state: Dict[str, float] = None) -> List[Document]:
    """
    Parse all PDF files using LlamaParse ASYNCHRONOUSLY with concurrency control.
    
    Args:
        data_dir: Directory containing PDF files
        ingestion_state: Dictionary mapping filename -> last_modified_timestamp
    
    Returns:
        List of LangChain Document objects
    """
    if not LLAMA_PARSE_AVAILABLE:
        print("❌ LlamaParse not available. Skipping PDF parsing.")
        return []
    
    llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not llama_api_key:
        print("❌ LLAMA_CLOUD_API_KEY not found. Skipping LlamaParse.")
        print("   Get your free API key at: https://cloud.llamaindex.ai/")
        return []
    
    ingestion_state = ingestion_state or {}
    
    # Initialize LlamaParse
    parser = LlamaParse(
        api_key=llama_api_key,
        result_type="markdown",
        verbose=False,  # Reduce noise in async mode
        language="ar",
    )
    
    # Find all PDF files
    pdf_files = list(Path(data_dir).glob("**/*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        print("⚠️ No PDF files found in data directory")
        return []
    
    # Filter out already indexed files (Smart Skip using JSON state)
    files_to_process = []
    skipped_count = 0
    
    for pdf_path in pdf_files:
        if should_process_file(pdf_path, ingestion_state):
            file_hash = get_file_hash(pdf_path)
            files_to_process.append((pdf_path, file_hash))
        else:
            skipped_count += 1
            print(f"⏭️  Skipping {pdf_path.name} - No changes detected")
    
    if skipped_count > 0:
        print(f"⏭️  Total skipped: {skipped_count} unchanged files")
    
    if not files_to_process:
        print("✅ All files already indexed. Nothing to process.")
        return []
    
    print(f"🚀 Processing {len(files_to_process)} new/modified files (max {MAX_CONCURRENT_PARSES} concurrent)")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PARSES)
    
    # Create async tasks for all files
    tasks = [
        parse_single_pdf_async(parser, pdf_path, semaphore, file_hash)
        for pdf_path, file_hash in files_to_process
    ]
    
    # Process with progress bar
    all_docs = []
    results = await tqdm_asyncio.gather(
        *tasks,
        desc="📖 Parsing PDFs",
        unit="file"
    )
    
    # Flatten results
    for doc_list in results:
        all_docs.extend(doc_list)
    
    # Update ingestion state for processed files
    for pdf_path, _ in files_to_process:
        ingestion_state[str(pdf_path)] = os.path.getmtime(pdf_path)
    
    print(f"\n📚 Parsed {len(all_docs)} document pages from {len(files_to_process)} files")
    return all_docs


def load_other_documents(data_dir: str, ingestion_state: Dict[str, float] = None) -> List[Document]:
    """
    Load non-PDF documents using standard loaders with state tracking.
    """
    ingestion_state = ingestion_state or {}
    
    loaders_config = {
        "txt": ("**/*.txt", TextLoader),
        "docx": ("**/*.docx", Docx2txtLoader),
        "csv": ("**/*.csv", CSVLoader),
        "xlsx": ("**/*.xlsx", UnstructuredExcelLoader),
    }

    docs = []
    
    for file_type, (glob_pattern, loader_cls) in loaders_config.items():
        try:
            # Find all files of this type
            files = list(Path(data_dir).glob(glob_pattern))
            
            if not files:
                continue
            
            # Filter files based on state
            files_to_process = []
            skipped_count = 0
            
            for file_path in files:
                if should_process_file(file_path, ingestion_state):
                    files_to_process.append(file_path)
                else:
                    skipped_count += 1
                    print(f"⏭️  Skipping {file_path.name} - No changes detected")
            
            if skipped_count > 0:
                print(f"⏭️  {file_type.upper()}: Skipped {skipped_count} unchanged files")
            
            # Load only files that need processing
            for file_path in files_to_process:
                try:
                    loader = loader_cls(str(file_path))
                    loaded_docs = loader.load()
                    docs.extend(loaded_docs)
                    
                    # Update state for processed file
                    ingestion_state[str(file_path)] = os.path.getmtime(file_path)
                    
                except Exception as e:
                    print(f"⚠️ Error loading {file_path.name}: {e}")
            
            if files_to_process:
                print(f"📄 Loaded {len(files_to_process)} {file_type.upper()} documents.")
                
        except Exception as e:
            if "No such file" not in str(e):
                print(f"⚠️ Error processing {file_type} files: {e}")
    
    return docs


def fallback_load_pdfs(data_dir: str) -> List[Document]:
    """
    Fallback: Load PDFs using PyPDFLoader if LlamaParse is unavailable.
    """
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    
    print("📄 Using fallback PyPDFLoader for PDFs...")
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True)
    
    try:
        docs = loader.load()
        print(f"📄 Loaded {len(docs)} PDF pages with PyPDFLoader")
        return docs
    except Exception as e:
        print(f"❌ Error loading PDFs with fallback: {e}")
        return []


async def ingest_documents_async():
    """
    Async version of document ingestion with JSON-based incremental sync.
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting Document Ingestion (Incremental Sync Mode)")
    print(f"{'='*60}\n")
    print(f"📁 Source directory: {DATA_DIR}")
    
    # Load ingestion state (JSON-based tracking)
    print("🔍 Loading ingestion state...")
    ingestion_state = load_ingestion_state()
    
    if ingestion_state:
        print(f"📦 Found {len(ingestion_state)} files in state tracking")
    else:
        print("📦 No previous state found - first run or fresh start")
    
    # Step 1: Parse PDFs with LlamaParse (or fallback)
    llama_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    
    if LLAMA_PARSE_AVAILABLE and llama_api_key:
        print("\n🦙 Using LlamaParse for PDF processing (async, parallel)")
        pdf_docs = await parse_pdfs_async(DATA_DIR, ingestion_state=ingestion_state)
    else:
        print("\n⚠️ LlamaParse unavailable. Using standard PDF loader.")
        pdf_docs = fallback_load_pdfs(DATA_DIR)
    
    # Step 2: Load other document types
    print("\n📂 Loading other document types (txt, docx, csv, xlsx)...")
    other_docs = load_other_documents(DATA_DIR, ingestion_state=ingestion_state)
    
    # Combine all documents
    all_docs = pdf_docs + other_docs
    
    if not all_docs:
        print("\n✅ No new documents to ingest.")
        # Save state even if no new docs (in case files were deleted)
        save_ingestion_state(ingestion_state)
        return

    print(f"\n📊 Total new documents to index: {len(all_docs)}")

    # Step 3: Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "|", " ", ""],
    )
    
    print("✂️ Splitting documents into chunks...")
    splits = text_splitter.split_documents(all_docs)
    print(f"✂️ Created {len(splits)} chunks")

    # Step 4: Initialize Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Step 5: Initialize Qdrant client
    qdrant_mode = os.getenv("QDRANT_MODE", "local").lower()
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    if qdrant_mode == "server":
        print(f"\n🔗 Connecting to Qdrant Server at {qdrant_url}...")
        client = QdrantClient(url=qdrant_url)
    else:
        print(f"\n📁 Using Local Qdrant at {QDRANT_PATH}...")
        client = QdrantClient(path=QDRANT_PATH)
    
    # Step 6: Ensure collection exists
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"✨ Created collection: {COLLECTION_NAME}")

    # Step 7: Index chunks with progress bar
    print("\n⬆️ Indexing chunks into Qdrant...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    # Add documents in batches with progress
    batch_size = 50
    for i in tqdm(range(0, len(splits), batch_size), desc="📤 Uploading", unit="batch"):
        batch = splits[i:i + batch_size]
        vector_store.add_documents(batch)
    
    # Step 8: Save updated state
    print("\n💾 Saving ingestion state...")
    save_ingestion_state(ingestion_state)
    
    print(f"\n{'='*60}")
    print(f"✅ Ingestion Complete!")
    print(f"{'='*60}")
    print(f"   📄 Documents processed: {len(all_docs)}")
    print(f"   ✂️ Chunks indexed: {len(splits)}")
    print(f"   📦 Collection: {COLLECTION_NAME}")
    print(f"   💾 State saved to: {STATE_FILE}")
    print(f"{'='*60}\n")


def ingest_documents():
    """
    Synchronous wrapper for async ingestion.
    """
    asyncio.run(ingest_documents_async())


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("   Please create a .env file with your API key.")
    else:
        ingest_documents()
