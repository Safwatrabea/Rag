import os
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader, 
    TextLoader, 
    Docx2txtLoader, 
    CSVLoader, 
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATA_DIR = "data"
COLLECTION_NAME = "saudi_market_knowledge"
QDRANT_PATH = "qdrant_db"

def ingest_documents():
    print(f"Loading documents from {DATA_DIR}...")
    
    # Define loaders for different file types
    loaders = {
        "pdf": DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
        "txt": DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader),
        "docx": DirectoryLoader(DATA_DIR, glob="**/*.docx", loader_cls=Docx2txtLoader),
        "csv": DirectoryLoader(DATA_DIR, glob="**/*.csv", loader_cls=CSVLoader),
        "xlsx": DirectoryLoader(DATA_DIR, glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader),
    }

    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files...")
        try:
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"Loaded {len(loaded_docs)} {file_type} documents.")
        except Exception as e:
            print(f"Error loading {file_type} files: {e}")
    
    if not docs:
        print("No documents found in data/ directory.")
        return

    print(f"Loaded {len(docs)} documents.")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    # Initialize Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize Qdrant
    qdrant_mode = os.getenv("QDRANT_MODE", "local").lower()
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    if qdrant_mode == "server":
        print(f"Connecting to Qdrant Server at {qdrant_url}...")
        client = QdrantClient(url=qdrant_url)
    else:
        print(f"Using Local Qdrant at {QDRANT_PATH}...")
        client = QdrantClient(path=QDRANT_PATH)
    
    # Check if collection exists, if not create it
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")

    # Index chunks using the existing client
    print("Indexing chunks into Qdrant...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(splits)
    print("Ingestion complete!")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key.")
    else:
        ingest_documents()
