# Saudi Market Knowledge Bank - Walkthrough

I have built a **Retrieval-Augmented Generation (RAG)** system tailored for Saudi market data, replacing the movie script logic from your reference project with a professional document analysis pipeline.

## 🎯 What We Built
A private "Google" for your company's internal reports. You feed it PDFs/text files, and it answers questions based *only* on those documents, citing its sources.

## 🏗 System Architecture

### 1. The "Brain" (Logic)
- **LangChain**: The framework connecting everything.
- **OpenAI GPT-4o**: The intelligence engine that formulates answers.
- **Qdrant**: The "Long-Term Memory" (Vector Database). It stores your documents in a mathematical format that allows for instant "semantic search" (finding meaning, not just keywords).

### 2. The Files
| File | Purpose |
| :--- | :--- |
| **`ingest.py`** | **The Librarian**. It reads every PDF in `data/`, splits them into small pages, and files them into the Qdrant database. |
| **`rag.py`** | **The Analyst**. It contains the `MarketExpert` class. It receives a question, finds the top 5 relevant pages from the database, and sends them to GPT-4o to write an answer. |
| **`app.py`** | **The Interface**. A web-based chat window (built with Streamlit) for employees to interact with the system. |
| **`data/`** | **The Vault**. Your folder for storing PDF reports and studies. |

## 🔍 Deep Dive: How "Vectoring" Works (The Backend Magic)
You asked how we save the data. We don't just save words; we save **concepts**. Here is the 4-step process happening inside `ingest.py`:

### 1. Chunking (Breaking it Down)
A 100-page PDF is too big for an AI to read all at once.
- **What we do**: We use a `RecursiveCharacterTextSplitter`.
- **The Result**: We slice the document into small "chunks" (paragraphs) of 1,000 characters each.

### 2. Vectorization (Text to Numbers)
Computers represent meaning as numbers.
- **The Model**: We use OpenAI's `text-embedding-3-small` model.
- **The Process**: It reads a chunk like *"Saudi GDP grew by 4%"* and converts it into a list of 1,536 numbers (coordinates).
- **Why?**: This places similar concepts close together in space. "Profit" and "Revenue" will have similar numbers, even if the words look different.

### 3. Indexing & Storage (Qdrant)
- **The Database**: We use **Qdrant** (stored in the `qdrant_db/` folder).
- **The Save**: We save the **Vector** (the numbers) + The **Payload** (the original text and filename).
- **Efficiency**: Qdrant builds a map (HNSW index) so it can search millions of pages in milliseconds.

### 4. Semantic Retrieval
When you ask *"How are banks performing?"*:
1.  We convert your question into numbers (Vector B).
2.  We look for stored vectors (Vector A) that are mathematically closest to Vector B.
3.  We find the "Al Rajhi Bank" report, even if you didn't use the word "Rajhi".

## 🚀 "Low-Code" Features
To make this easy to use without programming knowledge, I added:
- **`1_Update_Brain.command`**: A double-clickable script that runs `ingest.py` automatically.
- **`2_Open_App.command`**: A double-clickable script that launches the web interface.
- **`USER_GUIDE.md`**: A simple instruction manual.

## 🌍 How to Publish on a Server
To share this with your team, you can host it on a cloud server (like AWS, Azure, or DigitalOcean).

### 1. Server Requirements
- **OS**: Ubuntu Linux (Recommended).
- **Python**: Version 3.10 or higher.
- **Security**: Open port **8501** (Streamlit's default port).

### 2. Deployment Steps
1.  **Upload Files**: Copy all project files to the server (you can use Git or SCP).
2.  **Install Python**: `sudo apt install python3 python3-pip`
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set Environment Variables**:
    - Create the `.env` file on the server.
    - Add your `OPENAI_API_KEY`.
5.  **Run the App**:
    - To keep it running even when you close the terminal, use `nohup`:
    ```bash
    nohup streamlit run app.py &
    ```
    - Or use a professional process manager like **Docker** or **Systemd**.

### 3. Accessing the App
- Your employees can now visit: `http://YOUR_SERVER_IP:8501`


## 🚀 Going Big: Using Qdrant Server (For 10GB+ Data)
If you have massive datasets, you should switch from "Local Mode" to "Server Mode".

### 1. Install Docker
- Download and install **Docker Desktop** for your Mac.

### 2. Start the Server
- Open your terminal in the project folder and run:
  ```bash
  docker-compose up -d
  ```
- This starts a powerful Qdrant server in the background.

### 3. Update Configuration
- Open your `.env` file.
- Change `QDRANT_MODE=local` to:
  ```properties
  QDRANT_MODE=server
  ```

### 4. Re-Index
- Double-click **`1_Update_Brain.command`**.
- The system will now read all your files and store them in the robust Server database instead of the local folder.
