# Saudi Market Knowledge Bank 🏦

A RAG (Retrieval-Augmented Generation) system for querying Saudi market reports, feasibility studies, and banking statistics.

## Setup

1.  **Install Dependencies** (if not already done):
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API Key**:
    - Rename `.env.example` to `.env`.
    - Add your OpenAI API Key: `OPENAI_API_KEY=sk-...`

3.  **Add Data**:
    - Place PDF or Text files in the `data/` directory.

4.  **Ingest Data**:
    - Run the ingestion script to process documents and store them in the vector database:
    ```bash
    python ingest.py
    ```

5.  **Run the App**:
    - Start the employee interface:
    ```bash
    streamlit run app.py
    ```

## Features
- **Document Loading**: Supports PDFs and Text files.
- **Smart Retrieval**: Uses Vector Search (Qdrant) to find relevant sections.
- **Contextual Answers**: GPT-4o generates answers based *only* on your documents.
- **Sources**: Citations are provided for every answer.
