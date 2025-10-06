# RAG-using-NVIDIA_NIM-and-Langchain

This Streamlit application enables question answering over a collection of PDF documents using NVIDIA's embeddings and a LLaMA-based language model via LangChain. It loads PDFs from a folder, creates vector embeddings for semantic search, and provides accurate answers based on document context.

---

## Features

- **PDF Document Loading:** Automatically loads PDFs from the `./RAG_DOCS` directory.
- **Document Embedding:** Uses NVIDIAEmbeddings to create vector embeddings of document chunks.
- **Semantic Search:** Utilizes FAISS vector store for efficient document retrieval.
- **Question Answering:** Queries documents with a LLaMA 3 8B instruct model (`meta/llama3-8b-instruct`) to provide context-based answers.
- **Interactive UI:** Streamlit frontend with input box for questions, embedding trigger button, and expandable view for document similarity results.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/gauravkumbhar5344/RAG-using-NVIDIA_NIM-and-Langchain.git
   cd RAG-using-NVIDIA_NIM-and-Langchain
   ```
2. **Create and activate a virtual environment (recommended):**
  ```bash
      python -m venv venv
      source venv/bin/activate
  ```
3. **Install dependencies:**
   ```bash
     pip install -r requirements.txt
   ```
4. **Prepare your environment variables:**
   ```bash
    NVIDIA_API_KEY=your_nvidia_api_key_here
    ```

## Usage
1. Add your PDF documents:
  Place your PDFs inside the ./RAG_DOCS directory. The app will load and process these files.

2. Run the Streamlit app:
  ```bash
  streamlit run app.py
  ```
3. Using the app:
     - Click Document embedding to load the PDFs, split them into chunks, and create vector embeddings.
     - Enter a question related to the documents in the text input box.
     - The app will retrieve relevant document chunks and generate an answer based on the content.
     -Expand Document Similarity Search to view the document excerpts used for answering.


