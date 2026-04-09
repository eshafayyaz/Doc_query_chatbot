# 📄 PDF Query Chatbot

A smart chatbot that answers questions from your PDF documents using RAG (Retrieval Augmented Generation) technology.

## Features

- Upload and process PDF documents
- Ask questions in natural language
- Get accurate answers based on PDF content
- Chat history maintained during session
- Streaming responses for better UX
- Powered by Groq's LLaMA 3.3 70B model

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (LLaMA 3.3 70B)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **PDF Processing**: pypdf
- **Text Splitting**: LangChain

## Installation

### Prerequisites

- Python 3.13+
- Groq API Key (get it from [console.groq.com](https://console.groq.com))

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Doc_query_chatbot
```

2. Install dependencies using uv:
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

4. Place your PDF file in the `data` folder:
```bash
data/file.pdf
```

## Usage

Run the application:

```bash
uv run streamlit run app.py
```

Or without uv:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How It Works

1. The PDF is loaded and text is extracted
2. Text is split into chunks for better processing
3. Chunks are converted to embeddings using HuggingFace model
4. Embeddings are stored in FAISS vector database
5. When you ask a question:
   - Similar chunks are retrieved from vector store
   - Context is sent to Groq LLM
   - LLM generates answer based on context
   - Response is streamed back to you

## Project Structure

```
Doc_query_chatbot/
├── app.py              # Main application
├── data/
│   └── file.pdf        # Your PDF document
├── .env                # Environment variables
├── .gitignore          # Git ignore file
├── pyproject.toml      # Project dependencies (uv)
├── requirements.txt    # Project dependencies (pip)
└── README.md           # This file
```

## Configuration

You can modify these parameters in `app.py`:

- `chunk_size`: Size of text chunks (default: 500)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `k`: Number of similar chunks to retrieve (default: 3)
- `temperature`: LLM creativity (default: 0.3)
- `max_tokens`: Maximum response length (default: 1024)

## Notes

- The app caches the vector store for faster subsequent runs
- Warnings about torchvision can be safely ignored
- First run will download the embedding model (~90MB)

## License

MIT

## Author

Created with ❤️ using Claude Code
