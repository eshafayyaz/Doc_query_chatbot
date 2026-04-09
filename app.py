import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@st.cache_resource
def setup_rag():
    pdf_path = "data/file.pdf"

    if not os.path.exists(pdf_path):
        st.error(f"PDF not found at {pdf_path}")
        return None

    # Read PDF
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store
    vectorstore = FAISS.from_texts(chunks, embeddings)

    return vectorstore

# Page config
st.set_page_config(page_title="PDF Query Chatbot", page_icon="📄")
st.title("📄 PDF Query Chatbot")

# Initialize vectorstore
with st.spinner("Loading PDF and creating embeddings..."):
    vectorstore = setup_rag()

if vectorstore is None:
    st.stop()

st.success("Ready! Ask me anything from the PDF!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the PDF"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Search similar chunks from PDF
    docs = vectorstore.similarity_search(prompt, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Create prompt for LLM
    llm_prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context from the PDF.
If the answer is not in the context, say "I don't have that information in the document."

Context:
{context}

Question: {prompt}

Answer:"""

    # Get response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": llm_prompt}],
            stream=True,
            temperature=0.3,
            max_tokens=1024
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
