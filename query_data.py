import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader

def process_document(file):
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.name)

    with open(temp_path, "wb") as temp_file:
        temp_file.write(file.read())

    # Check the file type and use the appropriate loader
    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)  # Use PyPDFLoader for PDFs
    elif file.name.endswith(".txt"):
        loader = TextLoader(temp_path)  # Use TextLoader for plain text files
    else:
        raise ValueError("Unsupported file format. Only .txt and .pdf files are supported.")

    # Load the document
    documents = loader.load()

    # Split the document into smaller chunks for better embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Clean up the temporary file
    os.remove(temp_path)
    return texts


# Function to create embeddings and a vector store
def create_vector_store(texts):
    # Use HuggingFaceEmbeddings to generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a Chroma vector store from the text chunks
    db = Chroma.from_documents(texts, embeddings)
    return db

# Function to perform a similarity search
def perform_query(vector_store, query_text):
    results = vector_store.similarity_search(query_text, k=3)
    return results

# Streamlit interface
st.title("📄 Document Query Interface")

# Upload file section
uploaded_file = st.file_uploader("Upload a document (text file only)", type=["txt", "pdf"])
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    st.write("Processing the document...")
    
    # Process the uploaded document
    with st.spinner("Processing..."):
        texts = process_document(uploaded_file)
        vector_store = create_vector_store(texts)
    st.success("Document processed and ready for queries!")

    # Query input box
    query_text = st.text_input("Enter your query:")
    if query_text:
        # Perform the query on the document
        st.write("Searching for relevant content...")
        results = perform_query(vector_store, query_text)

        # Display the results
        st.markdown("### Top Results:")
        for i, result in enumerate(results):
            st.markdown(f"**Result {i + 1}:**")
            st.write(result.page_content[:500])  # Show the first 500 characters of the result