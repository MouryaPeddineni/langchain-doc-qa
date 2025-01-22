import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document  # Importing Document to wrap content
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

def generate_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def main():
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Load documents and create Chroma database
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks, model, tokenizer)

def load_documents():
    from langchain_community.document_loaders import DirectoryLoader
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True
    )
    return text_splitter.split_documents(documents)

def save_to_chroma(chunks, model, tokenizer):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Generate embeddings for each chunk
    embeddings = [generate_embedding(model, tokenizer, doc.page_content) for doc in chunks]

    # Create a list of Document objects with their corresponding embeddings
    documents_with_embeddings = [
        Document(page_content=doc.page_content) for doc in chunks
    ]

    # Create a Chroma vector store from the documents and embeddings
    embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        documents=documents_with_embeddings,
        embedding=embedding,  # Corrected to embedding
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
