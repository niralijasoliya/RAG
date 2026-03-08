from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load PDF
loader = PyPDFLoader("your_document.pdf")
documents = loader.load()

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)

# Step 3: Create embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 4: Create FAISS index
vector_store = FAISS.from_documents(docs, embeddings)

# Step 5: Save FAISS index
vector_store.save_local("faiss_index")

print("✅ PDF embedded and FAISS index saved.")