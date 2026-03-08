from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Step 1: Load embedding model (same as indexing)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Load FAISS index
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Step 3: Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Step 4: Load local LLM from Ollama
llm = ChatOllama(
    model="llama3",
    temperature=0
)

# Step 5: Prompt template
template = """
Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Step 6: Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# Step 7: Ask questions in loop
while True:

    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"query": query})

    print("\nAnswer:\n", result["result"])


---------------------------------------------------------------

# Step 7: Interactive query loop
while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # --- Retrieve relevant chunks ---
    docs = vectorstore.similarity_search(query, k=3)

    print("\nRetrieved Context Sources:")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1}")
        print("Page:", doc.metadata.get("page", "unknown"))
        print("Text preview:", doc.page_content[:200], "...")

    # --- Generate answer using RAG ---
    result = qa_chain.invoke({"query": query})

    print("\nAnswer:\n", result["result"])