from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 1. Load the vectorstore with the newer embedding model
OPENAI_API_KEY = "XXX"
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",  # ✅ Newer, faster, cost-effective model
    openai_api_key=OPENAI_API_KEY
)

vectorstore = FAISS.load_local("text_emb_3_small_model", embedding, allow_dangerous_deserialization=True)

# 2. Define your query
#query = "Could you please share a summary of the discussions or key points covered by the team on 2024-12-24?"
#query = "Could you please share a summary of the discussions or key points covered by the team last week?"
query = "Give me details about EC2 Instance Protection."
#query = "What did we discuss about migration?"
#query = "How to secure S3 bucket with public access enabled?"

# 3. Run similarity search with scores
results = vectorstore.similarity_search_with_score(query, k=5)

# 4. Print formatted results with metadata
print(f"\n🔎 Query: {query}\n")
for i, (doc, score) in enumerate(results, 1):
    start_time = doc.metadata.get("start_time", "⏳ Unknown start")
    end_time = doc.metadata.get("end_time", "⏳ Unknown end")
    users = doc.metadata.get("users", "N/A")
    content = doc.page_content.strip()

    print(f"📌 Result {i} [Score: {score:.2f}]")
    print(f"🕒 {start_time} → {end_time} | 👥 Participants: {users}")
    print(f"💬 {content}")
    print("-" * 80)
