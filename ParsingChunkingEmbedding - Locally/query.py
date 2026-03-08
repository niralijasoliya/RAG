from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 1. Load the vectorstore
OPENAI_API_KEY = "XXX"
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local("slack_faiss_index", embedding, allow_dangerous_deserialization=True)

# 2. Define your query
query = "What did team discuss from 2025-05-11 to 2025-05-14?"

# 3. Run similarity search
results = vectorstore.similarity_search_with_score(query, k=5)

# 4. Print formatted results
print(f"\n🔎 Query: {query}\n")
for i, (doc, score) in enumerate(results, 1):
    start_time = doc.metadata.get("start_time", "⏳ Unknown start")
    end_time = doc.metadata.get("end_time", "⏳ Unknown end")
    user_count = doc.metadata.get("users", "N/A")
    content = doc.page_content.strip()

    print(f"📌 Result {i} [Score: {score:.2f}]")
    print(f"🕒 {start_time} → {end_time} | 👥 Participants: {user_count}")
    print(f"💬 {content}")
    print("-" * 80)
