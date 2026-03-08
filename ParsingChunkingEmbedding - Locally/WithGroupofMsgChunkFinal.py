import time, json
from slack_sdk import WebClient
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from datetime import datetime,timedelta
from openai import OpenAI

# Constants
SLACK_TOKEN = "XXX"
CHANNEL_ID = "XXX"
VECTORDB_DIR = "slack_faiss_index"
OPENAI_API_KEY = "XXX"
TIME_WINDOW_MINUTES = 20  # grouping window

client = WebClient(token=SLACK_TOKEN)
openai_embed = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Step 1: Map Slack user IDs to names
def fetch_user_map():
    user_map = {}
    result = client.users_list()
    for user in result['members']:
        user_map[user['id']] = user.get('real_name', user['name'])
    return user_map

# Step 2: Fetch all messages (with thread context)
def fetch_all_messages(channel_id, user_map):
    all_msgs = []
    has_more, cursor = True, None

    while has_more:
        res = client.conversations_history(channel=channel_id, limit=200, cursor=cursor)
        for msg in res["messages"]:
            ts = float(msg["ts"])
            text = msg.get("text", "").strip()
            if not text:
                continue

            thread_ts = msg.get("thread_ts")
            if thread_ts and thread_ts != msg["ts"]:
                try:
                    parent_res = client.conversations_replies(channel=channel_id, ts=thread_ts, limit=1)
                    if parent_res["messages"]:
                        text = f"(Thread context): {parent_res['messages'][0].get('text', '')}\n{text}"
                except Exception:
                    pass

            all_msgs.append({
                "ts": ts,
                "dt": datetime.fromtimestamp(ts),
                "user": user_map.get(msg.get("user", ""), "Unknown"),
                "text": text
            })

        has_more = res.get("has_more", False)
        cursor = res.get("response_metadata", {}).get("next_cursor")

    return sorted(all_msgs, key=lambda x: x["ts"])  # Sort by time

# Step 3: Group messages into time windows (15–30 minutes)
def group_by_time_window(messages, window_minutes=20):
    grouped = []
    current_group = []
    start_time = None

    for msg in messages:
        if not start_time:
            start_time = msg["dt"]
            current_group.append(msg)
        elif (msg["dt"] - start_time) <= timedelta(minutes=window_minutes):
            current_group.append(msg)
        else:
            grouped.append(current_group)
            current_group = [msg]
            start_time = msg["dt"]

    if current_group:
        grouped.append(current_group)

    return grouped

# Step 4: Convert to LangChain documents with metadata
def build_documents(groups):
    docs = []
    for group in groups:
        combined_text = "\n".join([f"{msg['user']}: {msg['text']}" for msg in group])
        
        unique_users = sorted(set(m["user"] for m in group))  # get unique names alphabetically
        
        metadata = {
            "start_time": group[0]["dt"].strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": group[-1]["dt"].strftime("%Y-%m-%d %H:%M:%S"),
            "users": unique_users  # List of user names instead of count
        }
        
        docs.append(Document(page_content=combined_text, metadata=metadata))
    return docs


# Step 5: Vectorization
def create_vectorstore(docs):
    vectorstore = FAISS.from_documents(docs, openai_embed)
    vectorstore.save_local(VECTORDB_DIR)
    print(f"✅ Vectorstore saved to '{VECTORDB_DIR}'")

# MAIN pipeline
def main():
    print("📥 Fetching users and messages...")
    user_map = fetch_user_map()
    all_msgs = fetch_all_messages(CHANNEL_ID, user_map)
    if not all_msgs:
        raise ValueError("❌ No messages found.")

    print(f"📦 {len(all_msgs)} messages fetched. Grouping into time windows...")
    groups = group_by_time_window(all_msgs, window_minutes=TIME_WINDOW_MINUTES)
    docs = build_documents(groups)

    print(f"🧠 {len(docs)} message groups created for embedding.")
    create_vectorstore(docs)

# Optional querying
def query_vectorstore(query):
    store = FAISS.load_local(VECTORDB_DIR, openai_embed)
    results = store.similarity_search_with_score(query, k=5)

    print(f"\n🔎 Query: {query}\n")
    for doc, score in results:
        meta = doc.metadata
        print(f"🕒 {meta['start_time']} → {meta['end_time']} ({meta['user_count']} participants) — Score: {score:.2f}")
        print(doc.page_content)
        print("─" * 80)

if __name__ == "__main__":
    main()