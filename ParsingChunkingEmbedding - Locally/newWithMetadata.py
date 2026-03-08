import time,json
from slack_sdk import WebClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from datetime import datetime
from openai import OpenAI

# Constants
SLACK_TOKEN = "XXXXXX"
CHANNEL_ID = "XXX"
VECTORDB_DIR = "slack_faiss_index"
LOOKBACK_DAYS = 2
OPENAI_API_KEY = "XXX"

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

def extract_tags_from_openai(text):
    prompt = f"Extract the intent and keywords from the following Slack message:\n\n'{text}'\n\nFormat the response as JSON with fields: intent (string), keywords (list of strings)."
    try:
        res = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return json.loads(res.choices[0].message.content.strip())
    except Exception as e:
        print("Tag extraction failed:", e)
        return {"intent": "general", "keywords": []}
    
# === Step 2: Fetch Slack messages with metadata ===
def fetch_slack_messages(channel_id, user_map):
    messages = []
    #since_ts = time.time() - LOOKBACK_DAYS * 24 * 60 * 60
    #print("⏱️ Fetching messages since:", datetime.fromtimestamp(since_ts))
    has_more, cursor = True, None

    while has_more:
        response = client.conversations_history(
            channel=channel_id,
            limit=200,
            cursor=cursor
        )
        for msg in response['messages']:
            text = msg.get("text", "").strip()
            if not text:
                continue
            ts = float(msg["ts"])
            date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            user_id = msg.get("user", "UnknownUser")
            user_name = user_map.get(user_id, "Unknown")
            
            #tags = extract_tags_from_openai(text)
            
            metadata = {
                "timestamp": date,
                "user": user_name,
                #"intent": tags.get("intent", "general"),
                #"keywords": tags.get("keywords", []),
                "slack_user_id": user_id
            }
            messages.append(Document(page_content=text, metadata=metadata))

        has_more = response.get("has_more", False)
        cursor = response.get("response_metadata", {}).get("next_cursor")

    return messages

# === Step 3: Chunk, embed and save to FAISS ===
def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, openai_embed)
    vectorstore.save_local(VECTORDB_DIR)
    return vectorstore

# === Main build flow ===
def main():
    user_map = fetch_user_map()
    raw_docs = fetch_slack_messages(CHANNEL_ID, user_map)
    print(raw_docs)

    if not raw_docs:
        raise ValueError("No messages found in the given time range.")

    print(f"Fetched {len(raw_docs)} messages")
    create_vectorstore(raw_docs)
    print(f"Vectorstore saved to '{VECTORDB_DIR}'")

# === Optional: Query Example ===
def query_vectorstore(query):
    store = FAISS.load_local(VECTORDB_DIR, openai_embed)
    results = store.similarity_search_with_score(query, k=5)

    print(f"\n🔍 Query: {query}\n")
    for doc, score in results:
        ts = doc.metadata.get("timestamp", "Unknown time")
        user = doc.metadata.get("user", "Unknown user")
        print(f"🔹 [{ts}] {user}: {doc.page_content.strip()} (score: {score:.2f})")

# === Run ===
if __name__ == "__main__":
    main()