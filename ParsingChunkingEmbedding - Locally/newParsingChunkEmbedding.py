import time
from slack_sdk import WebClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from datetime import datetime

# Slack config
SLACK_TOKEN = "XXXXXXXXXXXXX"
CHANNEL_ID = "XXXXXXXXXXXX"
client = WebClient(token=SLACK_TOKEN)

# Fetch messages from Slack for the last 60 days
def fetch_slack_messages(channel_id):
    messages = []
    two_months_ago = time.time() - (60 * 24 * 60 * 60)
    has_more = True
    cursor = None

    while has_more:
        response = client.conversations_history(
            channel=channel_id,
            oldest=two_months_ago,
            limit=200,
            cursor=cursor
        )

        for msg in response["messages"]:
            ts = float(msg["ts"])
            date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            user = msg.get("user", "UnknownUser")
            text = msg.get("text", "")
            combined = f"[{date}] <@{user}>: {text}"
            messages.append(Document(page_content=combined))

        has_more = response.get("has_more", False)
        cursor = response.get("response_metadata", {}).get("next_cursor")

    return messages

# Split and embed
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
messages = fetch_slack_messages(CHANNEL_ID)
if not messages:
    raise ValueError("❌ No Slack messages were fetched. Check your channel ID or token.")

split_docs = splitter.split_documents(messages)
if not split_docs:
    raise ValueError("❌ No documents were created after splitting. Check your message formatting.")

# Embed and save
embedding = OpenAIEmbeddings(openai_api_key="XXXXXXXXXXXXX")
vectorstore = FAISS.from_documents(split_docs, embedding)
vectorstore.save_local("slack_faiss_index")

print(f"✅ Fetched {len(messages)} raw messages")
print(f"✅ Created {len(split_docs)} chunks for embedding")
