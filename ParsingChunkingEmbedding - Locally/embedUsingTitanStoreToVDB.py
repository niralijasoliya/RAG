import hashlib
import json
from datetime import datetime, timedelta
import boto3
import psycopg2
from langchain_aws import BedrockEmbeddings
from slack_sdk import WebClient
from langchain_aws import ChatBedrock
import time
from slack_sdk.errors import SlackApiError

SLACK_TOKEN = "XXXXXXXX"
CHANNEL_ID = "XXXXXX"
REGION = "eu-west-1"
TIME_WINDOW_MINUTES = 60

PG_CONN = psycopg2.connect(
    host="XXXXX",
    user="XX",
    password="XXXX",
    dbname="XXX",
    port=5432
)
CURSOR = PG_CONN.cursor()

# AWS & Slack Clients
client = WebClient(token=SLACK_TOKEN)
bedrock_embedder = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=boto3.client("bedrock-runtime", region_name=REGION)
)

claude_llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    client=boto3.client("bedrock-runtime", region_name=REGION)
)

# Step 1: Map Slack user IDs to names

def fetch_user_map():
    user_map = {}
    max_retries = 5
    retry_delay = 60  # 60 seconds

    for attempt in range(max_retries):
        try:
            result = client.users_list()
            for user in result['members']:
                user_map[user['id']] = user.get('real_name', user['name'])
            return user_map

        except SlackApiError as e:
            if e.response["error"] == "ratelimited":
                print(f"⚠️ Rate limited. Retrying in {retry_delay} seconds... (attempt {attempt+1})")
                time.sleep(retry_delay)
            else:
                raise e

    raise Exception("❌ Failed to fetch users after multiple retries due to rate limits.")


def get_channel_name(channel_id):
    try:
        response = client.conversations_info(channel=channel_id)
        return response["channel"]["name"]
    except Exception as e:
        print(f"⚠️ Could not fetch channel name: {e}")
        return "Unknown"

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
def group_by_time_window(messages, window_minutes=60):
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

def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_intent_and_keywords(chunk: str) -> dict:
    prompt = f"""
You are a semantic understanding agent. Read the following Slack message group and extract:

1. A short **intent** (1-3 words)
2. A list of **keywords** (3-7 important terms)

Respond ONLY in JSON like:
{{
  "intent": "infra-deployment",
  "keywords": ["ECS", "rollback", "cluster", "versioning"]
}}

Message group:
\"\"\"{chunk}\"\"\"
"""

    response = claude_llm.invoke(prompt)
    return json.loads(response.content.strip())

def store_chunk(session_id, chunk_index, text, embedding, metadata):
    content_hash = sha256_text(text)
    try:
        CURSOR.execute("""
            INSERT INTO public.data_embeddings (
                session_id, chunk_index, chunk_text, embedding_vector,
                title, start_time, end_time, participants,
                intent, keywords, source, channel_name,
                report_url, content_hash
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s
            ) ON CONFLICT (session_id, content_hash)
            DO UPDATE SET
                chunk_text       = EXCLUDED.chunk_text,
                embedding_vector = EXCLUDED.embedding_vector,
                title            = EXCLUDED.title,
                start_time       = EXCLUDED.start_time,
                end_time         = EXCLUDED.end_time,
                participants     = EXCLUDED.participants,
                intent           = EXCLUDED.intent,
                keywords         = EXCLUDED.keywords,
                source           = EXCLUDED.source,
                channel_name     = EXCLUDED.channel_name,
                report_url       = EXCLUDED.report_url,
                content_hash     = EXCLUDED.content_hash,
                updated_at       = CURRENT_TIMESTAMP
            WHERE public.data_embeddings.content_hash IS DISTINCT FROM EXCLUDED.content_hash
        """, (
            session_id, chunk_index, text, embedding,
            metadata["title"], metadata["start_time"], metadata["end_time"], json.dumps(metadata["users"]),
            metadata["intent"], json.dumps(metadata["keywords"]), metadata["source_name"], metadata["channel_name"],
            metadata.get("report_url", ""), content_hash
        ))
        PG_CONN.commit()
        # If rowcount is 0, it means the conflict occurred but WHERE was false (no change)
        if CURSOR.rowcount == 0:
            print(f"↩️ Chunk {chunk_index} unchanged, skipped.")
        else:
            print(f"✅ Stored/updated chunk {chunk_index} from session {session_id}")
    except Exception as e:
        PG_CONN.rollback()
        print(f"❌ Error inserting/updating chunk {chunk_index}: {e}")

def main():
    print("📥 Fetching Slack data...")
    user_map = fetch_user_map()
    channel_name = get_channel_name(CHANNEL_ID)
    messages = fetch_all_messages(CHANNEL_ID, user_map)
    groups = group_by_time_window(messages, window_minutes=TIME_WINDOW_MINUTES)

    for i, group in enumerate(groups):
        session_id = CHANNEL_ID
        combined_text = "\n".join([f"{msg['user']}: {msg['text']}" for msg in group])
        embedding = bedrock_embedder.embed_query(combined_text)
        tags = get_intent_and_keywords(combined_text)
        metadata = {
            "title": f"Slack Discussion {i+1}",
            "start_time": group[0]["dt"],
            "end_time": group[-1]["dt"],
            "users": sorted(set(msg["user"] for msg in group)),
            "intent": tags["intent"],
            "keywords": tags["keywords"],
            "source_name": "Slack",
            "channel_name": channel_name,
            "report_url": None
        }
        store_chunk(session_id, i, combined_text, embedding, metadata)

if __name__ == "__main__":
    main()