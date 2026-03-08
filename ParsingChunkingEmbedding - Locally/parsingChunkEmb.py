import time
from slack_sdk import WebClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from datetime import datetime

channel_id = "XXXX"

client = WebClient(token="XXXXXXXXXXX")

def fetch_slack_messages(channel):
    messages = []
    # Calculate timestamp for 60 days ago (approx 2 months)
    two_months_ago = time.time() - (60 * 24 * 60 * 60)

    has_more = True
    cursor = None

    while has_more:
        response = client.conversations_history(
            channel=channel,
            oldest=two_months_ago,
            limit=200,
            cursor=cursor
        )

        for msg in response['messages']:
            if 'text' in msg:
                ts = float(msg['ts'])  # Slack timestamp is a string float
                readable_date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                messages.append(f"[{readable_date}] {msg['text']}")

        has_more = response['has_more']
        cursor = response.get('response_metadata', {}).get('next_cursor')

    return messages


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

messages = fetch_slack_messages(channel_id)
docs = splitter.create_documents(messages)

#embedding = OpenAIEmbeddings()
embedding = OpenAIEmbeddings(openai_api_key="XXXXXXXXXXXXX")
vectorstore = FAISS.from_documents(docs, embedding)
vectorstore.save_local("slack_faiss_index")
