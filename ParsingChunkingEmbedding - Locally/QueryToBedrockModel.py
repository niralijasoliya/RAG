import boto3
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.embeddings import BedrockEmbeddings
import json

# CONFIG
OPENAI_API_KEY = "XXX" # optional if only OpenAI used for embedding
QUERY = "How to secure S3 bucket with public access enabled?"
K = 5  # Top chunks to fetch
MODEL_ID = "anthropic.claude-v2"  # or v2.1 if supported

# 1. Load vectorstore
embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local("text_emb_3_small_model", embedding, allow_dangerous_deserialization=True)

# 2. Embed and Search
results = vectorstore.similarity_search_with_score(QUERY, k=K)
context_chunks = [doc.page_content for doc, _ in results]

# 3. Assemble context
context_text = "\n\n---\n\n".join(context_chunks)

# 4. Prepare prompt for Claude
final_prompt = f"""You are a DevOps expert AI assistant.

Here are some past Slack or Confluence discussions and notes:

{context_text}

Now answer the user's question in a clear and helpful way.

User question:
"{QUERY}"
"""

# 5. Call Claude via Bedrock
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock_client.invoke_model(
    modelId=MODEL_ID,
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "prompt": f"\n\nHuman: {final_prompt}\n\nAssistant:",
        "max_tokens_to_sample": 800,
        "temperature": 0.5,
        "stop_sequences": ["\n\nHuman:"]
    })
)

# 6. Decode response

result_body = json.loads(response['body'].read())
answer = result_body['completion']

# 7. Display
print(f"\n🔎 Query: {QUERY}\n")
for i, (doc, score) in enumerate(results, 1):
    print(f"📌 Result {i} [Score: {score:.2f}]")
    print(doc.page_content.strip())
    print("-" * 80)

print("\n🤖 Claude's Answer:\n")
print(answer)
