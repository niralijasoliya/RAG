import boto3
from langchain_aws import BedrockEmbeddings

embedder = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=boto3.client("bedrock-runtime", region_name="eu-west-1")
)

vector = embedder.embed_query("Example text to embed")
print(len(vector))  # Should be 1536
