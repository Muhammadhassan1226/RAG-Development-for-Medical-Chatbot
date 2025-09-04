from dotenv import load_dotenv
import os
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from helper import filter_to_minimal_docs,download_embeddings,extract_data,text_split
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE



extracted_data = extract_data(data = "data/")


filtered_data = filter_to_minimal_docs(extracted_data)



text_chunks = text_split(filtered_data)
print("Length of Text Chunks", len(text_chunks))

embeddings = download_embeddings()


pinecone_key  = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_key)
print("pc", pc)

index_name = "medical-chatbot"



if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )


Index = pc.Index(index_name)


docsearch  = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)