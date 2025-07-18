from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Delhi is capital of India",
    "What is Artificial Intelligence"
]
result = embedding.embed_documents(documents)
print(result)
