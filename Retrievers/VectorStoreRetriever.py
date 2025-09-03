from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document 

# Sample documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Updated embedding model usage
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# Query
query = "Tell me about OpenAI"
results = retriever.invoke(query)

# Display results
for i, doc in enumerate(results):
    print(f"\n-- Result: {i+1}")
    print(f"Page Content: {doc.page_content}")

result = vector_store.similarity_search(query, k=2)
for i, doc in enumerate(result):
    print(f"\n-- Result: {i+1}")
    print(f"Content: {doc.page_content}")