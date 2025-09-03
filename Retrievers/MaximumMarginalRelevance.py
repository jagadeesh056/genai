from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k": 3, "lambda_mult": 0.5}
)

query = "What is Langchain?"

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Content number: {i+1}")
    print(f"Content: {doc.page_content}")