from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint

# Documents
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store
vector_store = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model
)

# Similarity search retriever
similarity_search = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# LLM for query expansion (free Hugging Face model)
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text2text-generation",
)

# Multi-query retriever
multiquery_search = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

# Query
query = "How to improve energy levels and maintain balance?"

# Run searches
similar_res = similarity_search.invoke(query)
multi_res = multiquery_search.invoke(query)

# Print results
print("\n--- Similarity Search Results ---")
for i, doc in enumerate(similar_res):
    print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")

print("\n--- MultiQuery Search Results ---")
for i, doc in enumerate(multi_res):
    print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")
