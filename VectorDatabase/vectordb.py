from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain.schema import Document

# Embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Documents
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )
]

# Create Chroma vector store
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="my_chroma_db",
    collection_name="sample"
)

vector_store.add_documents(docs)

# Search for captains
results = vector_store.similarity_search(
    query="Who among these are IPL captains?",
    k=5
)

# Filter manually for captain-related docs
captains = [doc for doc in results if "captain" in doc.page_content.lower() or "led" in doc.page_content.lower()]

# Take top 2
for i, doc in enumerate(captains[:2], start=1):
    print(f"\nCaptain {i}:")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
