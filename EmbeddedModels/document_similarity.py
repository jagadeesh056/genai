from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# List of documents
documents = [
    "Virat is the captain of test team",
    "Dhoni is best finisher in the cricket",
    "Rohit is best opener in ODI team"
]

# Your query
query = "Who is Virat Kohli?"

# Get embeddings
doc_embeddings = embedding_model.embed_documents(documents)
query_embedding = embedding_model.embed_query(query)

# Convert to numpy arrays
doc_embeddings = np.array(doc_embeddings)
query_embedding = np.array(query_embedding).reshape(1, -1)

# Compute cosine similarity
similarity_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

# Zip scores with documents and sort by highest similarity
ranked_results = sorted(zip(documents, similarity_scores), key=lambda x: x[1], reverse=True)

# Print ranked results
print("\nSimilarity Ranking:")
for i, (doc, score) in enumerate(ranked_results, start=1):
    print(f"{i}. Score: {score:.4f} | Text: {doc}")

# Print the most similar one
most_similar_text = ranked_results[0][0]
print(f"\n✅ Most similar text: \"{most_similar_text}\"")


# from langchain_huggingface import HuggingFaceEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Initialize embedding model
# embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Documents and query
# documents = [
#     "Virat is the captain of test team",
#     "Dhoni is best finisher in the cricket",
#     "Rohit is best opener in ODI team"
# ]
# query = "Who is Virat Kohli?"

# # Get embeddings
# doc_vectors = embed_model.embed_documents(documents)
# query_vector = embed_model.embed_query(query)

# # Compute similarity
# scores = cosine_similarity([query_vector], doc_vectors)[0]

# # Find the most similar document
# best_index = np.argmax(scores)
# print(f"✅ Most similar text: \"{documents[best_index]}\" (score: {scores[best_index]:.4f})")
