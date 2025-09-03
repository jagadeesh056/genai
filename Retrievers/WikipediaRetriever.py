from langchain_community.retrievers import WikipediaRetriever 

retriever = WikipediaRetriever(top_k_results=2, lang="en")
query = "Who is Ramanujan"
docs = retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"Search results: {i}")
    print(f"page content: {doc.page_content}")