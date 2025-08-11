from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

#docs = loader.load()
docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)
    
print(len(docs))
