from langchain_community.document_loaders import PyPDFLoader 


loader = PyPDFLoader('sample.pdf')

docs = loader.load()
print(docs[3])