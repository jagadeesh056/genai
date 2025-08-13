from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('text.pdf')

docs = loader.load()

text = """One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.It's obviously true that the returns for performance are superlinear in business."""

splitter = CharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 0,
    separator = ""
)

result = splitter.split_documents(docs)
print(result[51].page_content)