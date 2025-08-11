from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

prompt1 = PromptTemplate(
    template='What are we talking about {topic} \n {text}', 
    input_variables=['topic', 'text']
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

url = "https://www.amazon.in/Puma-Unisex-Adult-Ferrari-Black-Black-White-Sneaker/dp/B0CKJN74TG"
loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt1 | model | parser

print(chain.invoke({'topic': 'tell me product name', 'text': docs[0].page_content}))
