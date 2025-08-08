from langchain_community.document_loaders import TextLoader 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from dotenv import load_dotenv 

load_dotenv() 

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs[0])


prompt1 = PromptTemplate(
    template='Summarize above topic in 2 words {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt1 | model | parser 

res = chain.invoke({'topic': docs[0].page_content})

print(res)
