from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv 

load_dotenv() 

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    
)