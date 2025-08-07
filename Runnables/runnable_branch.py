from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv 

load_dotenv() 

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Give me content on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= "Write summarized version of {topic}",
    input_variables=['topic']
)

content_gen_chain = prompt1 | model | parser

branches = RunnableBranch(
    (lambda x: len(x.split()) > 250, prompt2 | model | parser),
    RunnablePassthrough()
)

chain = RunnableSequence(content_gen_chain, branches)

print(chain.invoke({'topic': 'avatar movie'}))