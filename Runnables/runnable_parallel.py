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
    template="Write 1 line about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Write 2 advantages of {topic}", 
    input_variables=['topic']
)

parser = StrOutputParser() 

parallel_chain = RunnableParallel({
    'context': RunnableSequence(prompt1, model, parser),
    'advantages' : RunnableSequence(prompt2, model, parser)
})

print(parallel_chain.invoke({'topic': 'diamond'}))