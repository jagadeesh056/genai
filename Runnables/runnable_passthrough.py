from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv 

load_dotenv() 

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

prompt2 = PromptTemplate(
    template= "Give me an small explanation {topic}",
    input_variables=['topic']
)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)
print(chain.invoke({'topic': 'quantum computing'}))