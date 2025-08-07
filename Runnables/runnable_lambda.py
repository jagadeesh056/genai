from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace 
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv 

load_dotenv() 

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write about the topic in 1 lines {topic}",
    input_variables=['topic']
)

joke_gen = RunnableSequence(prompt1, model, parser)

def word_counter(word):
    return len(word.split())

parallel_chain = RunnableParallel({
    'content' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_counter)
})

#parallel_chain = RunnableParallel({
#    'joke' : RunnablePassthrough(),
#    'function' : RunnableLambda(lambda x: len(x.split()))
#})

chain = RunnableSequence(joke_gen, parallel_chain)

print(chain.invoke({'topic': 'F1'}))