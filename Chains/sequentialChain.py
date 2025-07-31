from langchain_core.prompts import PromptTemplate 
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = 'Give me a report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Summarize the following {content} in 5 lines',
    input_variables=['content']
)

parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result1 = chain.invoke({'topic': 'formula1'})

print(result1)
chain.get_graph().print_ascii()