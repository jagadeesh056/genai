from langchain_core.prompts import PromptTemplate 
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field 
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class FeedBack(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give me the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=FeedBack)

prompt1 = PromptTemplate(
    template=(
        "You are a sentiment analysis classifier.\n"
        "Classify the sentiment of the following feedback as either 'positive' or 'negative'.\n"
        "Respond only in the following JSON format:\n"
        "{format_instruction}\n\n"
        "Feedback: {feedback}"
    ),
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

conditional_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template=("Write an appropriate response for positive feedback \n {feedback}"),
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template=("Write appropriate response for negative feedback \n {feedback}"),
    input_variables=['feedback']
)

parser = StrOutputParser()

result_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model | parser),
    (lambda x:x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Not found")
)

chain = conditional_chain | result_chain 

result = chain.invoke({'feedback': 'This is a wonderful watch'})
print(result)

