from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
from dotenv import load_dotenv
import os
load_dotenv()

envValue = os.getenv('HUGGING_FACE_API_KEY')

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token= envValue
)

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content="You are assitant helper"),
    HumanMessage(content="Tell me about AI in 2 lines")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)