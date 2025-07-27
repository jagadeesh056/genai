from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

envValue = os.getenv('HUGGING_FACE_API_KEY')

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="task-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token= envValue
)

model = ChatHuggingFace(llm = llm)

chat_history = [
    SystemMessage(content="you are assistant helper")
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))
    print('AI: ', result.content)

print(chat_history)