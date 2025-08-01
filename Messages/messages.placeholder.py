from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()

# chat template
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = []

# load chat_history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

#prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund'})
print(prompt)