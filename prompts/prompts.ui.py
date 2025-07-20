from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

envValue = os.getenv("HUGGING_FACE_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=envValue,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1
)

chat_model = ChatHuggingFace(llm=llm)

st.header("Research Tool")


user_input = st.text_input("Enter your prompt")

if st.button("Summarize"):
    result = chat_model.invoke(user_input)
    st.write(result.content)