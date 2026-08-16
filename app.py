import streamlit as st
from pathlib import Path
from src.rag.augmented_generation.generator import retrieval_augmented_generation
import os
import json
from dotenv import load_dotenv


load_dotenv()

ROOT=Path(__file__).resolve().parent

USER_TEMPLATE=ROOT/'src'/'rag'/'augmented_generation'/'prompt'/'user_template.txt'
SYSTEM_TEMPLATE=ROOT/'src'/'rag'/'augmented_generation'/'prompt'/'system_template.txt'
OUTPUT_PATH=ROOT/'src'/'rag'/'augmented_generation'/'samples'/'outputs'/'sample_augmented_generation_output.json'

hf_access_token=os.getenv('HF_ACCESS_TOKEN')
qdrant_cluster_url=os.getenv('QDRANT_URL')
qdrant_api_key=os.getenv('QDRANT_API_KEY')
gemini_api_key=os.getenv('GEMINI_API_KEY')

with open(USER_TEMPLATE, 'r') as f:
    user_template=f.read()

with open(SYSTEM_TEMPLATE, 'r') as f:
    system_prompt=f.read()



INTRO='Hi I am RAGGY, ask me anything about Football, World History, Indian Economy or Space Exploration'

st.write(INTRO)

query=st.text_input("What is your Question")

out=retrieval_augmented_generation(query=query, 
                            user_template=user_template, 
                            system_prompt=system_prompt, 
                            hf_access_token=hf_access_token, 
                            cluster_url=qdrant_cluster_url, 
                            cluster_api_key=qdrant_api_key,
                            gemini_api_key=gemini_api_key)


st.write(out)