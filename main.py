from src.rag.augmented_generation.generator import retrieval_augmented_generation
import os
from dotenv import load_dotenv

load_dotenv()

hf_access_token=os.getenv('HF_ACCESS_TOKEN')
qdrant_cluster_url=os.getenv('QDRANT_URL')
qdrant_api_key=os.getenv('QDRANT_API_KEY')
gemini_api_key=os.getenv('GEMINI_API_KEY')

with open(r'src\rag\augmented_generation\prompt\user_template.txt', 'r') as f:
    user_template=f.read()

with open(r'src\rag\augmented_generation\prompt\system_template.txt', 'r') as f:
    system_prompt=f.read()

query=input('What is your question: ')

out=retrieval_augmented_generation(query=query, 
                            user_template=user_template, 
                            system_prompt=system_prompt, 
                            hf_access_token=hf_access_token, 
                            cluster_url=qdrant_cluster_url, 
                            cluster_api_key=qdrant_api_key,
                            gemini_api_key=gemini_api_key)

print(out)
