from data_ingestion import main


PATH=r'data\raw'

embedding_model='BAAI/bge-m3'

with open(r'.venv/HF/access_token_key.txt','r') as f:
    hf_token=f.read()

with open(r'.venv/Qdrant/qdrant_API.txt','r') as f:
    cluster_api_key=f.read()

with open(r'.venv/Qdrant/qdrant_endpoint.txt','r') as f:
    cluster_url=f.read()

out=main(directory=PATH, embedding_model=embedding_model, hf_access_token=hf_token, cluster_api_key=cluster_api_key, cluster_url= cluster_url)

print(out)