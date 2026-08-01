from ..retriever.retriever import retriever

prompt=input('What is your question ? : ')

def generator(query, hf_access_token, cluster_url, cluster_api_key):
    
    retr_list=retriever(query=query, hf_access_token=hf_access_token, cluster_url=cluster_url, cluster_api_key=cluster_api_key)

    

