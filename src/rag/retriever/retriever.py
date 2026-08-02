from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient



def query_embedding(query,hf_access_token,model_name):

    model=SentenceTransformer(model_name_or_path=model_name, token=hf_access_token, model_kwargs={'attn_implementation':'flash_attention_2','torch_dtype':'bfloat16'})

    query_embedding=model.encode(query,normalize_embeddings=True)

    return query_embedding




def vector_database_info(cluster_url, api_key):

    qdrant_client=QdrantClient(url=cluster_url,api_key=api_key,timeout=60)
    collection_name='rag_vector_db'

    return qdrant_client, collection_name




def retrieval(query_embed, topk, cluster_url, cluster_api_key):

    client, collection_name=vector_database_info(cluster_url=cluster_url, api_key=cluster_api_key)

    hits=client.query_points(
        collection_name=collection_name,
        query=query_embed.tolist(),
        limit=topk,
        score_threshold=0.5
    ).points

    out=[]

    for i in hits:
        out.append(i.model_dump())

    return out



def retriever(query, hf_access_token, cluster_url, cluster_api_key, model_name='BAAI/bge-m3', topk=5):

    query_embed=query_embedding(query=query, hf_access_token=hf_access_token, model_name=model_name)

    out=retrieval(query_embed=query_embed, topk=topk, cluster_url=cluster_url, cluster_api_key=cluster_api_key)

    return out

    



if __name__=='__main__':

    import os
    from dotenv import load_dotenv

    load_dotenv()

    query='Mahatma Gandhi was Great'
    model=r'BAAI/bge-m3'

    hf_token=os.getenv('HF_ACCESS_TOKEN')
    cluster_api_key=os.getenv('QDRANT_API_KEY')
    cluster_url=os.getenv('QDRANT_URL')

    out=retriever(query=query, hf_access_token=hf_token, cluster_url=cluster_url, cluster_api_key=cluster_api_key)

    print(out)