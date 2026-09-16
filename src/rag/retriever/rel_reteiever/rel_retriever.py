import os
import httpx
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import tomllib
from dotenv import load_dotenv

with open('config.toml','rb') as f:
    config=tomllib.load(f)


load_dotenv()
qdrant_url=os.getenv('QDRANT_URL')
qdrant_api_key=os.getenv('QDRANT_API_KEY')

dense_model_name=config['data_ingestion']['dense_model']
sparse_model_name=config['data_ingestion']['sparse_model']
qdrant_collection_name=config['data_ingestion']['collection_name']
topk=config['retrieval']['topk']


def dense_search(query, collection_name=qdrant_collection_name, url=qdrant_url, api_key=qdrant_api_key, dense_model_name=dense_model_name, topk=topk):
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=0)
    client=QdrantClient(location=url, 
                        api_key=api_key,
                        check_compatibility=False,
                        limits=limits,
                        timeout=60.0)
    response=client.query_points(collection_name=collection_name,
                                 query=models.Document(text=query,
                                                       model=dense_model_name),
                                using='dense',
                                limit=topk)
    
    return [i.payload['text'] for i in response.points]




def sparse_search(query, collection_name=qdrant_collection_name, url=qdrant_url, api_key=qdrant_api_key, sparse_model_name=sparse_model_name, topk=topk):
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=0)
    client=QdrantClient(location=url, 
                        api_key=api_key,
                        check_compatibility=False,
                        limits=limits,
                        timeout=60.0)
    response=client.query_points(collection_name=collection_name,
                                 query=models.Document(text=query,
                                                       model=sparse_model_name),
                                using='sparse',
                                limit=topk)
    
    return [i for i in response.points]



def rrf_search(query, collection_name=qdrant_collection_name, url=qdrant_url, api_key=qdrant_api_key, dense_model_name=dense_model_name, sparse_model_name=sparse_model_name, topk=topk):
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=0)
    client=QdrantClient(location=url, 
                        api_key=api_key,
                        check_compatibility=False,
                        limits=limits,
                        timeout=60.0)
    response=client.query_points(collection_name=collection_name,
                                 prefetch=[models.Prefetch(query=models.Document(text=query, model=dense_model_name), 
                                                           using='dense',
                                                           limit=5),
                                            models.Prefetch(query=models.Document(text=query, model=sparse_model_name), 
                                                           using='sparse',
                                                           limit=5)],
                                query=models.FusionQuery(fusion=models.Fusion.RRF),
                                limit=topk)
    
    return [i.payload['text'] for i in response.points]




if __name__=='__main__':

    query='What laws are there against deepfakes'

    out=rrf_search(query=query)

    print(out)