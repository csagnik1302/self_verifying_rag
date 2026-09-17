import os
import httpx
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import tomllib
from dotenv import load_dotenv

#
import sys
from pathlib import Path

ORC=Path(__file__).resolve().parent
RAG=ORC.parent
SRC=RAG.parent
ROOT=SRC.parent

utils_dir=ROOT/'src'/'rag'/'utils'

sys.path.insert(0,str(utils_dir))
from utils import State
#

with open('config.toml','rb') as f:
    config=tomllib.load(f)


load_dotenv()
qdrant_url=os.getenv('QDRANT_URL')
qdrant_api_key=os.getenv('QDRANT_API_KEY')

dense_model_name=config['data_ingestion']['dense_model']
sparse_model_name=config['data_ingestion']['sparse_model']
qdrant_collection_name=config['data_ingestion']['collection_name']
dense_topk=config['retrieval']['dense_topk']
sparse_topk=config['retrieval']['sparse_topk']
rrf_dense_topk=config['retrieval']['rrf_dense_topk']
rrf_sparse_topk=config['retrieval']['rrf_sparse_topk']
rrf_topk=config['retrieval']['rrf_topk']


def dense_search(state: State, collection_name=qdrant_collection_name, url=qdrant_url, api_key=qdrant_api_key, dense_model_name=dense_model_name, dense_topk=dense_topk) -> State:
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=0)
    client=QdrantClient(location=url, 
                        api_key=api_key,
                        check_compatibility=False,
                        limits=limits,
                        timeout=60.0)
    response=client.query_points(collection_name=collection_name,
                                 query=models.Document(text=state['query'],
                                                       model=dense_model_name),
                                using='dense',
                                limit=dense_topk)
    
    return {'query':state['query'], 'retr_docs':[{'id':i.id, 'text':i.payload['text'], 'score':i.score, 'source':'dense'} for i in response.points]}




def sparse_search(state: State, collection_name=qdrant_collection_name, url=qdrant_url, api_key=qdrant_api_key, sparse_model_name=sparse_model_name, sparse_topk=sparse_topk) -> State:
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=0)
    client=QdrantClient(location=url, 
                        api_key=api_key,
                        check_compatibility=False,
                        limits=limits,
                        timeout=60.0)
    response=client.query_points(collection_name=collection_name,
                                 query=models.Document(text=state['query'],
                                                       model=sparse_model_name),
                                using='sparse',
                                limit=sparse_topk)
    
    return {'query':state['query'], 'retr_docs':[{'id':i.id, 'text':i.payload['text'], 'score':i.score, 'source':'sparse'} for i in response.points]}



def rrf_search(state: State, rrf_dense_topk=rrf_dense_topk, rrf_sparse_topk=rrf_sparse_topk, collection_name=qdrant_collection_name, url=qdrant_url, api_key=qdrant_api_key, dense_model_name=dense_model_name, sparse_model_name=sparse_model_name, rrf_topk=rrf_topk) -> State:
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=0)
    client=QdrantClient(location=url, 
                        api_key=api_key,
                        check_compatibility=False,
                        limits=limits,
                        timeout=60.0)
    response=client.query_points(collection_name=collection_name,
                                 prefetch=[models.Prefetch(query=models.Document(text=state['query'], model=dense_model_name), 
                                                           using='dense',
                                                           limit=rrf_dense_topk),
                                            models.Prefetch(query=models.Document(text=state['query'], model=sparse_model_name), 
                                                           using='sparse',
                                                           limit=rrf_sparse_topk)],
                                query=models.FusionQuery(fusion=models.Fusion.RRF),
                                limit=rrf_topk)
    
    return {'query':state['query'], 'retr_docs':[{'id':i.id, 'text':i.payload['text'], 'score':i.score, 'source':'rrf'} for i in response.points]}




if __name__=='__main__':

    query='What laws are there against deepfakes'

    out=rrf_search(query=query)

    print(out)