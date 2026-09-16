import os
import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
import tomllib
import uuid
from tqdm import tqdm

load_dotenv()
qdrant_url=os.getenv('QDRANT_URL')
qdrant_api_key=os.getenv('QDRANT_API_KEY')
hf_token=os.getenv('HF_ACCESS_TOKEN')

with open('config.toml','rb') as f:
    config=tomllib.load(f)

dense_model_name=config['data_ingestion']['dense_model']
sparse_model_name=config['data_ingestion']['sparse_model']
qdrant_collection_name=config['data_ingestion']['collection_name']
BATCH_SIZE=config['data_ingestion']['BATCH_SIZE']



def loader(directory):
    loader=DirectoryLoader(directory,glob='*.pdf',loader_cls=PyMuPDFLoader)
    out=loader.load()
    return out



def chunker(documents,chunk_size=1000,chunk_overlap=100):
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=['\n\n','\n',' ',''], length_function=len)
    split_docs=splitter.split_documents(documents)
    text_set=[i.page_content for i in split_docs]
    metadata_set=[i.metadata for i in split_docs]
    return text_set, metadata_set


def dense_emb_dim(model_name=dense_model_name):
    dense_embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    dense_embeddings = dense_embedding_model.embed_documents(['text'])
    return len(dense_embeddings[0])


def batch_upsert(client,points,collection_name,batch=BATCH_SIZE):

    for i in tqdm(range(0,len(points),batch)):
        points_batch=points[i:i+batch]
        client.upsert(collection_name=collection_name, 
                      points=points_batch)






def vector_db(text_set, metadata_set, dense_embed_dim, collection_name=qdrant_collection_name, qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key, sparse_name=sparse_model_name, dense_name=dense_model_name):

    # Disable HTTP keep-alive connection pooling to prevent socket reuse [WinError 10054] on Windows
    limits = httpx.Limits(max_connections=BATCH_SIZE, max_keepalive_connections=0)

    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        check_compatibility=False,
        limits=limits,
        timeout=60.0)


    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config={'dense': models.VectorParams(distance=models.Distance.COSINE, size=dense_embed_dim)},
        sparse_vectors_config={'sparse': models.SparseVectorParams(modifier=models.Modifier.IDF)})


    points=[models.PointStruct(id=uuid.uuid4().hex,
                                vector={'dense': models.Document(text=text,
                                                                  model=dense_name),
                                        'sparse': models.Document(text=text,
                                                                  model=sparse_name)},
                                payload={'text':text,
                                        'creation_date':metadata['creationdate'],
                                        'source':metadata['source'],
                                        'path':metadata['file_path'],
                                        'total_source_pages':metadata['total_pages'],
                                        'source_page_number':metadata['page']}) 

            for text, metadata in zip(text_set,metadata_set)]


    batch_upsert(client=qdrant_client, points=points, collection_name=collection_name)




def data_ingestion(directory):
    doc_loaded=loader(directory=directory)
    print('Document Loading complete')
    text_chunk, metadata_chunk=chunker(documents=doc_loaded)
    print('Document Chunking complete')
    dense_embed_dim=dense_emb_dim()
    print('Vector DB creation in progress')
    vector_db(text_set=text_chunk,dense_embed_dim=dense_embed_dim, metadata_set=metadata_chunk)
    print('Vector DB creation complete')

    print('ALL PROCESSES COMPLETE')


    

if __name__=='__main__':

    import os
    from dotenv import load_dotenv
    import tomllib

    load_dotenv()

    PATH=r'data\raw'

    data_ingestion(PATH)

