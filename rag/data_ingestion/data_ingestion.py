import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct


def loader(directory):

    loader=DirectoryLoader(directory,glob='*.pdf',loader_cls=PyMuPDFLoader)

    out=loader.load()

    return out



def chunker(documents,chunk_size=1000,chunk_overlap=200):

    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=['\n\n','\n',' ',''], length_function=len)

    split_docs=splitter.split_documents(documents)

    return split_docs



def embedding_creator(chunks,model,hf_access_token):

    model=SentenceTransformer(model_name_or_path=model,
                              token=hf_access_token,
                              model_kwargs={'attn_implementation':'flash_attention_2','torch_dtype':'bfloat16'})

    text_list=[i.page_content for i in chunks]

    embeddings=model.encode(text_list,normalize_embeddings=True)

    return embeddings


def vector_database(embeddings, chunks, cluster_url, api_key):

    user_input=int(input('What is your requirement:\n1. Get Database info\n2. Recreate the Database.\n Reply with 1 or 2: '))

    qdrant_client=QdrantClient(url=cluster_url,api_key=api_key)
    collection_name='rag_vector_db'

    if user_input==2:

        qdrant_client.recreate_collection(collection_name=collection_name,
                                        vectors_config=VectorParams(size=embeddings.shape[1],distance=Distance.COSINE))


        points=[PointStruct(id=i, vector=embeddings[i].tolist(), payload={**chunks[i].metadata,"page_content":chunks[i].page_content}) for i in range(len(chunks))]

        qdrant_client.upsert(collection_name=collection_name, points=points)

        string_out='Dataase Recreation Complete'

        return string_out


    elif user_input==1:

        return collection_name, qdrant_client



def main(directory,embedding_model,hf_access_token,cluster_url,cluster_api_key,chunk_size=1000,chunk_overlap=200):

    print('Document loading started')
    loaded_documents=loader(directory)
    print('Document loading complete')

    print('Document chunking started')
    chunks=chunker(loaded_documents,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    print('Document chunking complete')

    print('Embeddings creation started')
    embeddings=embedding_creator(chunks=chunks,model=embedding_model,hf_access_token=hf_access_token)
    print('Embeddings creation complete')

    print('DB Management started')
    output=vector_database(embeddings=embeddings, chunks=chunks, cluster_url=cluster_url, api_key=cluster_api_key)
    print('DB Management complete')

    print('Data Ingestion is complete')
    return output



if __name__=='__main__':

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
    

    