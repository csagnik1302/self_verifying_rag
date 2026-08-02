from ..retriever.retriever import retriever
from google import genai


def build_user_prompt(query, user_template, hf_access_token, cluster_url, cluster_api_key):

    retr_list=retriever(query=query, hf_access_token=hf_access_token, cluster_url=cluster_url, cluster_api_key=cluster_api_key)

    if len(retr_list)!=0:

        context_string=''

        for i in retr_list:

            id=i['id']
            source_temp=i['payload']['source']
            source=source_temp.split('\\')[-1].replace('.pdf','')
            page_content=i['payload']['page_content'].replace(r'\n',' ').strip()

            if retr_list.index(i)==len(retr_list)-1:
                context_temp=fr'Page ID: {id}, Page source: {source}, Page content: {page_content}'
            else:
                context_temp=fr'Page ID: {id}, Page source: {source}, Page content: {page_content}'+'\n\n'

            context_string+=context_temp

        query_final_temp=user_template.replace("[QUERY]",query)
        query_final=query_final_temp.replace("[SOURCES]",context_string)

    else:

        query_final='EMPTY'

    return query_final
    

def generator(prompt, system_prompt, gemini_api_key, generator_model= 'gemini-3.6-flash'):

    if prompt=='EMPTY':

        output='No relevant documents found to the provided query'

    else:

        gemini_client=genai.Client(api_key=gemini_api_key)
        interaction=gemini_client.interactions.create(model=generator_model, system_instruction=system_prompt, input=prompt, generation_config={'temperature':0.0})
        # System prompt goes to system instruction, user prompt goes to input

        output=interaction.output_text

    return output

def retrieval_augmented_generation(query, user_template, system_prompt, hf_access_token, cluster_url, cluster_api_key, gemini_api_key, generator_model= 'gemini-3.6-flash'):

    user_prompt=build_user_prompt(query, user_template, hf_access_token, cluster_url, cluster_api_key)
    output=generator(user_prompt, system_prompt, gemini_api_key, generator_model=generator_model)

    return output




if __name__=='__main__':

    import os
    from dotenv import load_dotenv

    load_dotenv()

    hf_access_token=os.getenv('HF_ACCESS_TOKEN')
    qdrant_cluster_url=os.getenv('QDRANT_URL')
    qdrant_api_key=os.getenv('QDRANT_API_KEY')
    gemini_api_key=os.getenv('GEMINI_API_KEY')

    with open(r'D:\RAG Project\src\rag\augmented_generation\prompt\user_template.txt', 'r') as f:
        user_template=f.read()

    with open(r'D:\RAG Project\src\rag\augmented_generation\prompt\system_template.txt', 'r') as f:
        system_prompt=f.read()

    query=input('What is your question: ')

    out=augmented_generation(query=query, 
                             user_template=user_template, 
                             system_prompt=system_prompt, 
                             hf_access_token=hf_access_token, 
                             cluster_url=qdrant_cluster_url, 
                             cluster_api_key=qdrant_api_key,
                             gemini_api_key=gemini_api_key)

    with open(r'D:\RAG Project\src\rag\augmented_generation\samples\prompts\sample_augmented_generation_output.txt','w', encoding='utf-8') as f:
        f.write(out)

    print(out)





    

