from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vector_database_info(qdrant_url, api_key):

    qdrant_client=QdrantClient(url=qdrant_url,api_key=api_key,timeout=60)
    collection_name='rag_vector_db'

    return qdrant_client, collection_name


def rag_output_parser(output_sent_list, qdrant_url, qdrant_api_key):

    client, collection_name= vector_database_info(qdrant_url=qdrant_url, api_key=qdrant_api_key)

    out=[]

    for i in output_sent_list:
        claim=i['claim']
        page_ids=i['page_ids']

        support_temp=client.retrieve(collection_name=collection_name, ids=page_ids)
        support=[i.payload['page_content'] for i in support_temp]

        temp={'claim': claim, 'support':support}
        out.append(temp)

    return out


def nli_verifier(model_name, parsed_rag_output, hf_token):

    tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name, token=hf_token)
    model=AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, token=hf_token).to(device)

    trial_list=[]

    for i in parsed_rag_output:
        hypothesis=i['claim']
        premise_list=i['support']

        temp_dict={'claim': hypothesis, 'support': premise_list, 'nli_probabilities':[]}

        for j in premise_list:
            input=tokenizer(j, hypothesis, truncation=True, return_tensors='pt').to(device)
            output=model(**input)

            prediction=torch.softmax(output["logits"][0], -1).tolist()
            prediction_labels=list(model.config.id2label.values())
            prediction_output={}

            for k in range(len(prediction_labels)):
                prediction_output[prediction_labels[k]]=prediction[k]

            temp_dict['nli_probabilities'].append(prediction_output)

        trial_list.append(temp_dict)

    return trial_list


    


if __name__=='__main__':

    test_cases = [{'claim': 'Adolf Hitler was an Axis leader and Führer (leader) of Germany.', 'page_ids': [16073, 23592]}, 
                {'claim': 'Hitler made an unsuccessful attempt to overthrow the German government in 1923.', 'page_ids': [23592]}, 
                {'claim': 'Hitler and the Nazi Party came to power in 1933 when he was appointed Chancellor of Germany.', 'page_ids': [2305, 23592]}, 
                {'claim': 'Following the death of President Paul von Hindenburg in 1934, Hitler proclaimed himself Führer of Germany.', 'page_ids': [23592, 23593]}]

    import os
    from dotenv import load_dotenv

    load_dotenv()

    qdrant_url=os.getenv('QDRANT_URL')
    qdrant_api_key=os.getenv('QDRANT_API_KEY')
    hf_token=os.getenv("HF_ACCESS_TOKEN")
    model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'

    out=rag_output_parser(test_cases,qdrant_url, qdrant_api_key)
    out1=nli_verifier(model_name=model_name, parsed_rag_output=out, hf_token=hf_token)

    print(out1)