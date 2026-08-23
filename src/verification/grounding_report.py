from src.verification.verification import nli_verifier, rag_output_parser
import json
from tqdm import tqdm

def label_aggregator(nli_output, threshold=0.7):

    final_aggregator_output=[]

    for i in tqdm(nli_output, desc='AGGREGATION PROCESSING'):
        entailment_probabilities=[]
        neutral_probabilities=[]
        contradiction_probabilities=[]
        aggregator_output={}

        claim=i['claim']
        support=i['support']
        nli_labels=i['nli_probabilities']

        aggregator_output['claim']=claim
        aggregator_output['support']=support

        for j in nli_labels:
            entailment_probabilities.append(j['entailment'])
            neutral_probabilities.append(j['neutral'])
            contradiction_probabilities.append(j['contradiction'])

        entailment_label=any(k > threshold for k in entailment_probabilities)
        contradiction_label=any(k > threshold for k in contradiction_probabilities)

        if entailment_label==True:
            aggregator_output['nli_status']={'label':'entailment', 'probability':max(entailment_probabilities)}
        elif contradiction_label==True:
            aggregator_output['nli_status']={'label':'contradiction', 'probability':max(contradiction_probabilities)}
        else:
            aggregator_output['nli_status']={'label':'neutral', 'probability':max(neutral_probabilities)}

        final_aggregator_output.append(aggregator_output)
        

    return final_aggregator_output




def get_grounding_report(model_name, parsed_output, hf_token, EXPORT_PATH, threshold=0.7):

    nli_output=nli_verifier(model_name=model_name, parsed_rag_output=parsed_output, hf_token=hf_token)

    aggregator_output=label_aggregator(nli_output=nli_output, threshold=threshold)
    num_claims=len(aggregator_output)

    num_entailed=0
    num_neutral=0
    num_contradicted=0

    for i in aggregator_output:
        nli_data=i['nli_status']['label']

        if nli_data=='entailment':
            num_entailed+=1
        if nli_data=='neutral':
            num_neutral+=1
        if nli_data=='contradiction':
            num_contradicted+=1


    grounding_score=num_entailed/num_claims
    has_contradictions=num_contradicted>0

    out={'claims':aggregator_output, 'num_claims':num_claims, 'num_entailed':num_entailed, 'num_neutral':num_neutral, 'num_contradicted':num_contradicted, 'grounding_score':grounding_score, 'has_contradictions':has_contradictions}

    with open(EXPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out,f,indent=4)
            
    return out



if __name__=='__main__':

    from pathlib import Path
    from datetime import datetime
    import os
    from dotenv import load_dotenv

    load_dotenv()

    PARENT=Path(__file__).resolve().parent
    SRC=PARENT.parent
    ROOT=SRC.parent

    EXPORT_PATH=ROOT/'reports'/'grounding_reports'/f'{datetime.now().strftime(r'%d_%m_%Y_%H_%M_%S')}'

    qdrant_url=os.getenv('QDRANT_URL')
    qdrant_api_key=os.getenv('QDRANT_API_KEY')
    hf_token=os.getenv("HF_ACCESS_TOKEN")
    model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'


    test=[{'claim': 'Adolf Hitler was an Axis leader and Führer (leader) of Germany.', 'page_ids': [16073, 23592]}, 
          {'claim': 'Hitler made an unsuccessful attempt to overthrow the German government in 1923.', 'page_ids': [23592]}, 
          {'claim': 'Hitler and the Nazi Party came to power in 1933 when he was appointed Chancellor of Germany.', 'page_ids': [2305, 23592]}, 
          {'claim': 'Following the death of President Paul von Hindenburg in 1934, Hitler proclaimed himself Führer of Germany.', 'page_ids': [23592, 23593]}]

    parsed_output=rag_output_parser(output_sent_list=test, qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)
    out=get_grounding_report(model_name=model_name, parsed_output=parsed_output, hf_token=hf_token, EXPORT_PATH=EXPORT_PATH)

    print(out)