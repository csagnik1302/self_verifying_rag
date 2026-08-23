from src.verification.verification import nli_verifier
from src.verification.grounding_report import label_aggregator
import json
from tqdm import tqdm

def eval_data(eval_data, model_name, hf_token, export_path, benchmark, threshold):

    keys=['claim','support']

    eval_parsed=[]
    out=[]

    for i in eval_data:
        input={j:i[j] for j in i if j in keys}
        eval_parsed.append(input)

    nli_output=nli_verifier(model_name=model_name, parsed_rag_output=eval_parsed, hf_token=hf_token)
    aggregator_output=label_aggregator(nli_output=nli_output, threshold=threshold)

    for i in tqdm(range(len(nli_output)), desc='EVALUATION RESULTS COMPILING'):
        temp_dict={}
        temp_dict['claim']=nli_output[i]['claim']
        temp_dict['support']=nli_output[i]['support']
        temp_dict['ground_truth']=eval_data[i]['hallucination_label']
        temp_dict['predicted_label']=aggregator_output[i]['nli_status']['label']
        temp_dict['predicted_confidence']=aggregator_output[i]['nli_status']['probability']
        temp_dict['full_nli_probabilities']=nli_output[i]['nli_probabilities']
        temp_dict['benchmark']=benchmark

        with open(export_path,'a',encoding='utf-8') as f:
            json.dump(temp_dict,f)
            f.write('\n')

        out.append(temp_dict)

    return out

    


if __name__=='__main__':

    from pathlib import Path
    import os
    from dotenv import load_dotenv
    from datetime import datetime

    load_dotenv()

    qdrant_url=os.getenv('QDRANT_URL')
    qdrant_api_key=os.getenv('QDRANT_API_KEY')
    hf_token=os.getenv("HF_ACCESS_TOKEN")
    model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'

    EVAL=Path(__file__).resolve().parent
    VER=EVAL.parent
    SRC=VER.parent
    ROOT=SRC.parent
    benchmark='HalluMix'

    EVAL_PATH=ROOT/'data'/f'{benchmark}'/'extracted'/'hallumix_eval.jsonl'

    threshold_list=[i/100 for i in range(40,95,5)]

    eval_data=[]
    with open(EVAL_PATH,'r',encoding='utf-8') as f:
        for i in f:
            eval_data.append(json.loads(i))

    for i in threshold_list:
        EXPORT_PATH=ROOT/'data'/f'{benchmark}'/'extracted'/'nli_eval'/f'{benchmark}_threshold_{i}.jsonl'
        out=eval_data(eval_data=eval_data, model_name=model_name, hf_token=hf_token, export_path=EXPORT_PATH, benchmark=benchmark, threshold=i)

    

