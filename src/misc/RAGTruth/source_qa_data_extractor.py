import json
from pathlib import Path

RAGTRUTH=Path(__file__).resolve().parent
MISC=RAGTRUTH.parent
SRC=MISC.parent
ROOT=SRC.parent

RAW_DATA_PATH=ROOT/'data'/'RAGTruth'/'raw'/'source_info.jsonl'
EXTRACTED_RAW_DATA_PATH=ROOT/'data'/'RAGTruth'/'raw'/'source_info_qa_extracted.jsonl'

data=[]
with open(RAW_DATA_PATH,'r',encoding='utf-8') as f:
    for i in f:
        data.append(json.loads(i))

data_extracted=[]

for i in data:
    if i['task_type']=="QA":
        data_extracted.append(i)



with open(EXTRACTED_RAW_DATA_PATH,'w',encoding='utf-8') as f:
    for i in data_extracted:
        json.dump(i,f)
        f.write('\n')
