## WARNING: Gotta check whether source qa data is already created or not (just run the source qa extracted script for once)

import json
from pathlib import Path

RAGTRUTH=Path(__file__).resolve().parent
MISC=RAGTRUTH.parent
SRC=MISC.parent
ROOT=SRC.parent

SOURCE_EXTRACTED_DATA_PATH=ROOT/'data'/'RAGTruth'/'raw'/'source_info_qa_extracted.jsonl'
RESPONSE_RAW_DATA_PATH=ROOT/'data'/'RAGTruth'/'raw'/'response.jsonl'
RESPONSE_EXTRACTED_DATA_PATH=ROOT/'data'/'RAGTruth'/'raw'/'response_extracted.jsonl'


source_data=[]
with open(SOURCE_EXTRACTED_DATA_PATH,'r',encoding='utf-8') as f:
    for i in f:
        source_data.append(json.loads(i))
