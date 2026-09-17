from transformers import pipeline
import tomllib

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


router_model_name=config['router']['model']

router=pipeline("text-classification", model=router_model_name)

def route(state:State ,router=router) -> State:
    prediction=router(state['query'])
    return {'query':state['query'], 'route':prediction[0]}