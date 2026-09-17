import sys
import os
from pathlib import Path

ORC=Path(__file__).resolve().parent
RAG=ORC.parent
SRC=RAG.parent
ROOT=SRC.parent

generator_dir=ROOT/'src'/'rag'/'generator'
retr_dir=ROOT/'src'/'rag'/'retriever'
router_dir=ROOT/'src'/'rag'/'router'
utils_dir=ROOT/'src'/'rag'/'utils'

sys.path.insert(0,str(generator_dir))
sys.path.insert(0,str(retr_dir))
sys.path.insert(0,str(router_dir))
sys.path.insert(0,str(utils_dir))


from router import route
from rel_retriever import dense_search, sparse_search, rrf_search
from reranker import reranker
from generator import generator
from utils import State, route_to_retr

from langgraph.graph import StateGraph, START, END
builder=StateGraph(state_schema=State)
builder.add_node('route', route)
builder.add_node('dense_search', dense_search)
builder.add_node('sparse_search', sparse_search)
builder.add_node('rrf_search', rrf_search)
builder.add_node('reranker', reranker)
builder.add_node('generator', generator)

builder.add_edge(START, 'route')
builder.add_conditional_edges('route', route_to_retr)
builder.add_edge('dense_search', 'reranker')
builder.add_edge('sparse_search', 'reranker')
builder.add_edge('rrf_search', 'reranker')
builder.add_edge('reranker', 'generator')
builder.add_edge('generator', END)


graph=builder.compile()

# png_bytes=graph.get_graph().draw_mermaid_png()

# export_dir=ROOT/'src'/'rag'/'orchestrator'/'misc'/'graph.png'
# with open(export_dir,'wb') as f:
#     f.write(png_bytes)


out=response=graph.invoke({"query":"What laws are there against deepfakes"})
print(out['response'])