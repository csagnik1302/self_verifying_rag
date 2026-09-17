from typing import TypedDict, Literal

class State(TypedDict):
    query: str
    route: dict
    retr_docs: dict
    rerank_docs: dict
    response: str


def route_to_retr(state:State) -> Literal['dense_search', 'sparse_search', 'rrf_search']:

    router_response=state['route']['label']

    if router_response=='factual':
        return 'dense_search'
    elif router_response=='abstractive':
        return 'sparse_search'
    elif router_response=='multi-hop':
        return 'rrf_search'