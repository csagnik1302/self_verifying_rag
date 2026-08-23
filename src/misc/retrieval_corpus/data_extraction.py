from src.misc.retrieval_corpus.utilities import export_pdf
import re
import wikipediaapi


def topic_page_creator(topic_iter):

    seen_titles=set()
    topic_keys=topic_iter.keys()

    out=[]

    for i in topic_keys:
        category_titles=topic_iter[i]

        for j in category_titles:
            title=f'Category:{j}'

            if title in seen_titles:
                continue

            temp_page=wiki.page(title)

            seen_titles.add(title)
            out.append(temp_page)

    return out




def topic_export(iter):

    exported_titles=set()
    
    for i in iter:

        category_members=i.categorymembers

        heading_keys=category_members.keys()

        for j in heading_keys:
            temp=category_members[j]

            if temp.ns!=wikipediaapi.Namespace.MAIN:
                continue
            if temp.title in exported_titles:
                continue

            temp_text=temp.text

            PATH=rf'D:\RAG Project\data\raw\{re.sub(r'[\\/:*?"<>|]','_',temp.title)}.pdf'
            export_pdf(temp_text,PATH)

            exported_titles.add(temp.title)

            print(f'Exported Topic: {re.sub(r'[\\/:*?"<>|]','_',temp.title)}')


def main(topic_iter):

    topic_list=topic_page_creator(topic_iter)
    topic_export(topic_list)

    print('All Topics Exported')





if __name__=='__main__':

    import os
    from pathlib import Path
    import ast

    PARENT=Path(__file__).resolve().parent
    MISC=PARENT.parent
    SRC=MISC.parent
    ROOT=SRC.parent

    topic_set_path=ROOT/'src'/'misc'/'resources'/'topic_set.txt'

    wiki=wikipediaapi.Wikipedia(user_agent='Project (csagnik180@gmail.com)', language='en')

    with open(topic_set_path,'r',encoding='utf-8') as f:
        topics_temp=f.read()

    topics = ast.literal_eval(topics_temp)

    topic_list=main(topics)

