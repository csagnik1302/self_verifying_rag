from utilities import export_pdf
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

    wiki=wikipediaapi.Wikipedia(user_agent='Project (csagnik180@gmail.com)', language='en')


    topics = {'Sports — Olympics / Football World Cup': [
            'Olympic Games',
            'Olympics stubs',
            'FIFA World Cup',
            'Summer Olympics',
            'Winter Olympics',
            'Olympic sports',
            'History of the Olympic Games',
            'FIFA World Cup finals',
            'Association football competitions',
        ],
        'World History (20th century)': [
            '20th-century history',
            'Wars by century',
            'Cold War',
            'World War I',
            'World War II',
            'Decolonization',
            '20th-century military history',
            'History of the United Nations',
            '20th-century conflicts',
        ],
        'Economy of India / Banking & Finance in India': [
            'Economy of India',
            'Banking in India',
            'Reserve Bank of India',
            'Indian rupee',
            'Economic history of India',
            'Financial services companies of India',
            'Stock exchanges in India',
            'Taxation in India',
            'Indian economists',
        ],
        'Space Exploration': [
            'Space exploration',
            'Space missions',
            'Human spaceflight',
            'Space probes',
            'Space agencies',
            'Artificial satellites',
            'NASA programs',
            'Mars exploration',
            'Moon landings',
            'Space stations',
        ],
    }


    topic_list=main(topics)

