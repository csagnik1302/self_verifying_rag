from verification import nli_verifier
import json

def label_aggregator(nli_output, threshold=0.7):

    final_aggregator_output=[]

    for i in nli_output:
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




def get_grounding_report(nli_output, EXPORT_PATH, threshold=0.7):

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

    PARENT=Path(__file__).resolve().parent
    SRC=PARENT.parent
    ROOT=SRC.parent

    EXPORT_PATH=ROOT/'reports'/'grounding_reports'/f'{datetime.now().strftime(r'%d_%m_%Y_%H_%M_%S')}'


    test=[{'claim': 'Adolf Hitler was an Axis leader and Führer (leader) of Germany.', 'support': ['China\n France\nOther major allies\nPoland\n Yugoslavia\n Greece\n Canada\n Netherlands\n Belgium\n Czechoslovakia\n India\n Australia\nPeople in World War II\nLeaders in World War II\nAxis leaders\nAdolf Hitler – Führer (leader) of Germany\n Hirohito – Emperor of Japan\n Benito Mussolini – Prime Minister and Duce of Italy\nAllied leaders\nJoseph Stalin – Leader of the Soviet Union\n Franklin D. Roosevelt – President of the United States\n Winston Churchill – Prime Minister of the United Kingdom\n Chiang Kai-shek – Leader of China', 'secure Italian entrance into the war were not fulfilled in the peace settlement. From 1922 to\n1925, the fascist movement led by Benito Mussolini seized power in Italy with a nationalist,\ntotalitarian, and class collaborationist agenda that abolished representative democracy,\nrepressed socialist, left-wing, and liberal forces and pursued an aggressive expansionist\nforeign policy aimed at making Italy a world power, promising the creation of a "New Roman\nEmpire".\nAdolf Hitler, after an unsuccessful attempt to overthrow the German government in 1923,\neventually became the chancellor of Germany in 1933 when President Paul von Hindenburg\nand the Reichstag appointed him. The Nazis soon abolished parliamentary democracy,\nespousing a radical, racially motivated revision of the world order, and began a massive\nrearmament campaign. Following Hindenburg\'s death in 1934, Hitler proclaimed himself\nFührer of Germany. France, seeking to secure its alliance with Italy, allowed Italy a free hand in'],
            'nli_probabilities': [{'entailment': 0.9990234375, 'neutral': 0.0008435249328613281, 'contradiction': 0.0001862049102783203}, {'entailment': 0.99951171875, 'neutral': 0.0003211498260498047, 'contradiction': 0.00034046173095703125}]}, 
            {'claim': 'Hitler made an unsuccessful attempt to overthrow the German government in 1923.', 'support': ['secure Italian entrance into the war were not fulfilled in the peace settlement. From 1922 to\n1925, the fascist movement led by Benito Mussolini seized power in Italy with a nationalist,\ntotalitarian, and class collaborationist agenda that abolished representative democracy,\nrepressed socialist, left-wing, and liberal forces and pursued an aggressive expansionist\nforeign policy aimed at making Italy a world power, promising the creation of a "New Roman\nEmpire".\nAdolf Hitler, after an unsuccessful attempt to overthrow the German government in 1923,\neventually became the chancellor of Germany in 1933 when President Paul von Hindenburg\nand the Reichstag appointed him. The Nazis soon abolished parliamentary democracy,\nespousing a radical, racially motivated revision of the world order, and began a massive\nrearmament campaign. Following Hindenburg\'s death in 1934, Hitler proclaimed himself\nFührer of Germany. France, seeking to secure its alliance with Italy, allowed Italy a free hand in'], 
            'nli_probabilities': [{'entailment': 0.96728515625, 'neutral': 0.003787994384765625, 'contradiction': 0.0287933349609375}]}, 
            {'claim': 'Hitler and the Nazi Party came to power in 1933 when he was appointed Chancellor of Germany.', 'support': ['major power, but perceived that Italy did have strong enough influence to alter the political\nsituation in Europe by placing the weight of its support onto one side or another, and sought\nto balance relations between the three.\nDanube alliance, dispute over Austria\nIn 1933, Adolf Hitler and the Nazi Party came to power in Germany. Hitler had advocated an\nalliance between Germany and Italy since the 1920s. Shortly after being appointed Chancellor\nof Germany, Hitler sent a personal message to Mussolini, declaring "admiration and homage"\nand declaring his anticipation of the prospects of German–Italian friendship and even alliance.\nHitler was aware that Italy held concerns over potential German land claims on South Tyrol,\nand assured Mussolini that Germany was not interested in South Tyrol. Hitler in Mein Kampf\nhad declared that South Tyrol was a non-issue considering the advantages that would be', 'secure Italian entrance into the war were not fulfilled in the peace settlement. From 1922 to\n1925, the fascist movement led by Benito Mussolini seized power in Italy with a nationalist,\ntotalitarian, and class collaborationist agenda that abolished representative democracy,\nrepressed socialist, left-wing, and liberal forces and pursued an aggressive expansionist\nforeign policy aimed at making Italy a world power, promising the creation of a "New Roman\nEmpire".\nAdolf Hitler, after an unsuccessful attempt to overthrow the German government in 1923,\neventually became the chancellor of Germany in 1933 when President Paul von Hindenburg\nand the Reichstag appointed him. The Nazis soon abolished parliamentary democracy,\nespousing a radical, racially motivated revision of the world order, and began a massive\nrearmament campaign. Following Hindenburg\'s death in 1934, Hitler proclaimed himself\nFührer of Germany. France, seeking to secure its alliance with Italy, allowed Italy a free hand in'], 
            'nli_probabilities': [{'entailment': 0.89794921875, 'neutral': 0.07232666015625, 'contradiction': 0.0294647216796875}, {'entailment': 0.97998046875, 'neutral': 0.007160186767578125, 'contradiction': 0.01279449462890625}]}, 
            {'claim': 'Following the death of President Paul von Hindenburg in 1934, Hitler proclaimed himself Führer of Germany.', 'support': ['secure Italian entrance into the war were not fulfilled in the peace settlement. From 1922 to\n1925, the fascist movement led by Benito Mussolini seized power in Italy with a nationalist,\ntotalitarian, and class collaborationist agenda that abolished representative democracy,\nrepressed socialist, left-wing, and liberal forces and pursued an aggressive expansionist\nforeign policy aimed at making Italy a world power, promising the creation of a "New Roman\nEmpire".\nAdolf Hitler, after an unsuccessful attempt to overthrow the German government in 1923,\neventually became the chancellor of Germany in 1933 when President Paul von Hindenburg\nand the Reichstag appointed him. The Nazis soon abolished parliamentary democracy,\nespousing a radical, racially motivated revision of the world order, and began a massive\nrearmament campaign. Following Hindenburg\'s death in 1934, Hitler proclaimed himself\nFührer of Germany. France, seeking to secure its alliance with Italy, allowed Italy a free hand in', "rearmament campaign. Following Hindenburg's death in 1934, Hitler proclaimed himself\nFührer of Germany. France, seeking to secure its alliance with Italy, allowed Italy a free hand in\nEthiopia, which Italy desired as a colonial possession. The situation was aggravated in early\n1935 when the Territory of the Saar Basin was legally reunited with Germany, and Hitler"], 
            'nli_probabilities': [{'entailment': 0.95703125, 'neutral': 0.0045166015625, 'contradiction': 0.03863525390625}, {'entailment': 0.9951171875, 'neutral': 0.0029449462890625, 'contradiction': 0.001773834228515625}]}]


    out=get_grounding_report(test, EXPORT_PATH=EXPORT_PATH)

    print(out)