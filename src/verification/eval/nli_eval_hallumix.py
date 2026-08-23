from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pathlib import Path

EVAL=Path(__file__).resolve().parent
VER=EVAL.parent
SRC=VER.parent
ROOT=SRC.parent

def eval_report(data):
    y_true=[]
    y_pred=[]

    for i in data:
        if i['ground_truth']==True:
            y_true.append(1)
        else:
            y_true.append(0)

        if i['predicted_label']=='entailment':
            y_pred.append(1)
        else:
            y_pred.append(0)

    out=classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)

    return out


def plot(data,benchmark):

    x=[]
    precision_0=[]
    precision_1=[]
    recall_0=[]
    recall_1=[]
    f1_score_0=[]
    f1_score_1=[]

    for i in data:
        x.append(i['threshold'])
        precision_0.append(i['0']['precision'])
        recall_0.append(i['0']['recall'])
        f1_score_0.append(i['0']['f1-score'])
        precision_1.append(i['1']['precision'])
        recall_1.append(i['1']['recall'])
        f1_score_1.append(i['1']['f1-score'])

    met=[precision_0,
         precision_1,
         recall_0,
         recall_1,
         f1_score_0,
         f1_score_1]

    name=['precision_0',
        'precision_1',
        'recall_0',
        'recall_1',
        'f1_score_0',
        'f1_score_1']


    for i in range(len(met)):
        EXPORT_PATH=ROOT/'reports'/'nli_benchmark_plots'/f'{benchmark}'/f'{name[i]}_{benchmark}.png'
        plt.plot(x, met[i])
        plt.savefig(EXPORT_PATH, dpi=300, bbox_inches='tight')
        plt.close()



if __name__=='__main__':

    import json
    import os
    from datetime import datetime

    benchmark='HalluMix'

    threshold_list=[i/100 for i in range(40,95,5)]

    eval_output=[]
    for i in threshold_list:
        EVAL_PATH=ROOT/'data'/f'{benchmark}'/'extracted'/'nli_eval'/f'{benchmark}_threshold_{i}.jsonl'

        eval_temp=[]
        with open(EVAL_PATH,'r',encoding='utf-8') as f:
            for j in f:
                eval_temp.append(json.loads(j))

        out=eval_report(eval_temp)
        out['threshold']=i
        eval_output.append(out)

        plot(eval_output,benchmark=benchmark)

    EXPORT_PATH=ROOT/'reports'/'nli_benchmark_report'/f'{benchmark}'/f'{datetime.now().strftime(r'%d_%m_%Y_%H_%M_%S')}.jsonl'

    with open(EXPORT_PATH,'w',encoding='utf-8') as f:
        for i in eval_output:
            json.dump(i,f)
            f.write('\n')

