import argparse
import jsonlines
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--abstract-retrieval', type=str, required=True)
args = parser.parse_args()

dataset = {data['id']: data for data in jsonlines.open(args.dataset)}

hit_one = 0
hit_all = 0
total = 0
yt = []
yp = []
for retrieval in jsonlines.open(args.abstract_retrieval):
    total += 1
    data = dataset[retrieval['claim_id']]

    pred_doc_ids = set(retrieval['doc_ids'][:3])
    true_doc_ids = set(map(int, data['evidence'].keys()))


    if pred_doc_ids.intersection(true_doc_ids) or not true_doc_ids:
        hit_one += 1
    if pred_doc_ids.issuperset(true_doc_ids):
        hit_all += 1
    for id in pred_doc_ids:
        yp.append(1)
        yt.append(id in true_doc_ids)
    for id in true_doc_ids:
        if id not in pred_doc_ids:
            yp.append(0)
            yt.append(1)

print(f'Hit one: {round(hit_one / total, 4)}')
print(f'Hit all: {round(hit_all / total, 4)}')
print(f'F1:        {round(f1_score(yt, yp), 4)}')
print(f'Precision: {round(precision_score(yt, yp), 4)}')
print(f'Recall:    {round(recall_score(yt, yp), 4)}')