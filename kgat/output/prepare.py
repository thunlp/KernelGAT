import argparse
import json
import sys
from fever.scorer import fever_score

parser = argparse.ArgumentParser()
parser.add_argument("--predicted",type=str)
parser.add_argument("--order",type=str)
parser.add_argument("--original",type=str)
parser.add_argument("--out_file",default="predictions.jsonl", type=str)

args = parser.parse_args()

predicted_labels =dict()
predictions = []


with open(args.predicted,"r") as predictions_file:
    for line in predictions_file:
        data = json.loads(line)
        predicted_labels[data["id"]] = {"id":data["id"], "predicted_label":data["predicted_label"]}


with open(args.original,"r") as predictions_file:
    for line in predictions_file:
        evidences = list()
        for evidence in json.loads(line)["evidence"]:
            evidences.append(evidence[:2])
        predicted_labels[json.loads(line)["id"]]["predicted_evidence"] = evidences[:5]

with open(args.order, "r") as order_file:
    for line in order_file:
        id = json.loads(line)["id"]
        predictions.append(predicted_labels[id])


with open(args.out_file,"w") as f:
    for line in predictions:
        f.write(json.dumps(line)+"\n")
