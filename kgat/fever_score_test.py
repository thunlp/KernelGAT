import argparse
import json
import sys
from fever.scorer import fever_score
from prettytable import PrettyTable

parser = argparse.ArgumentParser()
parser.add_argument("--predicted_labels",type=str)

parser.add_argument("--predicted_evidence",type=str)
parser.add_argument("--actual",type=str)

args = parser.parse_args()

predicted_labels =[]
predicted_evidence = []
actual = []

ids = dict()
with open(args.predicted_labels,"r") as predictions_file:
    for line in predictions_file:
        ids[json.loads(line)["id"]] = len(predicted_labels)
        predicted_labels.append(json.loads(line)["predicted_label"])
        predicted_evidence.append(0)
        actual.append(0)



with open(args.predicted_evidence,"r") as predictions_file:
    for line in predictions_file:
        evidences = list()
        if "predicted_evidence" in json.loads(line):
            for evidence in json.loads(line)["predicted_evidence"]:
                evidences.append(evidence[:2])
            predicted_evidence[ids[json.loads(line)["id"]]] = evidences
        if "evidence" in json.loads(line):
            for evidence in json.loads(line)["evidence"]:
                evidences.append(evidence[:2])
            predicted_evidence[ids[json.loads(line)["id"]]] = evidences

with open(args.actual, "r") as actual_file:
    for line in actual_file:
        actual[ids[json.loads(line)["id"]]] = json.loads(line)

predictions = []
for ev,label in zip(predicted_evidence,predicted_labels):
    predictions.append({"predicted_evidence":ev,"predicted_label":label})

score,acc,precision,recall,f1 = fever_score(predictions,actual)

tab = PrettyTable()
tab.field_names = ["FEVER Score", "Label Accuracy", "Evidence Precision", "Evidence Recall", "Evidence F1"]
tab.add_row((round(score,4),round(acc,4),round(precision,4),round(recall,4),round(f1,4)))

print(tab)