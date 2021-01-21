import json

with open("label_prediction_test.jsonl") as fin1, open("rationale_selection_test.jsonl") as fin2:
	for line in list(zip(fin1, fin2)):
		data1 = json.loads(line[0])
		data2 = json.loads(line[1])
		labels = data1["labels"]
		evidences = data2["evidence"]
		evidence1 = []
		evidence2 = []
		for key, _ in evidences.items():
			evidence2.append(key)
		for key, _ in labels.items():
			evidence1.append(key)
		assert len(evidence1) == len(evidence2) == 3
		for step in range(len(evidence1)):
			assert evidence1[step] == evidence2[step]