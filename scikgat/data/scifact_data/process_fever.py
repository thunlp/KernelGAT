import json
with open("fever.jsonl", "w") as fout:
    for path in ["./bert_train.json", "./bert_dev.json"]:
        with open(path) as fin:
            for line in fin:
                data = json.loads(line)
                new_evidence = list()
                for evidence in data["evidence"]:
                    evidence[2] = evidence[2].replace(" LRB ", " ( ").replace(" RRB ", " ) ")
                    evidence = [0] + evidence

                    new_evidence.append(evidence)
                data["evidence"] = new_evidence
                label = data["label"]
                if label == "SUPPORTS":
                    label = "SUPPORT"
                elif label == "REFUTES":
                    label = "CONTRADICT"
                else:
                    label = "NOT_ENOUGH_INFO"
                data["label"] = label
                fout.write(json.dumps(data) + "\n")

