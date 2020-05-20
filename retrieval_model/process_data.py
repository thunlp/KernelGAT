import json
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file')
    parser.add_argument('--retrieval_file')
    parser.add_argument('--output')
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    filter_dict = dict()
    data_dict = dict()
    golden_dict = dict()
    with open(args.gold_file) as f:
        for line in f:
            data = json.loads(line)
            data_dict[data["id"]] = {"id": data["id"], "evidence":[], "claim": data["claim"]}
            if "label" in data:
                data_dict[data["id"]]["label"] = data["label"]
            if not args.test:
                for evidence in data["evidence"]:
                    data_dict[data["id"]]["evidence"].append([evidence[0], evidence[1], evidence[2], 1.0])
                    string = str(data["id"]) + "_" + evidence[0] + "_" + str(evidence[1])
                    golden_dict[string] = 1
    with open(args.retrieval_file) as f:
        for line in f:
            data = json.loads(line)
            for step, evidence in enumerate(data["evidence"]):
                string = str(data["id"]) + "_" + str(evidence[0]) + "_" + str(evidence[1])
                if string not in golden_dict and string not in filter_dict:
                    data_dict[data["id"]]["evidence"].append([evidence[0], evidence[1], evidence[2], evidence[4]])
                    filter_dict[string] = 1
    with open(args.output, "w") as out:
        for data in data_dict.values():
            evidence_tmp = data["evidence"]
            evidence_tmp = sorted(evidence_tmp, key=lambda x:x[3], reverse=True)
            data["evidence"] = evidence_tmp[:5]
            out.write(json.dumps(data) + "\n")


