import os
import json
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile')
    parser.add_argument('--outfile')
    args = parser.parse_args()
    pairs = list()
    with open(args.infile) as f:
            for line in f:
                data = json.loads(line)
                claim = data["claim"]
                for evidence in data["evidence"]:
                    if evidence[3] == 1:
                        for evidence_ in data["evidence"]:
                            if evidence_[3] == 0:
                                sent1 = " ".join(evidence[2].strip().split())
                                sent2 = " ".join(evidence_[2].strip().split())
                                if sent1 != "" and sent2 != "":
                                    pairs.append([claim, evidence[0], sent1, evidence_[0], sent2])
    with open(args.outfile, "w") as out:
        np.random.shuffle(pairs)
        for pair in pairs:
            out.write("\t".join(pair) + "\n")