import numpy as np
scifact_data =list()
fever_data = list()
with open("scifact_train.jsonl") as fin:
    scifact_data = fin.readlines()
with open("fever.jsonl") as fin:
    fever_data = fin.readlines()

scifact_data = scifact_data * 50
fever_data = fever_data + scifact_data
np.random.shuffle(fever_data)
with open("../train.jsonl", "w") as fout:
    for data in fever_data:
        fout.write(data)



