import numpy as np
with open("cord19_0501_titabs_r2.txt") as fin:
    lines = fin.readlines()
np.random.shuffle(lines)

with open("train.txt", "w") as ftrain, open("dev.txt", "w") as fdev:
    for step, line in enumerate(lines):
        if step < 1000:
            fdev.write(line)
        else:ftrain.write(line)