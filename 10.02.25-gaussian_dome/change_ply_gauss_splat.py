import numpy as np

with open("./input1.txt", "rt") as fin:
    with open("./input1replace.txt", "wt") as fout:
        for line in fin:
            if "1.7724538 1.7724538 1.7724538" in line:
                fout.write(line.replace('1.7724538 1.7724538 1.7724538', '1.7724538 -1.7724538 -1.7724538'))
            else:
                fout.write(line)



