


import os

path = "/home/maxim/kaggle/OTTO/"

files = [
'submission_128-128H50drop_0.544374_0.txt',
'submission_normal_0.560924_0.txt',
'submission_35HAVR_0.543265_0.txt',
]


header = ""
N = 0
sums = {}

for f in files:
    with open(path + f, "r") as fin:
        if 0 == len(header):
            header = fin.readline().strip()
        else:
            fin.readline().strip()

        for line in fin:
            line = line.strip()
            tokens = line.split(',')
         
            id = int(tokens[0])   
            vals = [float(v) for v in tokens[1:]]

            if id in sums:
                existing = sums[id]
                for i in range(len(existing)):
                    existing[i] += vals[i]
            else:
                sums[id] = vals

        N += 1

fname = path + "sub_avr.txt"
with open(fname, "w+") as fout:
    fout.write("%s%s" % (header, os.linesep))

    keys = sums.keys()
    keys.sort()

    for id in keys:
        fout.write("%d" % id)
        vals = sums[id]
        for v in vals:
            v /= N
            fout.write(",%f" % v)
        fout.write(os.linesep)





