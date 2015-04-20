

import os
import numpy as np


path = "C:\\Temp\\test_python\\RRP\\"

files = [f for f in os.listdir(path) if "submission" in f]

def main():

    N = len(files)

    sums = np.zeros((100000,))

    with open(path + "sub_avr.txt", "w+") as fout:
        fout.write("Id,Prediction%s" % os.linesep)

        for fn in files:
            fin = open(path + fn, "r")

            fin.readline()

            for i in range(sums.shape[0]):
                line = fin.readline()
                line = line.strip()
                tokens = line.split(',')
                if len(tokens) == 2:
                    sums[i] += float(tokens[1])
                else:
                    print fn, i, tokens

        sums /= N

        for i in range(sums.shape[0]):
            fout.write("%s,%2.16f%s" % (i, sums[i], os.linesep))




if __name__ == '__main__':
    main()
