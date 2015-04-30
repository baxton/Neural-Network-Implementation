
import os
import scipy as sp


path = "C:\\Temp\\test_python\\RRP\\scripts\\ann_t\\"

files = [f for f in os.listdir(path) if "test" in f]

pairs = {}


def process_pair(fn1, fn2):

    fins = [
        open(fn1, "r"),
        open(fn2, "r"),
    ]

    total = 0
    cnts = [0] * 2
    vals = [0] * 2
    costs = [0] * 2

    stop = False

    try:
        while True:
            i = 0
            for fin in fins:
                line = fin.readline().strip()
                if line.startswith("COST"):
                    tokens = line.split(" ")
                    costs[i] = float(tokens[1])
                else:
                    tokens = line.split(" ")
                    vals[i] = float(tokens[1])
                i += 1

            total += 1

            idx = sp.argmin(vals)
            cnts[idx] += 1

    except:
        pass

    best_idx = sp.argmax(cnts)

    print fn1, "vs", fn2
    print "Total", total, "1st", cnts[0], "2nd", cnts[1]

    return {fn1 + " vs " + fn2 : (cnts[0], cnts[1])}


def  cmp(a, b):
    key = a + " vs " + b
    if key not in pairs:
        key = b + " vs " + a

    if pairs[key][0] > pairs[key][1]:
        return 1
    elif pairs[key][0] < pairs[key][1]:
        return -1
    return 0



def main():
    N = len(files)

    fins = [open(f, "r") for f in files]

    global pairs


    for i in range(N):
        for j in range(i+1, N):
            e = process_pair(files[i], files[j])
            pairs.update(e)



    files.sort(cmp=cmp)
    print files


if __name__ == '__main__':
    main()
