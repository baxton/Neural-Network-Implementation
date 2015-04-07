

import os


path = "C:\\Temp\\test_python\\RRP\\"

files = [f for f in os.listdir(path) if "submission" in f]

def main():

    N = len(files)

    fd = [open(path + fn, "r") for fn in files ]

    for f in fd:
        f.readline()

    with open(path + "sub_avr.txt", "w+") as fout:
        fout.write("Id,Prediction%s" % os.linesep)

        try:
            while True:
                id = ''
                v = 0.
                for f in fd:
                    line = f.readline()
                    line = line.strip()
                    tokens = line.split(',')
                    id = tokens[0]
                    v += float(tokens[1])
                fout.write("%s,%2.16f%s" % (id, v/N, os.linesep))
        except:
            pass




if __name__ == '__main__':
    main()
