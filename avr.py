

import os


path = "C:\\Temp\\test_python\\RRP\\"

files = [
"submission1.txt",
"submission2.txt",
"submission5.txt",

"submission6.txt",
"submission7.txt",
"submission8.txt",
"submission9.txt",
"submission10.txt",
"submission11.txt",
"submission12.txt",
"submission13.txt",
"submission14.txt",
"submission15.txt",
"submission16.txt",
"submission17.txt",
"submission18.txt",
"submission19.txt",
"submission20.txt",
"submission21.txt",
"submission22.txt",
"submission23.txt",
"submission24.txt",
"submission25.txt",
"submission26.txt",
"submission27.txt",
"submission28.txt",
"submission29.txt",
"submission30.txt",
"submission31.txt",
"submission32.txt",
"submission33.txt",
"submission34.txt",
"submission35.txt",
"submission36.txt",
"submission37.txt",
"submission38.txt",
"submission39.txt",
"submission40.txt",
"submission41.txt",
"submission42.txt",
"submission43.txt",
"submission44.txt",
"submission45.txt",
"submission46.txt",
"submission47.txt",
"submission48.txt",
"submission49.txt",
"submission50.txt",
"submission51.txt",
"submission52.txt",
"submission53.txt",
"submission54.txt",
"submission55.txt",
"submission56.txt",
"submission57.txt",
"submission58.txt",
"submission59.txt",
"submission60.txt",
"submission61.txt",
"submission62.txt",
"submission63.txt",
"submission64.txt",
"submission65.txt",
##"submission66.txt",
##"submission67.txt",
##"submission68.txt",
##"submission69.txt",
##"submission70.txt",


"submission_t1.txt",
"submission_t2.txt",
"submission_t3.txt",
"submission_t4.txt",
"submission_t5.txt",
"submission_t6.txt",
"submission_t7.txt",
"submission_t8.txt",
"submission_t9.txt",
"submission_t10.txt",
"submission_t11.txt",
"submission_t12.txt",
"submission_t13.txt",
"submission_t14.txt",
"submission_t15.txt",
"submission_t16.txt",
"submission_t17.txt",
"submission_t18.txt",
"submission_t19.txt",
"submission_t20.txt",
"submission_t21.txt",
"submission_t22.txt",
"submission_t23.txt",
"submission_t24.txt",
"submission_t25.txt",
"submission_t26.txt",
"submission_t27.txt",
"submission_t28.txt",
"submission_t29.txt",
"submission_t30.txt",


]

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
