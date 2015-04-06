

import os
import numpy as np
import scipy as sp
import ctypes
from array import array
import datetime as dt


ANN_DLL = ctypes.cdll.LoadLibrary(r"C:\Temp\test_python\RRP\scripts\ann_t\ann.dll")


path_data = "C:\\Temp\\test_python\\RRP\\data\\"

fname_train = "train_data.csv"
fname_test  = "test_data.csv"

REV_MEAN = 4453532.6131386859

ARR_LEN = 42
VEC_LEN = 41




def load(fname):
    data = np.loadtxt(fname, delimiter=',')
    return data.astype(np.float64)



def process3():
    data = load(path_data + fname_train)

    Y = data[:,-1]

    # preproc
    Y_LEN = 1
    Y = Y.reshape((Y.shape[0],Y_LEN))
##    Y = np.append(Y, Y, axis=1)
##    Y = np.append(Y, Y, axis=1)

    Ymean = Y.mean()
    Y -= Ymean
    Ymin = Y.min()
    Ymax = Y.max()
    Y /= (Ymax - Ymin)

    print "Y mean:", Ymean, "Ymin:", Ymin, "Ymax:", Ymax

    X = data[:,:-1]

    # pre proc
##    dn = dt.date.today()
##    for i in range(X.shape[0]):
##        date = int(X[i,0])
##        y = date / 10000
##        m = (date - y * 10000) / 100
##        d = (date - y * 10000 - m * 100)
##        print y, m, d, int(X[i,0])
##        date = dt.date(y, m, d)
##        delta = (dn - date)
##        X[i,0] = delta.days

    Xmean = X.mean(axis=0)
    X -= Xmean
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    X /= (Xmax - Xmin)

    # aug

    for i in range(1):
        XA = X.copy()
        for i in range(XA.shape[0]):
            for j in range(4):
                rc = sp.random.randint(0, XA.shape[1])
                XA[i,rc] = Xmean[rc]
        X = np.append(X, XA, axis=0)
        Y = np.append(Y, Y, axis=0)


    print "X mean:", Xmean, "Xmin:", Xmin, "Xmax:", Xmax

    N = Y.shape[0]
    alpha = ctypes.c_double(.4)


    ann = ANN_DLL.ann_create()

    train_set = range(N)
    sp.random.shuffle(train_set)
    train_set = train_set[:int(N*.99)]
    test_set = [i for i in range(N) if i not in train_set]
##    train_set = train_set
##    test_set = train_set[:int(N*.3)]

#    train_set = [43, 104, 54, 117, 118, 53, 119, 90, 47, 1, 49, 105, 55, 115, 130, 58, 11, 76, 82, 101, 37, 89, 70, 68, 72, 122, 66, 40, 19, 107, 106, 57, 6, 69, 45, 5, 71, 31, 61, 100, 13, 36, 110, 123, 128, 67, 95, 83, 102, 74, 103, 96, 50, 126, 93, 62, 65, 28, 46, 133, 84, 9, 3, 4, 42, 97, 94, 24, 136, 91, 79, 52, 109, 18, 113, 129, 108, 73, 33, 34, 60, 23, 120, 21, 86, 135, 59, 44, 121, 134, 112, 27, 63, 35, 14, 132, 88, 51, 38, 29, 12, 131, 30, 99, 124, 39, 32, 15, 81]
#    test_set = [0, 2, 7, 8, 10, 16, 17, 20, 22, 25, 26, 41, 48, 56, 64, 75, 77, 78, 80, 85, 87, 92, 98, 111, 114, 116, 125, 127]

    MBS = len(train_set) / 1

    prediction = np.array([0]*Y_LEN, dtype=np.float64)

    prev_test_cost = 99999999999
    prev_cost_less_cnt = 0

    for i in range(1000):
        #indices = train_set[:MBS]
        indices = sp.random.choice(train_set, MBS, replace=False)

        Ytmp = Y[indices,:].astype(np.float64)
        Xtmp = X[indices,:].astype(np.float64)

        ANN_DLL.ann_fit(ctypes.c_void_p(ann), Xtmp.ctypes.data, Ytmp.ctypes.data, ctypes.c_int(MBS), ctypes.addressof(alpha), ctypes.c_double(.1), ctypes.c_int(200))
        if 4 >= sp.random.randint(0, 10, 1):
            alpha.value *= sp.random.randint(2, 4, 1)

##        if MBS < len(train_set):
##            MBS += 1
        print "MBS:", MBS, "ITER:", i

        ## COST
        cost = 0.
        for i in test_set:
            x = X[i,:].astype(np.float64)
            ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))
            v = prediction * (Ymax - Ymin) + Ymean
            v = np.mean(v)
            cost += (v - Y[i,0]) * (v - Y[i,0])
        cost /= len(test_set)
        cost = np.sqrt(cost)
        print "COST:", cost


        if round(cost, 9) == round(prev_test_cost, 9):
            prev_cost_less_cnt += 1
            if prev_cost_less_cnt > 10:
                break
        else:
            prev_cost_less_cnt = 0
        prev_test_cost = cost
        ##


    # regression
    data = load(path_data + fname_test)

    X = data[:,:-1]

    # pre proc
##    for i in range(X.shape[0]):
##        date = int(X[i,0])
##        y = date / 10000
##        m = (date - y * 10000) / 100
##        d = (date - y * 10000 - m * 100)
##        date = dt.date(y, m, d)
##        delta = (dn - date)
##        X[i,0] = delta.days

    X -= Xmean
    X /= (Xmax - Xmin)


    vals = np.zeros((X.shape[0],))

    with open(path_data + "..\\submission_t.txt", "w+") as fout:
        fout.write("Id,Prediction%s" % os.linesep)
        for row in range(data.shape[0]):
            x = X[row,:].astype(np.float64)
            ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))
            v = prediction * (Ymax - Ymin) + Ymean
            v = np.mean(v)

            if 0 == (row % 5000):
                print "ID:", data[row,-1], "val:", v
            fout.write("%d,%2.16f%s" % (data[row,-1], v, os.linesep))

            vals[row] = v

        print "ID:", data[row,-1], "val:", v
        print "STD:", vals.std(), "NEGS:", len(vals[vals < 0.])






def main():
    sp.random.seed()
    process3()




if __name__ == '__main__':
    main()




#
# Ymean = 4453532.49635; Ymin = -3.30366e+06; Ymax = 1.52434e+07
#
