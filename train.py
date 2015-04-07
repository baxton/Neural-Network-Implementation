

import os
import numpy as np
import scipy as sp
import ctypes
from array import array
import datetime as dt


ANN_DLL = ctypes.cdll.LoadLibrary(r"C:\Temp\test_python\RRP\scripts\ann_t\ann2.dll")


path_data = "C:\\Temp\\test_python\\RRP\\data\\"

fname_train = "train_data.csv"
fname_test  = "test_data.csv"

REV_MEAN = 4453532.6131386859

ARR_LEN = 42
VEC_LEN = 41




def load(fname):
    data = np.loadtxt(fname, delimiter=',')
    return data.astype(np.float64)


def augment_one(X, Y, idx, num=10):
    x = X[idx,:].copy()
    y = Y[idx,:].copy()
    for i in range(num):
        for j in range(10):
            rc = sp.random.randint(41, x.shape[0], 1)
            x[rc] = 0.
        X = np.append(X, x.reshape((1,x.shape[0])), axis=0)
        Y = np.append(Y, y.reshape((1,1)), axis=0)
    return X, Y


def process3():
    data = load(path_data + fname_train)

    Y = data[:,-1].copy()

    # preproc
    Y_LEN = 1
    Y = Y.reshape((Y.shape[0],Y_LEN))
##    Y = np.append(Y, Y, axis=1)
##    Y = np.append(Y, Y, axis=1)

    Ymean = Y.mean()
    Ymin = Y.min()
    Ymax = Y.max()
    Y -= Ymean
    Y /= (Ymax - Ymin)

    print "Y mean:", Ymean, "Ymin:", Ymin, "Ymax:", Ymax

    X = data[:,:-1].copy()

    Xmean = X.mean(axis=0)
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    X[:,0:3] -= Xmean[0:3]
    X[:,41:] -= Xmean[41:]
    X[:,0:3] /= (Xmax - Xmin)[0:3]
    X[:,41:] /= (Xmax - Xmin)[41:]

    #
    # aug
    #

    X, Y = augment_one(X, Y, 0)
    X, Y = augment_one(X, Y, 1)
    X, Y = augment_one(X, Y, 5)
    X, Y = augment_one(X, Y, 6)
    X, Y = augment_one(X, Y, 8)
    X, Y = augment_one(X, Y, 9)
    X, Y = augment_one(X, Y, 11)
    X, Y = augment_one(X, Y, 13)
    X, Y = augment_one(X, Y, 16)
    X, Y = augment_one(X, Y, 17)
    X, Y = augment_one(X, Y, 18)
    X, Y = augment_one(X, Y, 20)
    X, Y = augment_one(X, Y, 24)
    X, Y = augment_one(X, Y, 27)
    X, Y = augment_one(X, Y, 28)
    X, Y = augment_one(X, Y, 38)
    X, Y = augment_one(X, Y, 40)
    X, Y = augment_one(X, Y, 41)
    X, Y = augment_one(X, Y, 42)
    X, Y = augment_one(X, Y, 47)
    X, Y = augment_one(X, Y, 48)
    X, Y = augment_one(X, Y, 49)
    X, Y = augment_one(X, Y, 53)
    X, Y = augment_one(X, Y, 54)
    X, Y = augment_one(X, Y, 55)
    X, Y = augment_one(X, Y, 62)
    X, Y = augment_one(X, Y, 74)
    X, Y = augment_one(X, Y, 75)
    X, Y = augment_one(X, Y, 76)
    X, Y = augment_one(X, Y, 79)
    X, Y = augment_one(X, Y, 83)
    X, Y = augment_one(X, Y, 85)
    X, Y = augment_one(X, Y, 87)
    X, Y = augment_one(X, Y, 92)
    X, Y = augment_one(X, Y, 96)
    X, Y = augment_one(X, Y, 97)
    X, Y = augment_one(X, Y, 99)
    X, Y = augment_one(X, Y, 100)
    X, Y = augment_one(X, Y, 101)
    X, Y = augment_one(X, Y, 106)
    X, Y = augment_one(X, Y, 115)
    X, Y = augment_one(X, Y, 116)
    X, Y = augment_one(X, Y, 124, 20)
    X, Y = augment_one(X, Y, 125)
    X, Y = augment_one(X, Y, 127)
    X, Y = augment_one(X, Y, 132)
    X, Y = augment_one(X, Y, 133)
    X, Y = augment_one(X, Y, 135)
    X, Y = augment_one(X, Y, 136)

    for i in range(data.shape[0]):
        X, Y = augment_one(X, Y, i, 25)

    for i in range(1):
        XA = X.copy()
        for i in range(XA.shape[0]):
            for j in range(4):
                idx = sp.random.randint(41, X.shape[1], 1)
                XA[i,idx] = Xmean[idx]
        X = np.append(X, XA, axis=0)
        Y = np.append(Y, Y, axis=0)

    #
    #
    #

    print "X mean:", Xmean, "Xmin:", Xmin, "Xmax:", Xmax

    N = Y.shape[0]
    alpha = ctypes.c_double(.4)


    ann = ANN_DLL.ann_create()

    train_set = range(N)
    sp.random.shuffle(train_set)
    train_set = train_set[:int(N*.80)]
    test_set = [i for i in range(N) if i not in train_set]
##    train_set = train_set
##    test_set = train_set[:int(N*.3)]

#    train_set = [43, 104, 54, 117, 118, 53, 119, 90, 47, 1, 49, 105, 55, 115, 130, 58, 11, 76, 82, 101, 37, 89, 70, 68, 72, 122, 66, 40, 19, 107, 106, 57, 6, 69, 45, 5, 71, 31, 61, 100, 13, 36, 110, 123, 128, 67, 95, 83, 102, 74, 103, 96, 50, 126, 93, 62, 65, 28, 46, 133, 84, 9, 3, 4, 42, 97, 94, 24, 136, 91, 79, 52, 109, 18, 113, 129, 108, 73, 33, 34, 60, 23, 120, 21, 86, 135, 59, 44, 121, 134, 112, 27, 63, 35, 14, 132, 88, 51, 38, 29, 12, 131, 30, 99, 124, 39, 32, 15, 81]
#    test_set = [0, 2, 7, 8, 10, 16, 17, 20, 22, 25, 26, 41, 48, 56, 64, 75, 77, 78, 80, 85, 87, 92, 98, 111, 114, 116, 125, 127]

    MBS = len(train_set)

    prediction = np.array([0]*Y_LEN, dtype=np.float64)

    prev_test_cost = 99999999999
    prev_cost_less_cnt = 0

    for i in range(500):
        #indices = train_set[:MBS]
        indices = sp.random.choice(train_set, MBS, replace=False)
        sp.random.shuffle(indices)

        Ytmp = Y[indices,:].astype(np.float64)
        Xtmp = X[indices,:].astype(np.float64)

        ANN_DLL.ann_fit(ctypes.c_void_p(ann), Xtmp.ctypes.data, Ytmp.ctypes.data, ctypes.c_int(MBS), ctypes.addressof(alpha), ctypes.c_double(1), ctypes.c_int(250))
        print "MBS:", MBS, "ITER:", i

        ## COST
        if 0 == (i % 10):
            cost = 0.
            for i in test_set:
                x = X[i,:].astype(np.float64)
                ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))
                v = prediction[0] * (Ymax - Ymin) + Ymean
                y = Y[i,0] * (Ymax - Ymin) + Ymean
                cost += (v - y) * (v - y)
            cost /= len(test_set)
            cost = np.sqrt(cost)
            print "COST:", cost


##        if round(cost, 9) == round(prev_test_cost, 9):
##            prev_cost_less_cnt += 1
##            if prev_cost_less_cnt > 10:
##                break
##        else:
##            prev_cost_less_cnt = 0

##        if prev_test_cost < cost:
##            alpha.value /= 2.;
##        else:
        if 4 >= sp.random.randint(0, 10, 1):
            alpha.value *= sp.random.randint(2, 8, 1)

        prev_test_cost = cost
        ##


    # regression
    data = load(path_data + fname_test)

    X = data[:,:-1].copy()

    # pre proc
##    for i in range(X.shape[0]):
##        date = int(X[i,0])
##        y = date / 10000
##        m = (date - y * 10000) / 100
##        d = (date - y * 10000 - m * 100)
##        date = dt.date(y, m, d)
##        delta = (dn - date)
##        X[i,0] = delta.days

    X[:,0:3] -= Xmean[0:3]
    X[:,41:] -= Xmean[41:]
    X[:,0:3] /= (Xmax - Xmin)[0:3]
    X[:,41:] /= (Xmax - Xmin)[41:]


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
