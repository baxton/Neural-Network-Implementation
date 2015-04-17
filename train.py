

import os
import sys
import numpy as np
import scipy as sp
import ctypes
from array import array
import datetime as dt

#from sklearn.linear_model import LinearRegression


#ANN_DLL = ctypes.cdll.LoadLibrary(r"/home/maxim/kaggle/RRP/scripts/ann/libann.so")
ANN_DLL = ctypes.cdll.LoadLibrary(r"C:\Temp\test_python\RRP\scripts\ann_t\ann2.dll")


#path_data = "/home/maxim/kaggle/RRP/data/"
path_data = "C:\\Temp\\test_python\\RRP\\data\\"

fname_train = "train_data.csv"
fname_test  = "test_data.csv"

REV_MEAN = 4453532.6131386859

ARR_LEN = 44
VEC_LEN = 41


NULL = .0001

Rmin = -9.5
Rmax = 9.5

x_beg = 0
x_end = -1



def minmax(min1, max1, min2, max2, X):

    if min1 == None or max1 == None:
        min1 = X.min(axis=0)
        max1 = X.max(axis=0)

    k = (max2 - min2) / (max1 - min1)
    X -= min1
    X *= k
    X += min2

    return min1, max1, X





def get_k_of_n(k, n):
    numbers = np.array([0] * k, dtype=int)

    for i in range(k):
        numbers[i] = i

    for i in range(k, n):
        r = np.random.randint(0, i)
        if r < k:
            numbers[r] = i

    return numbers


def choice(arr, k):
    n = len(arr)
    indices = get_k_of_n(k, n)


    if isinstance(arr, np.ndarray):
        return arr[indices]

    t = type(arr[0])
    return np.array([arr[i] for i in indices], dtype=t)



def load(fname):
    data = np.loadtxt(fname, delimiter=',')
    return data.astype(np.float64)



def augment_one(X, Y, idx, num=10):
    x = X[idx,:].copy()
    y = Y[idx,:].copy()

    #vals = X.max(axis=0)

    for i in range(num):
        for j in range(5):
            rc = sp.random.randint(0, x.shape[0], 1)
            x[rc] = NULL
        X = np.append(X, x.reshape((1,x.shape[0])), axis=0)
        Y = np.append(Y, y.reshape((1,y.shape[0])), axis=0)
    return X, Y




def train_classifier(train_data):
    data = train_data

    Y = data[:,-1].copy()
    X = data[:,:-1].copy()

    # < 3M, 3M <= && < 5M, 5M <= && < 10M, 10M <=
    YC = np.array([[0]] * Y.shape[0], dtype=np.float64)

    ii = Y > 15000000
    YC[ii,0] = 1
    YC[~ii,0] = 0
    Y = YC

    Xmin, Xmax, X = minmax(None, None, Rmin, Rmax, X)

    sizes = np.array([X.shape[1]] + [51]*2 + [1], dtype=np.int32)
    ann = ANN_DLL.ann_create(sizes.ctypes.data, ctypes.c_int(sizes.shape[0]), ctypes.c_int(0))

    alpha = ctypes.c_double(.08)
    MBS = Y.shape[0]

    ii = range(Y.shape[0])
    sp.random.shuffle(ii)

    for i in range(1000):
        indices = ii

        Xtmp = X[indices,:].astype(np.float64)
        Ytmp = Y[indices,:].astype(np.float64)
        ANN_DLL.ann_fit(ctypes.c_void_p(ann), Xtmp.ctypes.data, Ytmp.ctypes.data, ctypes.c_int(MBS), ctypes.addressof(alpha), ctypes.c_double(20), ctypes.c_int(5))

        ## COST
        prediction = np.array([0], dtype=np.float64)
        if i > 0 and 0 == (i % 20):
            TP = 0.
            FP = 0.
            for i in ii:
                x = X[i,:].astype(np.float64)
                ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))

                y = 1. if data[i,-1] > 15000000 else 0.
                v = 1. if prediction[0] > .5 else 0.

                TP += 1 if v == 1 and y == 1 else 0
                FP += 1 if v == 1 and y == 0 else 0

                ##print prediction, v, "\t", y, "(", v - y , ")", i
            print "TP:", TP, "FP", FP

    return ann



def process3(train_data):


    data = train_data

    Y = data[:,-1].copy()

    # preproc
    Y_LEN = 1

    ii = Y < 15000000.  # 15000000
    Y = Y[ii]
    Y = Y.reshape((Y.shape[0],Y_LEN))

    N = Y.shape[0]
    train_set = range(N)
    sp.random.shuffle(train_set)
    train_set = train_set[:int(N*.9)]
    test_set = [i for i in range(N) if i not in train_set]


    Ymin, Ymax, Y = minmax(None, None, Rmin, Rmax, Y)

    X = data[:,x_beg:x_end].copy()
    X = X[ii,0:]

    Xmin, Xmax, X = minmax(None, None, Rmin, Rmax, X)


##    for i in range(Y.shape[0]):
##        X, Y = augment_one(X, Y, i, num=1)

    #
    #
    #



    sizes = np.array([X.shape[1]] + [51]*2 + [1], dtype=np.int32)
    ann = ANN_DLL.ann_create(sizes.ctypes.data, ctypes.c_int(sizes.shape[0]), ctypes.c_int(1))
    ##lr = LinearRegression()

    if 0 == len(test_set):
        test_set = train_set

    l = 24
    alpha = ctypes.c_double(.08)
    MBS = len(train_set)

    prediction = np.array([0]*1, dtype=np.float64)


    cost = 0.

    indices = train_set

    for i in range(20000):
        #indices = train_set[:MBS]
        #sindices = choice(train_set, MBS)

        #MBS = len(train_set) if 0 == (i % 2) else len(train_set) / 3 * 2

        Ytmp = Y[indices,:].astype(np.float64)
        Xtmp = X[indices,:].astype(np.float64)

        ANN_DLL.ann_fit(ctypes.c_void_p(ann), Xtmp.ctypes.data, Ytmp.ctypes.data, ctypes.c_int(MBS), ctypes.addressof(alpha), ctypes.c_double(l), ctypes.c_int(5))
#        alpha.value = .02
        if i > 400000:
            alpha.value = .00002
        elif i > 250000:
            alpha.value = .00004
        elif i > 200000:
            alpha.value = .00006
        elif i > 140000:
            alpha.value = .00008
        elif i > 100000:
            alpha.value = .0001
        elif i > 80000:
            alpha.value = .0002
        elif i > 60000:
            alpha.value = .0004
        elif i > 50000:
            alpha.value = .0006
        elif i > 40000:
            alpha.value = .0008
        elif i > 30000:
            alpha.value = .001
        elif i > 20000:
            alpha.value = .002
        elif i > 10000:
            alpha.value = .004
        elif i > 1000:
            alpha.value = .008
##        elif i > 40:
##            alpha.value = .01
        elif i > 30:
            alpha.value = .02
        elif i > 20:
            alpha.value = .04
        elif i > 10:
            alpha.value = .08


        ## COST
        if i > 0 and 0 == (i % 2000):
            print "MBS:", MBS, "ITER:", i

            cost = 0.
            for i in test_set:
                x = X[i,:].astype(np.float64)
                ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))
                m1, m2, v = minmax(Rmin, Rmax, Ymin, Ymax, prediction[0])
                m1, m2, y = minmax(Rmin, Rmax, Ymin, Ymax, Y[i,0])
                cost += (v - y) * (v - y)
                ##print prediction, v, "\t", y, "(", v - y , ")", i
            cost /= len(test_set)
            cost = np.sqrt(cost)
            print "COST:", cost, "Rmin/max", Rmin, Rmax

            if cost < 900000. or cost > 1100000.:
                break

            #break


        if alpha.value == 0:
            break;

##        if 4 >= sp.random.randint(0, 10, 1):
##            alpha.value *= sp.random.randint(2, 5, 1)


        if True == os.path.exists("C:\\Temp\\test_python\\RRP\\scripts\\ann_t\\STOP.txt"):
            break

    # COST last one
    cost = 0.
    for i in test_set:
        x = X[i,:].astype(np.float64)
        ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))
        m1, m2, v = minmax(Rmin, Rmax, Ymin, Ymax, prediction[0])
        m1, m2, y = minmax(Rmin, Rmax, Ymin, Ymax, Y[i,0])
        cost += (v - y) * (v - y)
        ##print prediction, v, "\t", y, "(", v - y , ")", i
    cost /= len(test_set)
    cost = np.sqrt(cost)
    print "COST:", cost, "Rmin/max", Rmin, Rmax


    return ann, cost, Xmin, Xmax, Ymin, Ymax
    ##


def regression(ann, test_data, Xmin, Xmax, Ymin, Ymax, fnum, cost, pref):
    # regression
    data = test_data

    X = data[:,x_beg:x_end].copy()

    print "REG: Xmin/max", Xmin, Xmax
    m1, m2, X = minmax(Xmin, Xmax, Rmin, Rmax, X)

    vals = np.zeros((X.shape[0],))

    prediction = np.array([0]*1, dtype=np.float64)

    with open(path_data + "../submission_%s_%f_%d.txt" % (pref, cost, fnum), "w+") as fout:
        fout.write("Id,Prediction%s" % os.linesep)
        for row in range(data.shape[0]):
            x = X[row,:].astype(np.float64)
            ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))
            #v = prediction[0] * (Ymax - Ymin) # + Ymean
            m1, m2, v = minmax(Rmin, Rmax, Ymin, Ymax, prediction[0])

            if 0 == (row % 5000):
                print "ID:", data[row,-1], "val:", v
            fout.write("%d,%2.16f%s" % (data[row,-1], v, os.linesep))

            vals[row] = v

        print "ID:", data[row,-1], "val:", v
        print "STD:", vals.std(), "NEGS:", len(vals[vals < 0.])






def main():
    sp.random.seed()

    pref = sys.argv[1]

    train = load(path_data + fname_train)
    test = load(path_data + fname_test)


    global Rmin, Rmax
    Rmin = -9.
    Rmax = 9.

    Xmin = 0.
    Xmax = 0.

    ann = None

    fnum = 0

    N = 1000000
    cost = 0.
    for i in range(0, N):
        ann, tmp_cost, Xmin, Xmax, Ymin, Ymax = process3(train)
        if tmp_cost < 1000000:
            cost += tmp_cost
            regression(ann, test, Xmin, Xmax, Ymin, Ymax, fnum, tmp_cost, pref)
            fnum += 1
    cost /= N

    print "AVR COST:", cost



   # train_classifier(train)



if __name__ == '__main__':
    main()




#
# Ymean = 4453532.49635; Ymin = -3.30366e+06; Ymax = 1.52434e+07
#

