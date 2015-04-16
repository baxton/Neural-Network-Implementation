

import os
import numpy as np
import scipy as sp
import ctypes
from array import array
import datetime as dt

#from sklearn.linear_model import LinearRegression


#ANN_DLL = ctypes.cdll.LoadLibrary(r"/home/maxim/kaggle/RRP/scripts/ann/libann.so")
ANN_DLL = ctypes.cdll.LoadLibrary(r"C:\Temp\test_python\RRP\scripts\ann_t\ann.dll")


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
    for i in range(num):
        for j in range(5):
            rc = sp.random.randint(0, x.shape[0], 1)
            x[rc] = NULL
        X = np.append(X, x.reshape((1,x.shape[0])), axis=0)
        Y = np.append(Y, y.reshape((1,y.shape[0])), axis=0)
    return X, Y




def process3(fnum, train_data, test_data):
    data = train_data

    Y = data[:,-1].copy()

    # preproc
    Y_LEN = 1

    #ii = Y < 2000000.
    #Y = Y[ii]
    Y = Y.reshape((Y.shape[0],Y_LEN))


    Ymin, Ymax, Y = minmax(None, None, Rmin, Rmax, Y)

    X = data[:,:-1].copy()
    #X = X[ii,0:]

    X[ X == 0. ] = NULL

    Xmin, Xmax, X = minmax(None, None, Rmin, Rmax, X)

    for i in range(Y.shape[0]):
        X, Y = augment_one(X, Y, i, num=3)

    #
    #
    #

    N = Y.shape[0]



    sizes = np.array([X.shape[1]] + [51]*33 + [1], dtype=np.int32)
    ann = ANN_DLL.ann_create(sizes.ctypes.data, ctypes.c_int(sizes.shape[0]), ctypes.c_int(1))
    ##lr = LinearRegression()

    train_set = range(N)
    sp.random.shuffle(train_set)
    train_set = train_set[:int(N*.95)]
    test_set = [i for i in range(N) if i not in train_set]

    if 0 == len(test_set):
        test_set = train_set

    alpha = ctypes.c_double(.08)
    MBS = len(train_set)

    prediction = np.array([0]*1, dtype=np.float64)


    for i in range(2000000):
        #indices = train_set[:MBS]
        indices = choice(train_set, MBS)

        #MBS = len(train_set) if 0 == (i % 2) else len(train_set) / 3 * 2

        Ytmp = Y[indices,:].astype(np.float64)
        Xtmp = X[indices,:].astype(np.float64)

        ANN_DLL.ann_fit(ctypes.c_void_p(ann), Xtmp.ctypes.data, Ytmp.ctypes.data, ctypes.c_int(MBS), ctypes.addressof(alpha), ctypes.c_double(20), ctypes.c_int(5))
#        alpha.value = .02
        if i > 400000:
            alpha.value = .00002
        elif i > 250000:
            alpha.value = .00004
        elif i > 200000:
            alpha.value = .00008
        elif i > 140000:
            alpha.value = .0001
        elif i > 100000:
            alpha.value = .0002
        elif i > 80000:
            alpha.value = .0004
        elif i > 60000:
            alpha.value = .0008
        elif i > 50000:
            alpha.value = .001
        elif i > 40000:
            alpha.value = .002
        elif i > 30000:
            alpha.value = .003
        elif i > 20000:
            alpha.value = .004
        elif i > 10000:
            alpha.value = .008
        elif i > 1000:
            alpha.value = .009
        elif i > 40:
            alpha.value = .01
        elif i > 30:
            alpha.value = .02
        elif i > 20:
            alpha.value = .04
        elif i > 10:
            alpha.value = .08














        #lr.fit(Xtmp, Ytmp)

        print "MBS:", MBS, "ITER:", i


        ## COST
        if i > 0 and 0 == (i % 100):
            cost = 0.
            for i in test_set:
                x = X[i,:].astype(np.float64)
                ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))
                m1, m2, v = minmax(Rmin, Rmax, Ymin, Ymax, prediction[0])
                m1, m2, y = minmax(Rmin, Rmax, Ymin, Ymax, Y[i,0])
                #v = prediction[0] * Ysum
                #y = Y[i,0] * Ysum
                cost += (v - y) * (v - y)
                ##print prediction, v, "\t", y, "(", v - y , ")", i
            cost /= len(test_set)
            cost = np.sqrt(cost)
            print "COST:", cost, "Rmin/max", Rmin, Rmax

##            if cost > 2500000:
##                break


        if alpha.value == 0:
            break;

##        if 4 >= sp.random.randint(0, 10, 1):
##            alpha.value *= sp.random.randint(2, 5, 1)


        if True == os.path.exists("C:\\Temp\\test_python\\RRP\\scripts\\ann_t\\STOP.txt"):
            break

        ##


    # regression
    data = test_data

    X = data[:,:-1].copy()

    m1, m2, X = minmax(Xmin, Xmax, Rmin, Rmax, X)

    vals = np.zeros((X.shape[0],))

    with open(path_data + "../submission_AUG_a.02_l20_i5_INF_51x33_-9_9" + str(fnum) + ".txt", "w+") as fout:
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

    train = load(path_data + fname_train)
    test = load(path_data + fname_test)


    global Rmin, Rmax
    Rmin = -9.
    Rmax = 9.

    for i in range(0, 1):
        process3(i, train, test)
#        Rmin *= 2.
#        Rmax *= 2.




if __name__ == '__main__':
    main()




#
# Ymean = 4453532.49635; Ymin = -3.30366e+06; Ymax = 1.52434e+07
#

