

import os
import sys
import numpy as np
import scipy as sp
import ctypes
from array import array
import datetime as dt

from sklearn.ensemble import RandomForestClassifier


#ANN_DLL = ctypes.cdll.LoadLibrary(r"/home/maxim/kaggle/RRP/scripts/ann/libann.so")
ANN_DLL = ctypes.cdll.LoadLibrary(r"C:\Temp\test_python\RRP\scripts\ann_t\redann.dll")


#path_data = "/home/maxim/kaggle/RRP/data/"
path_data = "C:\\Temp\\test_python\\RRP\\data\\"

fname_train = "train_data.csv"
fname_test  = "test_data.csv"

REV_MEAN = 4453532.6131386859

ARR_LEN = 44
VEC_LEN = 41


outlier_detector_trained = False
rf_outliers = None
FX = None


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




def train_rf_for_outliers(train_data):

    mean = train_data[:,-1].mean()
    std = train_data[:,-1].std()

    rf = RandomForestClassifier(n_estimators=1000, bootstrap=False)

    Y = np.zeros((train_data.shape[0],1))
    for i in range(train_data.shape[0]):
        v = train_data[i, -1]
        if np.sqrt((v - mean) ** 2) > std:
            Y[i] = 1.

    rf.fit(train_data[:,:-1], Y)

    return rf, Y



def train_bias_remover(ann, X, Y, train_set):
    print "Train bias remover"
    prediction = np.array([0],dtype=np.float64)
    Ytmp = Y.copy()
    for i in train_set:
        x = X[i,:]
        ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))
        d = prediction[0] - Y[i,0]
        Ytmp[i] = d

    Ytmp = Ytmp[train_set].astype(np.float64)
    Xtmp = X[train_set].astype(np.float64)
    Ytmp = np.append(Ytmp, Ytmp, axis=0)
    Xtmp = np.append(Xtmp, Xtmp, axis=0)

    MBS  = Ytmp.shape[0]
    alpha = ctypes.c_double(.008)
    cost_one = ctypes.c_double(0.)

    sizes = np.array([X.shape[1]] + [15]*2 + [1], dtype=np.int32)
    ann_bias = ANN_DLL.ann_create(sizes.ctypes.data, ctypes.c_int(sizes.shape[0]), ctypes.c_int(3))

    for i in range(200):
        ANN_DLL.ann_fit(ctypes.c_void_p(ann_bias), Xtmp.ctypes.data, Ytmp.ctypes.data, ctypes.c_int(X.shape[1]), ctypes.c_int(MBS), ctypes.addressof(alpha), ctypes.c_double(0.), ctypes.addressof(cost_one))
        print "BR:", cost_one.value

    return ann_bias


def process3(train_data):

    data = train_data

    Y = data[:,-1].copy()

    # preproc
    Y_LEN = 1

    #ii = Y < 15000000.  # 15000000
    ii = range(Y.shape[0])
    Y = Y[ii]
    Y = Y.reshape((Y.shape[0],Y_LEN))

    N = Y.shape[0]
    train_set = range(N)
    sp.random.shuffle(train_set)
    train_set = train_set[:int(N*.85)]
    test_set = [i for i in range(N) if i not in train_set]

##    test_set =  [23, 2, 27, 99, 80, 97, 114, 131, 124]
##    train_set =  [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130, 132, 133, 134]


    global outlier_detector_trained, rf_outliers, FX
##    if not outlier_detector_trained:
##        rf_outliers, FX = train_rf_for_outliers(train_data[ii,:][train_set,:])
##        outlier_detector_trained = True


    X = data[:,x_beg:x_end].copy()
##    X = np.append(X, [[0]]*X.shape[0], axis=1)
    X = X[ii,0:]

##    for i in range(X.shape[0]):
##        p = rf_outliers.predict(X[i,:-1])
##        X[i,-1] = p

    #X = np.append(X, np.sin(X), axis=1)
    X = np.append(X, np.log(1. + np.sin(X)), axis=1)

#    for i in choice(range(Y.shape[0]), Y.shape[0] / 3 * 2):
#    for i in range(Y.shape[0]):
#        X, Y = augment_one(X, Y, i, num=1)

    Ymin, Ymax, Y = minmax(None, None, Rmin, Rmax, Y)
    Xmin, Xmax, X = minmax(None, None, Rmin, Rmax, X)


    #
    #
    #



    sizes = np.array([X.shape[1]] + [55]*1 + [15] + [1], dtype=np.int32)
    ann = ANN_DLL.ann_create(sizes.ctypes.data, ctypes.c_int(sizes.shape[0]), ctypes.c_int(5))
    ##lr = LinearRegression()

    if 0 == len(test_set):
        test_set = train_set

    cost_one = ctypes.c_double(0.)
    prev_cost_one = 999999999.

    l = 0
    alpha = ctypes.c_double(.08)
    MBS = len(train_set)

    prediction = np.array([0]*1, dtype=np.float64)


    cost = 0.
    best_cost = 999999999.
    prev_cost = 999999999.
    cost_cnt = 0

    weights_saved = False

    indices = train_set
    np.random.shuffle(indices)

    for i in range(5000):
        #indices = train_set[:MBS]
        #indices = choice(train_set, MBS)
#        indices = np.append(indices, choice(train_set, MBS / 3 * 2))

        #MBS = len(train_set) if 0 == (i % 2) else len(train_set) / 3 * 2

        Ytmp = Y[indices,:].astype(np.float64)
        Xtmp = X[indices,:].astype(np.float64)

        ANN_DLL.ann_fit(ctypes.c_void_p(ann), Xtmp.ctypes.data, Ytmp.ctypes.data, ctypes.c_int(X.shape[1]), ctypes.c_int(MBS), ctypes.addressof(alpha), ctypes.c_double(l), ctypes.addressof(cost_one))

        print cost_one.value, "[", alpha.value, "]"
        #if prev_cost_one < cost_one.value:
        #    alpha.value /= 2.
        prev_cost_one = cost_one.value

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
            alpha.value = .0015
        elif i > 20000:
            alpha.value = .001
        elif i > 10000:
            alpha.value = .002
        elif i > 1000:
            alpha.value = .04
        elif i > 40:
            alpha.value = .08
        elif i > 30:
            alpha.value = .08
        elif i > 20:
            alpha.value = .08
        elif i > 10:
            alpha.value = .08



        if np.isnan(cost_one):
            break


        ## COST
        if i > 0 and 0 == (i % 200):
            print "MBS:", MBS, "ITER:", i

            cost = 0.
            for i in test_set:
                x = X[i,:].astype(np.float64)
                ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))
                m1, m2, v = minmax(Rmin, Rmax, Ymin, Ymax, prediction[0])
                m1, m2, y = minmax(Rmin, Rmax, Ymin, Ymax, Y[i,0])
                #v = prediction[0]
                #y = Y[i,0]
                cost += (v - y) * (v - y)
                print prediction, v, "\t", y, "(", v - y , ")", i
            cost /= len(test_set)
            cost = np.sqrt(cost)
            print "COST:", cost, "Rmin/max", Rmin, Rmax

            if cost < best_cost:
                best_cost = cost
                weights_saved = True
                ANN_DLL.ann_save(ctypes.c_void_p(ann))


        if alpha.value == 0:
            break;

        if True == os.path.exists("C:\\Temp\\test_python\\RRP\\scripts\\ann_t\\STOP.txt"):
            break

    if weights_saved:
        ANN_DLL.ann_restore(ctypes.c_void_p(ann))

    #ann_bias = train_bias_remover(ann, X, Y, test_set)
    ann_bias = None

    pbias = np.array([0], dtype=np.float64)

    # COST last one
    cost = 0.
    for i in test_set:
#    for i in train_set:
        x = X[i,:].astype(np.float64)
        ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))

        ##ANN_DLL.ann_predict(ctypes.c_void_p(ann_bias), x.ctypes.data, pbias.ctypes.data, ctypes.c_int(1))

        m1, m2, v = minmax(Rmin, Rmax, Ymin, Ymax, prediction[0] - pbias[0])
        m1, m2, y = minmax(Rmin, Rmax, Ymin, Ymax, Y[i,0])
        #v = prediction[0]
        #y = Y[i,0]
        cost += (v - y) * (v - y)
        ##print prediction, v, "\t", y, "(", v - y , ")", i
    cost /= len(test_set)
    cost = np.sqrt(cost)
    print "COST:", cost, "Rmin/max", Rmin, Rmax, "(best)", best_cost


    return ann, ann_bias, rf_outliers, cost, Xmin, Xmax, Ymin, Ymax
    ##


def regression(ann, ann_bias, rf_outliers, test_data, Xmin, Xmax, Ymin, Ymax, fnum, cost, pref):
    # regression
    data = test_data

    X = data[:,x_beg:x_end].copy()

##    FX = rf_outliers.predict(X)
##    X = np.append(X, FX.reshape(FX.shape[0],1), axis=1)

    #X = np.append(X, X**2, axis=1)
    #X = np.append(X, np.sin(X), axis=1)
    X = np.append(X, np.log(1. + np.sin(X)), axis=1)


    print "REG: Xmin/max", Xmin, Xmax
    m1, m2, X = minmax(Xmin, Xmax, Rmin, Rmax, X)

    vals = np.zeros((X.shape[0],))

    prediction = np.array([0]*1, dtype=np.float64)
    pbias = np.array([0], dtype=np.float64)

    std = 0.
    mean = 0.

    fname = path_data + "../submission_%s_%f_%d.txt" % (pref, cost, fnum)
    with open(fname, "w+") as fout:
        fout.write("Id,Prediction%s" % os.linesep)
        for row in range(data.shape[0]):
            x = X[row,:].astype(np.float64)
            ANN_DLL.ann_predict(ctypes.c_void_p(ann), x.ctypes.data, prediction.ctypes.data, ctypes.c_int(1))
            if None != ann_bias:
                ANN_DLL.ann_predict(ctypes.c_void_p(ann_bias), x.ctypes.data, pbias.ctypes.data, ctypes.c_int(1))

            #v = prediction[0] * (Ymax - Ymin) # + Ymean
            m1, m2, v = minmax(Rmin, Rmax, Ymin, Ymax, prediction[0] - pbias[0] / 2.)
            #v = prediction[0]

            if 0 == (row % 5000):
                print "ID:", data[row,-1], "val:", v
            fout.write("%d,%2.16f%s" % (data[row,-1], v, os.linesep))

            vals[row] = v


        std = vals.std()
        mean = vals.mean()

        print "ID:", data[row,-1], "val:", v
        print "STD:", std, "MEAN:", mean, "NEGS:", len(vals[vals < 0.])

    #if std <= 1000000. or 1700000. <= std or mean <= 3000000. or 4400000. <= mean:
    #    os.remove(fname)






def main():
    sp.random.seed()

    #pref = sys.argv[1]
    pref = "2nd"

    train = load(path_data + fname_train)
    test = load(path_data + fname_test)


    global Rmin, Rmax
    Rmin = -.5
    Rmax = .5

    Xmin = 0.
    Xmax = 0.

    ann = None

    fnum = 0
    cost = 0.

    N = 10
    cnt = 0.
    for i in range(0, N):
        ann, ann_bias, rf_outliers, tmp_cost, Xmin, Xmax, Ymin, Ymax = process3(train)
        if not np.isnan(tmp_cost):  ## and tmp_cost < 1700000.:
            cost += tmp_cost
            regression(ann, ann_bias, rf_outliers, test, Xmin, Xmax, Ymin, Ymax, fnum, tmp_cost, pref)
            fnum += 1

        ANN_DLL.ann_free(ctypes.c_void_p(ann))
        #ANN_DLL.ann_free(ctypes.c_void_p(ann_bias))

        if True == os.path.exists("C:\\Temp\\test_python\\RRP\\scripts\\ann_t\\STOP.txt"):
            break
    cost /= fnum

    print "AVR COST:", cost



   # train_classifier(train)



if __name__ == '__main__':
    main()




#
# Ymean = 4453532.49635; Ymin = -3.30366e+06; Ymax = 1.52434e+07
#

