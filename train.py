

import os
import numpy as np
import scipy as sp
import ctypes
from array import array


ANN_DLL = ctypes.cdll.LoadLibrary(r"C:\Temp\test_python\RRP\scripts\ann\ann.dll")


path_data = "C:\\Temp\\test_python\\RRP\\data\\"

fname_train = "train_data.csv"
fname_test  = "test_data.csv"

REV_MEAN = 4453532.6131386859

ARR_LEN = 42
VEC_LEN = 41




def load(fname):
    data = np.loadtxt(fname, delimiter=',')
    return data.astype(np.float32)



def process():
    data = load(path_data + fname_train)

    Y = data[:,-1]
    Ymean = Y.mean()
    Y -= Ymean
    Ymin = Y.min()
    Ymax = Y.max()
    Y /= (Ymax - Ymin)

    print "Y mean:", Ymean, "Ymin:", Ymin, "Ymax:", Ymax

    X = data[:,:-1]
    Xmean = X.mean(axis=0)
    X -= Xmean
    Xmin = X.min(axis=0)
    Xmax = X.max(axis=0)
    X /= (Xmax - Xmin)

    print "X mean:", Xmean, "Xmin:", Xmin, "Xmax:", Xmax

    N = Y.shape[0]
    alpha = ctypes.c_float(.008)

    ann = ANN_DLL.ann_create()
    ANN_DLL.ann_fit(ctypes.c_void_p(ann), X.ctypes.data, Y.ctypes.data, ctypes.c_int(N), ctypes.addressof(alpha), ctypes.c_float(0), ctypes.c_int(20000))



def main():
    process()




if __name__ == '__main__':
    main()
