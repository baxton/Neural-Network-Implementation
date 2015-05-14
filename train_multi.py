

import os
import sys
import numpy as np
import scipy as sp
import ctypes
from array import array
import datetime as dt


FOR_LINUX = False


OUTPUT_SIZE     = 9

COST_EVERY_ITER = 40
ITERATION_NUM   = 2000

CV_SET_RATE     = .2

TRAIN_SET_RATE  = .6
CONST_ALPHA     = 2.
BATCH_SIZE_RATE = .05

BIG_NUMBER      = 999999999.

DATA_TYPE       = np.float64


delimiter = "\\" if not FOR_LINUX else "/"

path_base = "C:\\Temp\\test_python\\OTTO\\" if not FOR_LINUX else ""
path_data = path_base + "data" + delimiter
path_scripts = path_base + "scripts" + delimiter + "ann" + delimiter


ANN_DLL = ctypes.cdll.LoadLibrary(path_scripts + "libann.so") if FOR_LINUX else \
          ctypes.cdll.LoadLibrary(path_scripts + "ann_70do.dll")


fname_train = "train_data.csv"
fname_test  = "test_data.csv"


#
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
    return data.astype(DATA_TYPE)





def logloss(y, yhat):
    cost = 0.
    for i in range(y.shape[0]):
        cost += y[i] * np.log(yhat[i])
    return cost





def prep_data(X):


    N = X.shape[0]
    C = X.shape[1]

    tmp = X.copy()

    m = tmp.mean(axis=0)
    #s = tmp.std(axis=0)
    s = 1.
    tmp = (tmp - m) / s

    X = tmp;

    return X




class ANN(object):
    def __init__(self, sizes, train_data):
        self.init(sizes, train_data)


    def reset(self):
        ANN_DLL.ann_free(self.ann)
        self.init(self.sizes, self.data)



    def init(self, sizes, train_data):
        self.sizes = sizes

        self.ann = ANN_DLL.ann_create(sizes.ctypes.data, ctypes.c_int(sizes.shape[0]), ctypes.c_int(0))
        self.data = train_data

        self.input_size = sizes[0]
        self.output_size = sizes[-1]

        # prepare target
        self.Y = np.zeros((self.data.shape[0], self.output_size), dtype=DATA_TYPE)
        for i in range(self.Y.shape[0]):
            self.Y[i, int(self.data[i,-1])] = 1.

        # prepare X matrix
        self.X = self.data[:,1:-1]

        # split into train/test sets
        N = self.Y.shape[0]
        self.train_set = range(N)
        sp.random.shuffle(self.train_set)
        self.train_set = self.train_set[:int(N * TRAIN_SET_RATE)]
        self.test_set = [i for i in range(N) if i not in self.train_set]

        # meta parameters
        self.inner_iter = 3
        self.L = .0
        self.alpha = ctypes.c_double(CONST_ALPHA)
        self.MBS = int(len(self.train_set) * BATCH_SIZE_RATE)

        # utility array to store predictions
        self.prediction = np.array([0]*self.output_size, dtype=DATA_TYPE)

        # for memorizing best configuration
        self.best_cost = BIG_NUMBER



    def train(self, iter_num):
        # select the next mini-batch
        indices = choice(self.train_set, self.MBS)

        # prepare data
        Ytmp = self.Y[indices,:].astype(DATA_TYPE)
        Xtmp = self.X[indices,:].astype(DATA_TYPE)

        # feed the data to the ann
        ANN_DLL.ann_fit(ctypes.c_void_p(self.ann), Xtmp.ctypes.data, Ytmp.ctypes.data, ctypes.c_int(self.MBS), ctypes.addressof(self.alpha), ctypes.c_double(self.L), ctypes.c_int(self.inner_iter))
        self.alpha = ctypes.c_double(CONST_ALPHA)

        # COST
        if iter_num > 0 and 0 == (iter_num % COST_EVERY_ITER):
            cost = self.calc_cost()
            print "COST:", cost, "(best)", self.best_cost



    def finish(self):
        final_cost = self.calc_cost()
        return final_cost


    def restore_nn(self):
        ANN_DLL.ann_restore(ctypes.c_void_p(self.ann))

    def save_nn(self):
        ANN_DLL.ann_save(ctypes.c_void_p(self.ann))


    def predict(self, x):
        ANN_DLL.ann_predict(ctypes.c_void_p(self.ann), x.ctypes.data, self.prediction.ctypes.data, ctypes.c_int(1))
        # to make it as close to the real estimation as possible
        self.prediction /= self.prediction.sum()
        # just to prevent a big penalty
        self.prediction[self.prediction==0.] = .001
        return self.prediction

    def calc_cost(self):
        cost = 0.
        for i in self.test_set:
            x = self.X[i,:].astype(DATA_TYPE)
            self.predict(x)
            cost += logloss(self.Y[i], self.prediction)

        cost /= -len(self.test_set)
        return cost





def calc_cv_cost(cv_data, cv_y, anns):
    cost = 0.
    for i in range(cv_data.shape[0]):
        x = cv_data[i,:].astype(DATA_TYPE)
        pred_sum = np.array([0.] * OUTPUT_SIZE, dtype=DATA_TYPE)

        for nn in anns:
            pred_sum += nn.predict(x)

        pred_sum /= len(anns)   # averaging
        pred_sum /= pred_sum.sum()

        cost += logloss(cv_y[i], pred_sum)
    cost /= -cv_data.shape[0]
    return cost


def process(anns, cv_data, cv_y):

    best_cost = BIG_NUMBER

    for iter_num in range(ITERATION_NUM):
        for nn in anns:
            nn.train(iter_num)

        if 0 == (iter_num % COST_EVERY_ITER):
            cv_cost = calc_cv_cost(cv_data, cv_y, anns)
            print "AVR INTERMED COST", cv_cost, "(best)", best_cost

            if cv_cost < best_cost:
                best_cost = cv_cost
                for nn in anns:
                    nn.save_nn()

    # last one
    cv_cost = calc_cv_cost(cv_data, cv_y, anns)
    print "AVR INTERMED COST (L)", cv_cost, "(best)", best_cost

    # check if we have a better solution
    if cv_cost < best_cost:
        best_cost = cv_cost
        for nn in anns:
            nn.save_nn()

        # recalc score
        cv_cost = calc_cv_cost(cv_data, cv_y, anns)
        print "AVR INTERMED COST (re-calc)", cv_cost, "(best)", best_cost


    #
    for nn in anns:
        nn.finish()

    if best_cost != BIG_NUMBER:
        for nn in anns:
            nn.restore_nn()



def regression(anns, test_data, fnum, cost, pref):
    # regression
    data = test_data

    X = data[:,1:-1].copy()

    pred_sum = np.array([0.] * OUTPUT_SIZE, dtype=DATA_TYPE)

    with open(path_data + "../submission_%s_%f_%d.txt" % (pref, cost, fnum), "w+") as fout:
        fout.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9%s" % os.linesep)

        for row in range(data.shape[0]):
            x = X[row,:].astype(DATA_TYPE)

            for nn in anns:
                pred_sum += nn.predict(x)
            pred_sum /= len(anns)   # averaging
            pred_sum /= pred_sum.sum()

            out_str = "%d" % data[row,0]
            for v in pred_sum:
                out_str += ",%f" % v
            out_str += os.linesep

            if 0 == (row % 10000):
                print "ID:", data[row,-1], out_str
            fout.write(out_str)








def main():
    sp.random.seed()

    pref = sys.argv[1]
    #pref = "2nd"

    train = load(path_data + fname_train)
    test = load(path_data + fname_test)

    ROWS    = train.shape[0]
    COLUMNS = train.shape[1] - 2        # -id and -Y

    # select CV set before passing train set to ANNs
    cv_set = range(ROWS)
    np.random.shuffle(cv_set)
    cv_set = cv_set[:int(ROWS * CV_SET_RATE)]
    train_set = [i for i in range(ROWS) if i not in cv_set]


    cv_y = np.zeros((len(cv_set), OUTPUT_SIZE), dtype=DATA_TYPE)
    for i, idx in enumerate(cv_set):
        cv_y[i, int(train[idx,-1])] = 1.

    cv_data = train[cv_set,1:-1]


    # init ANNs
    anns = [
        ANN(np.array([COLUMNS, 66, 55, OUTPUT_SIZE],dtype=int), train[train_set]),
        ANN(np.array([COLUMNS, 77, 66, 55, OUTPUT_SIZE],dtype=int), train[train_set]),
        ANN(np.array([COLUMNS, 88, 66, 33, OUTPUT_SIZE],dtype=int), train[train_set]),
        ANN(np.array([COLUMNS, 43, 22, OUTPUT_SIZE],dtype=int), train[train_set]),
        ANN(np.array([COLUMNS, 35, 35, OUTPUT_SIZE],dtype=int), train[train_set]),
        ANN(np.array([COLUMNS, 45, 45, 15, OUTPUT_SIZE],dtype=int), train[train_set]),

        ANN(np.array([COLUMNS, 55, 55, 55, OUTPUT_SIZE],dtype=int), train[train_set]),
        ANN(np.array([COLUMNS, 33, 33, 33, OUTPUT_SIZE],dtype=int), train[train_set]),
        ANN(np.array([COLUMNS, 65, 55, OUTPUT_SIZE],dtype=int), train[train_set]),
    ]

    fnum = 0
    cost = 0.

    N = 1
    cnt = 0.
    for i in range(0, N):
        fnum += 1

        process(anns, cv_data, cv_y)

        cv_cost = cv_cost = calc_cv_cost(cv_data, cv_y, anns)
        cost += cv_cost
        print "FINAL AVR COST", cv_cost

        regression(anns, test, fnum, cv_cost, pref)

        for nn in anns:
            nn.reset()

        if True == os.path.exists(path_scripts + "STOP.txt"):
            break

    cost /= fnum
    print "AVR COST:", cost



if __name__ == '__main__':
    main()





