

import os
import sys
import numpy as np
import scipy as sp
import ctypes
from array import array
import datetime as dt


FOR_LINUX = False


OUTPUT_SIZE     = 9

COST_EVERY_ITER = 30
ITERATION_NUM   = 10000

CV_SET_RATE     = .2

TRAIN_SET_RATE  = .6
CONST_ALPHA     = .2
BATCH_SIZE_RATE = .5

BIG_NUMBER      = 999999999.

DATA_TYPE       = np.float64


delimiter = "\\" if not FOR_LINUX else "/"

path_base = "C:\\Temp\\test_python\\OTTO\\" if not FOR_LINUX else ""
path_data = path_base + "data" + delimiter
path_scripts = path_base + "scripts" + delimiter + "ann" + delimiter


ANN_DLL = ctypes.cdll.LoadLibrary(path_scripts + "libann.so") if FOR_LINUX else \
          ctypes.cdll.LoadLibrary(path_scripts + "ann2.dll")


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



def minmax(min1, max1, min2, max2, X, ignore=True):
    if ignore:
        return 0, 0, X

    if min1 == None or max1 == None:
        min1 = X.min(axis=0)
        max1 = X.max(axis=0)


    tmp = (max1 - min1)
    tmp[tmp==0.] = 0.00000001

    k = (max2 - min2) / tmp
    X -= min1
    X *= k
    X += min2

    return min1, max1, X




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
    def __init__(self, sizes, train_data, target, internal_iters=5):
        self.init(sizes, train_data, target, internal_iters)


    def reset(self):
        ANN_DLL.ann_free(self.ann)
        self.init(self.sizes, self.data, self.Y, self.inner_iter)



    def init(self, sizes, train_data, target, internal_iters):
        self.sizes = sizes

        self.ann = ANN_DLL.ann_create(sizes.ctypes.data, ctypes.c_int(sizes.shape[0]), ctypes.c_int(0))
        self.data = train_data

        self.input_size = sizes[0]
        self.output_size = sizes[-1]

        # prepare target
        self.Y = target

        # prepare X matrix
        self.X = self.data

        # split into train/test sets
        N = self.Y.shape[0]
        self.train_set = range(N)
        sp.random.shuffle(self.train_set)
        self.train_set = self.train_set[:int(N * TRAIN_SET_RATE)]
        self.test_set = [i for i in range(N) if i not in self.train_set]

        # meta parameters
        self.inner_iter = internal_iters
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


    def set_output_scale(self, val):
        ANN_DLL.ann_set_output_scale(ctypes.c_void_p(self.ann), ctypes.c_double(float(val)))

    def get_output(self, l):
        Y = np.zeros((self.sizes[-l],), dtype=DATA_TYPE)
        ANN_DLL.ann_get_output(ctypes.c_void_p(self.ann), Y.ctypes.data, ctypes.c_int(2))
        return Y


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


    def get_weights(self):
        ww_size = 0
        bb_size = 0
        for l in range(1, len(self.sizes)):
            ww_size += self.sizes[l] * self.sizes[l-1]
            bb_size += self.sizes[l]

        ww = np.zeros((ww_size,), dtype=DATA_TYPE)
        bb = np.zeros((bb_size,), dtype=DATA_TYPE)

        ANN_DLL.ann_get_weights(ctypes.c_void_p(self.ann), bb.ctypes.data, ww.ctypes.data)

        return ww, bb


    def get_1st_weights(self):
        ww_size = self.sizes[0] * self.sizes[1]
        bb_size = self.sizes[1]

        ww, bb = self.get_weights()

        return ww[:ww_size], bb[:bb_size], ww_size, bb_size


    def set_weights(self, ww, bb):
        ANN_DLL.ann_set_weights(ctypes.c_void_p(self.ann), bb.ctypes.data, ww.ctypes.data)


#############################################################################
##
#############################################################################



def feed_through_stack(ann_stack, x):
    p = x
    for ann in ann_stack:
        ann.predict(x)
        p = ann.get_output(2)
        x = p
    return p


def ann_deep_pretrain(sizes, train_data, target):
    hidden_num = len(sizes) - 3
    if hidden_num <= 0:
        raise Exception("ANN is not deep: %s" %sizes)

    Xmax = train_data.max()

    data = None
    ann_steck = []
    cur_ann = None
    ann_num = 0

    output_scale = 10.

    global BATCH_SIZE_RATE
    batch_size = BATCH_SIZE_RATE
    BATCH_SIZE_RATE = 1.

    while ann_num < hidden_num:
        # current level to train
        l = ann_num + 1
        print "training level", l

        if l > 1:
            output_scale = 1.

        # prepare data
        if data == None:
            data = train_data
        else:
            tmp = np.zeros((train_data.shape[0], sizes[l-1]), dtype=DATA_TYPE)
            for idx, x in enumerate(train_data):
                p = feed_through_stack(ann_steck, x)
                tmp[idx,:] = p
            data = tmp

        cur_ann = ANN(np.array([sizes[l-1], sizes[l], sizes[l-1]], dtype=int), data, data, 10)
        cur_ann.set_output_scale(output_scale)


        N = 1000 if l == 1 else 50
        for iter_num in range(N):
            cur_ann.train(iter_num)

            if l == 1 and output_scale < Xmax and iter_num > 0 and 0 == (iter_num % 10):
                output_scale += 5.
                cur_ann.set_output_scale(output_scale)
                print "set scale:", output_scale


        # add to the steck
        cur_ann.set_output_scale(1.)
        ann_steck.append(cur_ann)
        ann_num += 1


    # composing ww / bb
    ww_size = 0
    bb_size = 0
    for l in range(1, len(sizes)):
        ww_size += sizes[l] * sizes[l-1]
        bb_size += sizes[l]

    ww = sp.rand(ww_size)
    bb = sp.rand(bb_size)

    ww_idx = 0
    bb_idx = 0
    for ann in ann_steck:
        ww_tmp, bb_tmp, ww_size, bb_size = ann.get_1st_weights()
        ww[ww_idx : ww_idx + ww_size] = ww_tmp
        bb[bb_idx : bb_idx + bb_size] = bb_tmp

        ww_idx += ww_size
        bb_idx += bb_size

    BATCH_SIZE_RATE = batch_size

    ann = ANN(sizes, train_data, target)
    ann.set_weights(ww, bb)

    return ann




#############################################################################
##
#############################################################################



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
        if True == os.path.exists(path_scripts + "STOP.txt"):
            break


    # last one
    cv_cost = calc_cv_cost(cv_data, cv_y, anns)
    print "AVR INTERMED COST (L)", cv_cost, "(best)", best_cost

    # check if we have a better solution
    if cv_cost > best_cost:
        for nn in anns:
            nn.restore_nn()

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

    target = np.zeros((ROWS, OUTPUT_SIZE), dtype=DATA_TYPE)
    for i in range(ROWS):
        target[i, int(train[i,-1])] = 1.

    cv_y = target[cv_set]
    cv_data = train[cv_set,1:-1]


##    # init ANNs
##    anns = [
##        ANN(np.array([COLUMNS, 80, 70, 60, 70, 80, OUTPUT_SIZE],dtype=int), train[train_set,1:-1], target[train_set]),
##        ANN(np.array([COLUMNS, 80, 70, 60, 70, 80, OUTPUT_SIZE],dtype=int), train[train_set,1:-1], target[train_set]),
##        #ANN(np.array([COLUMNS, 80, 70, OUTPUT_SIZE],dtype=int), train[train_set]),
##
##    ]

    ann = ann_deep_pretrain(np.array([COLUMNS, 80, 20, 10, 9, OUTPUT_SIZE],dtype=int), train[train_set,1:-1], target[train_set])
    anns = [ ann ]


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





