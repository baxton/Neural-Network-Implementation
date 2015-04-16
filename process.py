
import os
import numpy as np
import scipy as sp
from array import array
import datetime as dt



#path_data = "/home/maxim/kaggle/RRP/data/"
path_data = "C:\\Temp\\test_python\\RRP\\data\\"

fname_train = "train.csv"
fname_test  = "test.csv"

today = dt.date.today()

REV_MEAN = 4453532.6131386859

ARR_LEN = 43
ROWS = 137

NULL = 0.0001

city_cat = None
cafe_type_cat = None
pp_cat = [None] * (ARR_LEN - 5 - 1)

def fill_dicts(token_idx):
    categories = {}
    idx = 1

    with open(path_data + fname_train, "r") as fin:
        fin.readline()  # header
        for line in fin:
            line = line.strip()
            tokens = line.split(',')

            key = tokens[token_idx]
            if not key in categories:
                categories[key] = idx
                idx += 1

    return categories
##

def get_one_hot(category, key):
    val = np.zeros((len(category),))
    if key in category:
        val[category[key]] = 1.
    return val
##




def save_data(data, fname):
    np.savetxt(path_data + fname + ".csv", data, delimiter=',')


def save_stat(data, fname):
    means = data.mean(axis=0)
    mins = data.min(axis=0).flatten()
    maxs = data.max(axis=0).flatten()
    std = data.std(axis=0).flatten()
    with open(path_data + fname + ".csv", "w+") as fout:
        fout.write("ROWS: %d%s" % (data.shape[0], os.linesep))
        fout.write("COLS: %d%s" % (data.shape[1], os.linesep))

        np.savetxt(fout, means.reshape((1, means.shape[0])), delimiter=',')
        np.savetxt(fout, mins.reshape((1, mins.shape[0])), delimiter=',')
        np.savetxt(fout, maxs.reshape((1, maxs.shape[0])), delimiter=',')
        np.savetxt(fout, std.reshape((1, maxs.shape[0])), delimiter=',')





def get_row(tokens, for_test=False):
    row = []

    if for_test:
        # id
        id = int(tokens[0])


    # date
    date_tokens = tokens[1].split('/')
    date_y = int(date_tokens[2])
    date_m = int(date_tokens[0])
    date_d = int(date_tokens[1])

    # month
    #month = [0] * 12
    #month[date_m-1] = 1

    # age
    d = dt.date(date_y, date_m, date_d)
    age = (today - d).days + 1

    # city type
    city_type = .9 if tokens[3] == "Big Cities" else .1

    # city
    city = city_cat[tokens[2]] if tokens[2] in city_cat else NULL

    # cafe type
    cafe = cafe_type_cat[tokens[4]] if tokens[4] in cafe_type_cat else NULL

    # Pxxx
    if for_test:
        PP = [float(v) for v in tokens[5:]]
    else:
        PP = [float(v) for v in tokens[5:-1]]

    # revenue
    revenue = float(tokens[-1])


    # reconstruct
    row.append(date_y)
    row.append(date_m)
    row.append(date_d)
    row.append(age)
    row.append(city_type)
    row.append(city)
    row.append(cafe)
    row.extend(PP)
    if for_test:
        row.append(id)
    else:
        row.append(revenue)

    return row


def aug_one(tokens, data, num=5):
    # only touch PPs: 5..(ARR_LEN-1)
    tt = tokens[:]  # copy
    for i in range(num):
        for j in range(4):
            idx = np.random.randint(5, ARR_LEN-1, 1)
            tt[idx] = 0
        row = get_row(tt, False)
        data.append(row)




def process_train():

    data = []

    with open(path_data + fname_train, "r") as fin:
        fin.readline()  # header
        for line in fin:
            line = line.strip()
            tokens = line.split(',')

            row = get_row(tokens)
            data.append(row)

            ##aug_one(tokens, data, num=15)

    return np.array(data, dtype=np.float64)




def process_test():

    data = []

    with open(path_data + fname_test, "r") as fin:
        fin.readline()  # header
        for line in fin:
            line = line.strip()
            tokens = line.split(',')

            row = get_row(tokens, True)
            data.append(row)

    return np.array(data, dtype=np.float64)



def main():
    #
    # prepare categorical vars
    #
    global city_cat, cafe_type_cat

    city_cat = fill_dicts(2)
    cafe_type_cat = fill_dicts(4)

    for i in range(5, ARR_LEN-1):
        pp_cat[i-5] = fill_dicts(i)


    #
    # encoding train set
    #
    data = process_train()

    save_stat(data, "train_stat")
    save_data(data, "train_data")

    #
    # encoding test set
    #
    test_data = process_test()

    save_stat(test_data, "test_stat")
    save_data(test_data, "test_data")




if __name__ == '__main__':
    main()

