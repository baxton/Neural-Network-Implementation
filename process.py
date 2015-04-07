
import os
import numpy as np
import scipy as sp
from array import array



path_data = "C:\\Temp\\test_python\\RRP\\data\\"

fname_train = "train.csv"
fname_test  = "test.csv"

REV_MEAN = 4453532.6131386859

ARR_LEN = 42
ROWS = 137


city_cat = None
cafe_type_cat = None


def fill_dicts(token_idx):
    categories = {}
    idx = 0

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





def process_train():

    data = []

    with open(path_data + fname_train, "r") as fin:
        fin.readline()  # header
        for line in fin:
            line = line.strip()
            tokens = line.split(',')

            row = []

            # date
            date_tokens = tokens[1].split('/')
            date_y = date_tokens[2]
            date_m = date_tokens[0]
            date_d = date_tokens[1]

            # city type
            city_type = -1. if tokens[3] == "Big Cities" else 1.

            # city
            city = get_one_hot(city_cat, tokens[2])

            # cafe type
            cafe = get_one_hot(cafe_type_cat, tokens[4])

            # Pxxx
            PP = [float(v) for v in tokens[5:-1]]

            # revenue
            revenue = float(tokens[-1])


            # reconstruct
            row.append(date_y)
            row.append(date_m)
            row.append(date_d)
            row.append(city_type)
            row.extend(city)
            row.extend(cafe)
            row.extend(PP)
            row.append(revenue)

            data.append(row)

    return np.array(data, dtype=np.float64)



def process_test():

    data = []

    with open(path_data + fname_test, "r") as fin:
        fin.readline()  # header
        for line in fin:
            line = line.strip()
            tokens = line.split(',')

            row = []

            # id
            id = int(tokens[0])

            # date
            date_tokens = tokens[1].split('/')
            date_y = date_tokens[2]
            date_m = date_tokens[0]
            date_d = date_tokens[1]

            # city type
            city_type = -1. if tokens[3] == "Big Cities" else 1.

            # city
            city = get_one_hot(city_cat, tokens[2])

            # cafe type
            cafe = get_one_hot(cafe_type_cat, tokens[4])

            # Pxxx
            PP = [float(v) for v in tokens[5:]]

            # reconstruct
            row.append(date_y)
            row.append(date_m)
            row.append(date_d)
            row.append(city_type)
            row.extend(city)
            row.extend(cafe)
            row.extend(PP)
            row.append(id)

            data.append(row)

    return np.array(data, dtype=np.float64)



def main():
    #
    # prepare categorical vars
    #
    global city_cat, cafe_type_cat

    city_cat = fill_dicts(2)
    cafe_type_cat = fill_dicts(4)

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
