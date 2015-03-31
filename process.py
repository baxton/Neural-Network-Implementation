
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



def save_dic(map, fname):
    with open(path_data + fname + ".txt", "w+") as fout:
        for k in map:
            fout.write("%s,%2.16f%s" % (k, map[k], os.linesep))


def save_data(data, fname):
    np.savetxt(path_data + fname + ".csv", data, delimiter=',')


def save_stat(data, fname):
    means = data.mean(axis=0)
    mins = data.min(axis=0).flatten()
    maxs = data.max(axis=0).flatten()
    with open(path_data + fname + ".csv", "w+") as fout:
        np.savetxt(fout, means.reshape((1, means.shape[0])), delimiter=',')
        np.savetxt(fout, mins.reshape((1, mins.shape[0])), delimiter=',')
        np.savetxt(fout, maxs.reshape((1, maxs.shape[0])), delimiter=',')



def process_train():

    current_val = 0
    vals = sp.linspace(-.8, .8, num=500)
    vals = vals[vals != 0.0]    # remove zero as it's a special value
    sp.random.shuffle(vals)


    data = np.zeros((ROWS, ARR_LEN))

    city_map = {}
    cafe_type_map = {}

    row = 0

    with open(path_data + fname_train, "r") as fin:
        fin.readline()  # header
        for line in fin:
            line = line.strip()
            tokens = line.split(',')

            # date
            date_tokens = tokens[1].split('/')
            date = int(date_tokens[2]) * 10000 + int(date_tokens[0]) * 100 + int(date_tokens[1])

            # city type
            city_type = -1. if tokens[3] == "Big Cities" else 1.

            # city
            city = 0
            if not tokens[2] in city_map:
                city_map[tokens[2]] = vals[current_val]
                current_val += 1
            city = city_map[tokens[2]]

            # cafe type
            cafe = 0
            if not tokens[4] in cafe_type_map:
                cafe_type_map[tokens[4]] = vals[current_val]
                current_val += 1
            cafe = cafe_type_map[tokens[4]]

            # Pxxx
            PP = [float(v) for v in tokens[5:-1]]

            # revenue
            revenue = float(tokens[-1])


            # reconstruct
            data[row, :] = [date, city, city_type, cafe] + PP + [revenue]

            row += 1

    return data, city_map, cafe_type_map



def process_test(city_map, cafe_type_map):

    data = np.zeros((100001, ARR_LEN))

    row = 0

    with open(path_data + fname_test, "r") as fin:
        fin.readline()  # header
        for line in fin:
            line = line.strip()
            tokens = line.split(',')

            # id
            id = int(tokens[0])

            # date
            date_tokens = tokens[1].split('/')
            date = int(date_tokens[2]) * 10000 + int(date_tokens[0]) * 100 + int(date_tokens[1])

            # city type
            city_type = -1. if tokens[3] == "Big Cities" else 1.

            # city
            city = 0
            if tokens[2] in city_map:
                city = city_map[tokens[2]]

            # cafe type
            cafe = 0
            if tokens[4] in cafe_type_map:
                cafe = cafe_type_map[tokens[4]]

            # Pxxx
            PP = [float(v) for v in tokens[5:]]

            # reconstruct
            data[row, :] = [date, city, city_type, cafe] + PP + [id]

            row += 1

    return data



def main():
    data, city_map, cafe_type_map = process_train()

    save_dic(city_map, "city_map")
    save_dic(cafe_type_map, "cafe_type_map")
    save_stat(data, "train_stat")
    save_data(data, "train_data")

    test_data = process_test(city_map, cafe_type_map)
    save_data(test_data, "test_data")




if __name__ == '__main__':
    main()
