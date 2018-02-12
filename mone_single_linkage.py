import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import random
from sklearn.cluster import KMeans
from numpy import array, zeros, argmin, inf
import csv
from numpy import array, zeros, argmin, inf
import mlpy
from sklearn.metrics.pairwise import manhattan_distances
way_csv_file = '/home/bolschikov/Downloads/s02.csv'
dist_fun = manhattan_distances


def read_data_csv(way, num_amplt):
    with open(way, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        name = [list(map(float, i)) for i in reader]  # столбец - объект, значит нужно транспонировать
        print(name)
    np_name = np.transpose(np.array(name))  # транспонируем
    Amplt = [i[: num_amplt] for i in np_name]  # обрежим набор амплитуд до 50 значений
    return np.array(Amplt)

def generate_noise(arrayAmplt, error, metrics="relative"):
    if metrics == "relative":
        for j in range(len(arrayAmplt)):  # цикл для создания погрешности сигнала
            array_Sign = np.random.randint(2, size=len(
                arrayAmplt[0]))  # определение занака входящей погрешности т.е. 0 - "-error", 1 - "+error"
            array_error = np.random.uniform(0, error, size=len(arrayAmplt[0]))
            for q in range(len(arrayAmplt[0])):
                if array_Sign[q] == 1:  # проверка на знак
                    # arrayAmplt[j][q] = arrayAmplt[j][q] + array_error[q]  # * aAmplt[j][numAmplt] / 100.0 #зашумление амплитуд
                    arrayAmplt[j][q] = arrayAmplt[j][q] + array_error[q] * arrayAmplt[j][q] / 100.0
                else:
                    # arrayAmplt[j][q] = arrayAmplt[j][q] - array_error[q]  # * aAmplt[j][numAmplt] / 100.0
                    arrayAmplt[j][q] = arrayAmplt[j][q] - array_error[q] * arrayAmplt[j][q] / 100.0
    if metrics == "swing":
        for j in range(len(arrayAmplt)):  # цикл для создания погрешности сигнала
            array_Sign = np.random.randint(2, size=len(
                arrayAmplt[0]))  # определение занака входящей погрешности т.е. 0 - "-error", 1 - "+error"
            # array_error = np.random.uniform(0, error, size=len(arrayAmplt[0]))
            res_error = (abs(max(arrayAmplt[j])) + abs(min(arrayAmplt[j]))) * error / 100
            for q in range(len(arrayAmplt[0])):
                if array_Sign[q] == 1:  # проверка на знак
                    # arrayAmplt[j][q] = arrayAmplt[j][q] + array_error[q]  # * aAmplt[j][numAmplt] / 100.0 #зашумление амплитуд
                    arrayAmplt[j][q] = arrayAmplt[j][q] + res_error
                else:
                    # arrayAmplt[j][q] = arrayAmplt[j][q] - array_error[q]  # * aAmplt[j][numAmplt] / 100.0
                    arrayAmplt[j][q] = arrayAmplt[j][q] - res_error
    return arrayAmplt


def MonteCarlo(arrayAmplt, numIter, error, clusters):
    #aAmplt - матрица амплитуд сигнала
    #numIter - количество итераций случайного процесса
    #error - погрешность
    Middle = 1
    max_dist, min_dist, midl_dist = [], [], []
    for clust in range(1, clusters + 1):  # внешний цикл для перебора кластеров
        print("num of clust:", clust)
        arr_dist_clust = []  # расстояние между кластерами для каждого clust
        for i in range(numIter):  # цикл для перебора зашумленных сигналов
            print("iteration:", i)
            sum_lenght_clust = 0
            copy_mtrx = copy.deepcopy(arrayAmplt)
            aAmplt = generate_noise(copy_mtrx, error, metrics="swing")

            for j in range(Middle):
                sum_lenght_clust += single_linkage(aAmplt, clust)  # фнкция возвращае в данном случае сумму расстояний
                print("Sum distance:", sum_lenght_clust)
            arr_dist_clust.append(sum_lenght_clust / Middle)  # добавляем эту сумму в массив
        max_dist.append(max(arr_dist_clust))
        min_dist.append(min(arr_dist_clust))
        midl_dist.append(np.mean(np.array(arr_dist_clust)))
        if clust > 1:  # проверка на существование более 2-х кластеров
            print(clust)
            if midl_dist[clust - 1] > min_dist[clust - 2]:
                return max_dist, midl_dist, min_dist, clust


def euclid_dist(fst_obj, sec_obj):
    return (fst_obj[0] - sec_obj[0])**2 + (fst_obj[1] - sec_obj[1])**2


def recurs_in_single_linkage(dist, X, clst, sum_of_dist):
    minim = dist[0][0]
    # print(minim)
    for i in dist:
        if min(i) < minim:
            minim = min(i)
    # flag_one_from_couple = 0
    for index_col, row in enumerate(dist):
        if minim in row:
            buf = [index_col, row.index(minim)]
            # print(buf, dist[buf[0]][buf[1]])
    # print(X)
    for obj in X[buf[0] + buf[1] + 1]:
        X[buf[0]].append(obj)
    X.remove(X[buf[0] + buf[1] + 1])
    # print(X)
    new_dist = []
    for i in X[:-1]:
        buf_dist = []
        min_buf_dist = []
        for j in i:
            min_dist_col = []
            for k in X[X.index(i) + 1:]:
                min_dist_row = []
                for m in k:
                    min_dist_row.append(mlpy.dtw_std(j, m))
                min_dist_col.append(min(min_dist_row))
            buf_dist.append(min_dist_col)
        # print(buf_dist)
        # print(len(buf_dist))
        if len(buf_dist) > 1:  # проверка на матрицу
            for obj in range(len(buf_dist[0])):
                buf_min_for_buf_dist = []
                for min_obj in range(len(buf_dist)):
                    buf_min_for_buf_dist.append(buf_dist[min_obj][obj])
                min_buf_dist.append(min(buf_min_for_buf_dist))
            new_dist.append(min_buf_dist)
        else:
            new_dist.append(buf_dist[0])
    # print(new_dist)
    # print()
    # print()
    if len(X) > clst:
        # print("New recursion")
        sum_of_dist = recurs_in_single_linkage(new_dist, X, clst, sum_of_dist)
    else:
        center = [np.ndarray.tolist(np.mean(i, axis=0)) for i in X]
        # sum_of_dist = 0
        for i in range(len(X)):
            for j in range(len(X[i])):
                sum_of_dist += euclid_dist(center[i], X[i][j])
        return sum_of_dist
    return sum_of_dist



def single_linkage(x, clst):
    # dist = [[mlpy.dtw_std(cur_obj, other_obj) for other_obj in range(cur_obj + 1, len(X))] for cur_obj in range(len(X))]
    dist = []
    X = [[list(i)] for i in x]
    # print()
    # print(len(X))
    # print(len(X[0]))
    # print(len(X[0][0]))
    # print("HELLO")
    for i in X[:-1]:
        buf_dist = []
        for j in i:
            for k in X[X.index(i) + 1:]:
                for m in k:
                    buf_dist.append(mlpy.dtw_std(j, m))
        dist.append(buf_dist)
    print("Dist: ", dist)
    res = recurs_in_single_linkage(dist, X, clst, sum_of_dist=0)
    print(res)
    return res

error = 50
clusters_max = 50
num_iter_for_montecarlo = 10
X = read_data_csv(way_csv_file, num_amplt=100)
# # print(x)
# X = [[i] for i in x]
# # print(X)
listClst = MonteCarlo(X, num_iter_for_montecarlo, error, clusters_max)

o = np.arange(1, listClst[-1] + 1)
ax = plt.figure().gca()
#print len(listClst[0])
# plt.ylabel('Sum of distance ')
# plt.xlabel('Count of clusters')
plt.scatter(o, listClst[0], color='red')
plt.scatter(o, listClst[1], color='black')
plt.scatter(o, listClst[2], color='blue')
plt.grid()
plt.show()
#plt.scatter(o, listClst[3], color = 'green')

# X = [[[0.4, 0.53]], [[0.22, 0.38]], [[0.35, 0.32]], [[0.26, 0.19]], [[0.08, 0.41]], [[0.45, 0.30]]]
# # [plt.scatter(i[0], i[1]) for i in X]
# # plt.gca().set_xlim(left=0)
# # plt.gca().set_ylim(0)
# # plt.grid()
# # plt.show()
# st = time.time()
# single_linkage(X, 2)
# print(time.time() - st)