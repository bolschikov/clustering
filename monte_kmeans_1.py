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


def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    # else:
    #     path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape)


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def metrics(obj, centr, metr="euclid"):  # выбор метрики для формирования расстояния между векторами и центроидами
    if metr == "euclid":
        minus2 = [(obj[i] - centr[i])**2 for i in range(len(obj))]
        return sum(minus2) #  np.sqrt(sum(minus2))
    if metr == "dtw":
        return dtw(obj, centr, dist_fun)


def kmeans(X, clusters, num_iter=300, labels=0, metr="euclid"):
    rnd_centr = random.sample(range(len(X)), clusters)
    print("random center:", rnd_centr)
    # print("random number centroid:", rnd_centr)
    # num_centr = [X[0], X[2]]
    num_centr = [X[i] for i in rnd_centr]
    print("len num of centr in start:", len(num_centr))
    global sum_of_dist
    for n in range(num_iter):
        dist = []
        for obj in X:  # для каждого объекта исходной матрицы ищем расстояние до центройдов
            buf = [mlpy.dtw_std(obj, centr) for centr in num_centr]  # расстояние до каждого центройда
            # print("len of buf:", len(buf))
            dist.append(buf)  # формируем матрицу расстояний
        print("len DIST:", len(dist))
        num_clst = [[d.index(min(d)), min(d)] for d in dist]  # минимальное расстояние до какого либо центройда для каждого объекта
        print()
        print("len num_clst", len(num_clst))
        dict_clst = {}  # словарь для вормирования принодлежности каждого вектора к конкретному классу
        sum_of_dist = 0  # переменная для накопления суммы расстояний до центройдов
        arr_clst = []
        for i in range(len(num_clst)):
            if num_clst[i][0] not in dict_clst:  # проверка на существоание номера кластера в словаре
                dict_clst[num_clst[i][0]] = []
                arr_clst.append(num_clst[i][0])
            dict_clst[num_clst[i][0]].append(X[i])  # добавление конкретного вектора к конкретному кластеру
            sum_of_dist += num_clst[i][1]
        arr_clst.sort()
        print("ln dictionary:", len(dict_clst))
        # print("Sum of dist:", sum_of_dist, " for centroids:", num_centr)
        last_centr = copy.deepcopy(num_centr)  # запоминание предыдущих центров
        num_centr = []
        for key, value in dict_clst.items():
            print("dict items:", key, value)
            num_centr.append(np.ndarray.tolist(np.array(value).mean(axis=0)))  # находим среднее по координатам для пересчета центроидов
        # print("New centroid:", num_centr)
        # print("last_centr:", last_centr, len(last_centr), len(last_centr[0]))
        # print("new center:", num_centr, len(num_centr), len(num_centr[0]))
        if len(num_centr) != len(last_centr):
            continue
        try:
            if (np.array(last_centr) == np.array(num_centr)).all():  # выходим из алгоритма если новые центры совпадают со старыми
                if labels:
                    return dict_clst
                else:
                    return sum_of_dist
        except AttributeError:
            print(len(last_centr))
            print(len(num_centr))
        # print(num_centr, sum_of_dist)
        # print()
    return sum_of_dist


def read_data_csv(way, num_amplt):
    with open(way, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        name = [list(map(float, i)) for i in reader]  # столбец - объект, значит нужно транспонировать
        print(name)
    np_name = np.transpose(np.array(name))  # транспонируем
    Amplt = [i[: num_amplt] for i in np_name]  # обрежим набор амплитуд до 50 значений
    return np.array(Amplt)


def generate_noise(arrayAmplt, error):
    for j in range(len(arrayAmplt)):  # цикл для создания погрешности сигнала
        array_Sign = np.random.randint(2, size=len(arrayAmplt[0]))  # определение занака входящей погрешности т.е. 0 - "-error", 1 - "+error"
        array_error = np.random.uniform(0, error, size=len(arrayAmplt[0]))
        for q in range(len(arrayAmplt[0])):
            if array_Sign[q] == 1:  # проверка на знак
                # arrayAmplt[j][q] = arrayAmplt[j][q] + array_error[q]  # * aAmplt[j][numAmplt] / 100.0 #зашумление амплитуд
                arrayAmplt[j][q] = arrayAmplt[j][q] + array_error[q] * arrayAmplt[j][q] / 100.0
            else:
                # arrayAmplt[j][q] = arrayAmplt[j][q] - array_error[q]  # * aAmplt[j][numAmplt] / 100.0
                arrayAmplt[j][q] = arrayAmplt[j][q] - array_error[q] * arrayAmplt[j][q] / 100.0
    return arrayAmplt


def MonteCarlo(arrayAmplt, numIter, error, clusters):
    #aAmplt - матрица амплитуд сигнала
    #numIter - количество итераций случайного процесса
    #error - погрешность
    Middle = 10
    max_dist, min_dist, midl_dist = [], [], []
    for clust in range(1, clusters + 1):  # внешний цикл для перебора кластеров
        print("num of clust:", clust)
        arr_dist_clust = []  # расстояние между кластерами для каждого clust
        for i in range(numIter):  # цикл для перебора зашумленных сигналов
            print("iteration:", i)
            sum_lenght_clust = 0
            copy_mtrx = copy.deepcopy(arrayAmplt)
            aAmplt = generate_noise(copy_mtrx, error)
            for j in range(Middle):
                sum_lenght_clust += kmeans(aAmplt, clust, metr="dtw")  # фнкция возвращае в данном случае сумму квадратов расстояний
                print("Sum distance:", sum_lenght_clust)
            arr_dist_clust.append(sum_lenght_clust / Middle)  # добавляем эту сумму в массив
        max_dist.append(max(arr_dist_clust))
        min_dist.append(min(arr_dist_clust))
        midl_dist.append(np.mean(np.array(arr_dist_clust)))
        if clust > 1:  # проверка на существование более 2-х кластеров
            print(clust)
            if midl_dist[clust - 1] > min_dist[clust - 2]:
                return max_dist, midl_dist, min_dist, clust


error = 50
clusters_max = 50
num_iter_for_montecarlo = 100
X = read_data_csv(way_csv_file, num_amplt=100)

listClst = MonteCarlo(X, num_iter_for_montecarlo, error, clusters_max)

o = np.arange(1, listClst[-1] + 1)
ax = plt.figure().gca()
#print len(listClst[0])
# plt.ylabel('Sum of distance ')
# plt.xlabel('Count of clusters')
plt.scatter(o, listClst[0], color='red')
plt.scatter(o, listClst[1], color='black')
plt.scatter(o, listClst[2], color='blue')
#plt.scatter(o, listClst[3], color = 'green')