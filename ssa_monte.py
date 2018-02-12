import numpy as np
import csv
import copy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.ticker import MaxNLocator
import time
import random
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('axes', labelsize=20)    # fontsize of the x and y labels
way_csv_file = '/home/bolschikov/Downloads/s02.csv'


def read_data_csv(way, num_amplt):
    with open(way, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        name = [list(map(float, i)) for i in reader]  # столбец - объект, значит нужно транспонировать
        print(name)
    np_name = np.transpose(np.array(name))  # транспонируем
    Amplt = [i[: num_amplt] for i in np_name]  # обрежим набор амплитуд до 50 значений
    return np.array(Amplt)

# def generate_noise(arrayAmplt, error):
#     for j in range(len(arrayAmplt)):  # цикл для создания погрешности сигнала
#         array_Sign = np.random.randint(2, size=len(arrayAmplt[0]))  # определение занака входящей погрешности т.е. 0 - "-error", 1 - "+error"
#         array_error = np.random.uniform(0, error, size=len(arrayAmplt[0]))
#         for q in range(len(arrayAmplt[0])):
#             if array_Sign[q] == 1:  # проверка на знак
#                 # arrayAmplt[j][q] = arrayAmplt[j][q] + array_error[q]  # * aAmplt[j][numAmplt] / 100.0 #зашумление амплитуд
#                 arrayAmplt[j][q] = arrayAmplt[j][q] + array_error[q] * arrayAmplt[j][q] / 100.0
#             else:
#                 # arrayAmplt[j][q] = arrayAmplt[j][q] - array_error[q]  # * aAmplt[j][numAmplt] / 100.0
#                 arrayAmplt[j][q] = arrayAmplt[j][q] - array_error[q] * arrayAmplt[j][q] / 100.0
#     return arrayAmplt

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


def snake(X, L):
    N = len(X)
    k = N - L + 1
    XX = []
    for i in range(k):
        buf = []
        for j in range(L):
            buf.append(X[i + j])
        XX.append(buf)
    return XX


def diag_mid(arr_X, N, L):
    f = []
    K = N - L + 1
    for k in range(N):
        if k < L - 1:
            summ = 0
            for m in range(0, k + 1):
                summ += arr_X[m][k - m]
            f.append(1 / (k + 1) * summ)
        if k >= L - 1 and k < K:
            summ = 0
            for m in range(0, L):
                summ += arr_X[m][k - m]
            f.append(1 / L * summ)
        if k >= K and k < N:
            summ = 0
            for m in range(k - K + 1, N - K + 1):
                summ += arr_X[m][k - m]
            f.append(1 / (N - k) * summ)
    return f


def filter_noize(array_mtrx, error, arr_from_scv, L, N, show=0):
    arr_up = arr_from_scv + error * arr_from_scv / 100
    arr_down = arr_from_scv - error * arr_from_scv / 100
    X = array_mtrx[0] - array_mtrx[0]
    n = len(array_mtrx)
    step = int(n / 20) # % от количества собственных значений будет брибавляться
    if step == 0:
        step = 1
    amnt_step = int(n / step)

    for cnt in range(1, amnt_step):
        for num in range((cnt - 1) * step, cnt * step):
            X += array_mtrx[num]
        res_ssa = diag_mid(X, N, L)
        for i in range(N):
            if abs(arr_up[i]) < abs(res_ssa[i]) or abs(res_ssa[i]) < abs(arr_down[i]):

                break
        # print(i)
        if show:
            d = [i for i in range(N)]
            plt.plot(d, arr_from_scv, color="green")
            # plt.plot(d, arr_up - error, color="pink")
            plt.plot(d, res_ssa, color='blue')
            plt.scatter(d, res_ssa, color='blue')
            plt.scatter(i, res_ssa[i], color='red')
            plt.plot(d, arr_up, color="black")
            plt.plot(d, arr_down, color='black')
            plt.show()
        if i == len(arr_up) - 1:
            return num + 1

def create_eign_after_ssa(matrix_traketor, eig_vec):
    trans_mtrx = np.transpose(np.array(matrix_traketor))  # преобразование матрицы по Голяндиной
    mtrx_for_svd = np.dot(trans_mtrx, np.transpose(trans_mtrx))  # X*X.Trans
    u, s, v_utrans = np.linalg.svd(mtrx_for_svd)  # сингулярное разложение
    V = [np.dot(np.transpose(trans_mtrx), v_utrans[i]) / np.sqrt(s[i]) for i in range(len(s))]
    X = [np.outer((v_utrans[i]), V[i]) * np.sqrt(s[i]) for i in range(len(s))]
    if eig_vec:
        return X
    else:
        return s


def num_of_eign(matrix_of_sign, L, error):
    arr_eign = []
    flag = 1
    for length in matrix_of_sign:
        mtrx = snake(length, L)
        X = create_eign_after_ssa(mtrx, flag)
        arr_eign.append(filter_noize(X, error, length, L, len(length)))
        if arr_eign[-1] > 6: # посмотреть вид графика если число собственных значений велико
            print(length)
            print(arr_eign[-1])
            X = create_eign_after_ssa(mtrx, 1)
            filter_noize(X, error, length, L, len(length), 1)
    print(arr_eign)
    print(len(arr_eign))
    return max(arr_eign)


def euclid_dist(fst_obj, sec_obj):
    return (fst_obj[0] - sec_obj[0])**2 + (fst_obj[1] - sec_obj[1])**2


def clustering(clst, array, flag):
    return kmeans(array, clst)
    # model = KMeans(n_clusters=clst)
    # model.fit(array)
    # if flag == 0:
    #     return model.inertia_
    # else:
    #     return model.labels_
def kmeans(X, clusters, num_iter=300, labels=0, metr="euclid"):
    rnd_centr = random.sample(range(len(X)), clusters)
    # print("random center:", rnd_centr)
    # print("random number centroid:", rnd_centr)
    # num_centr = [X[0], X[2]]
    num_centr = [X[i] for i in rnd_centr]
    # print("len num of centr in start:", len(num_centr))
    global sum_of_dist
    for n in range(num_iter):
        dist = []
        for obj in X:  # для каждого объекта исходной матрицы ищем расстояние до центройдов
            buf = [euclid_dist(obj, centr) for centr in num_centr]  # расстояние до каждого центройда
            # print("len of buf:", len(buf))
            dist.append(buf)  # формируем матрицу расстояний
        # print("len DIST:", len(dist))
        num_clst = [[d.index(min(d)), min(d)] for d in
                    dist]  # минимальное расстояние до какого либо центройда для каждого объекта
        # print()
        # print("len num_clst", len(num_clst))
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
        # print("ln dictionary:", len(dict_clst))
        # print("Sum of dist:", sum_of_dist, " for centroids:", num_centr)
        last_centr = copy.deepcopy(num_centr)  # запоминание предыдущих центров
        num_centr = []
        for key, value in dict_clst.items():
            # print("dict items:", key, value)
            num_centr.append(np.ndarray.tolist(
                np.array(value).mean(axis=0)))  # находим среднее по координатам для пересчета центроидов
        # print("New centroid:", num_centr)
        # print("last_centr:", last_centr, len(last_centr), len(last_centr[0]))
        # print("new center:", num_centr, len(num_centr), len(num_centr[0]))
        if len(num_centr) != len(last_centr):
            continue
        try:
            if (np.array(last_centr) == np.array(
                    num_centr)).all():  # выходим из алгоритма если новые центры совпадают со старыми
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

def Monte_Carlo(numIter, matrix_of_sign, error, clusters, L, num_eign):
    num_of_iter_for_clustering = 10
    max_dist, min_dist, midl_dist = [], [], []
    for clust in range(1, clusters + 1):
        arr_dist_clust = []  # расстояние между кластерами для каждого clust
        N = len(matrix_of_sign[0])  # количество отсчетов в каждом векторе
        for iter in range(numIter):
            copy_mtrx = copy.deepcopy(matrix_of_sign)
            # print(matrix_of_sign[0])
            matrix_noize = generate_noise(copy_mtrx, error, metrics="swing")
            arr_eign = []  # list собственных значений кажого вектора time series
            for m in range(len(matrix_noize)):
                mtrx = snake(matrix_noize[m], L)  # построенире траекторной матрицы
                arr_eign.append(create_eign_after_ssa(mtrx, eig_vec=0))
                # # нажать "+" и раскомментировать кусок кода для вывода графиков и результатов разложения ssa
                # #
                # X = create_eign_after_ssa(mtrx, eig_vec=1)  # создание массива собственных троек/чисел
                # sum_of_eign_vect = X[0] - X[0]  # неходимо сформировть матрицу нулей
                # for eign_vect in range(num_eign):  # суммирование по собственным тройкам
                #     sum_of_eign_vect += X[eign_vect]
                # test = sum(X[0:num_eign])  # суммирование за счет python функций
                # test_res = diag_mid(test, N, L)  # диаганальное усреднение python суммы
                # res = diag_mid(sum_of_eign_vect, N, L)  # диаганальное усреднение
                #
                # # отрисовка результатов восстановления графика функции
                # d = [i for i in range(N)]
                # plt.ylabel(u'ЗНАЧЕНИЕ СИГНАЛА')
                # plt.xlabel(u'ВРЕМЯ')
                # plt.title(u'СИГНАЛ')
                # plt.plot(d, res, color="green")
                # plt.plot(d, test_res, color='blue')
                # # plt.plot(d, matrix_noize[m], color="pink")  # вывод сгенерированного шума
                # plt.plot(d, matrix_of_sign[m] + error * matrix_of_sign[m] / 100, color="black")
                # plt.plot(d, matrix_of_sign[m] - error * matrix_of_sign[m] / 100, color='black')
                # plt.grid()
                # plt.show()
            sum_lenght_clust = 0
            for count in range(num_of_iter_for_clustering):
                sum_lenght_clust += clustering(clust, arr_eign, 0)  #  усреднение расстояний между кластерами
            arr_dist_clust.append(sum_lenght_clust / num_of_iter_for_clustering)  #  хранение результатов монтекарло
        max_dist.append(max(arr_dist_clust))
        min_dist.append(min(arr_dist_clust))
        midl_dist.append(np.mean(np.array(arr_dist_clust)))
        if clust > 1:
            print(clust)
            if midl_dist[clust - 1] > min_dist[clust - 2]:
                return max_dist, midl_dist, min_dist, clust

st_time = time.time()
error = 50
clusters_max = 50
num_iter_for_montecarlo = 1000

matrix_from_csv = read_data_csv(way_csv_file, num_amplt=100)
L = int(len(matrix_from_csv[0]) / 2 - len(matrix_from_csv[0]) / 10)  # длина окна (длина гусеницы)
max_eign_value = num_of_eign(matrix_from_csv, L, error)
list_answer = Monte_Carlo(num_iter_for_montecarlo, matrix_from_csv, error, clusters_max, L, max_eign_value)
print(time.time() - st_time)
ax = plt.figure().gca()
o = np.arange(1, list_answer[-1] + 1)
ax.scatter(o, list_answer[0], color='red')
ax.scatter(o, list_answer[1], color='black')
ax.scatter(o, list_answer[2], color='blue')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid()
plt.show()