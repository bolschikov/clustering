import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import copy
import random
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('axes', labelsize=20)    # fontsize of the x and y labels
from matplotlib.ticker import MaxNLocator
way_csv_file = '/home/bolschikov/Downloads/s02.csv'


def read_data_csv(way, num_amplt):
    with open(way, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        name = [list(map(float, i)) for i in reader]  # столбец - объект, значит нужно транспонировать
    np_name = np.transpose(np.array(name))  # транспонируем
    Amplt = [i[: num_amplt] for i in np_name]  # обрежим набор амплитуд до 50 значений
    return np.array(Amplt)


def euclid_dist(fst_obj, sec_obj):
    return (fst_obj[0] - sec_obj[0])**2 + (fst_obj[1] - sec_obj[1])**2


def Clustering(clst, array, flag):
    pca = PCA()
    pca.fit(array)
    k = 0
    sumRatio = 0
    #sumRes = [j for j in pca.explained_variance_ratio_]
    while sumRatio <= 0.99:# нас интересуют главные компоненты которые покрывают 99% разброса
        sumRatio = sumRatio + pca.explained_variance_ratio_[k]
        k = k + 1
    trnsfRes = pca.transform(array)
    total = [x[:k] for x in trnsfRes]
    return kmeans(total, clst)
    # model = KMeans(n_clusters=clst)
    # model.fit(total)
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
        num_clst = [[d.index(min(d)), min(d)] for d in dist]  # минимальное расстояние до какого либо центройда для каждого объекта
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
    # resTotal = [] #создание массива для суммирования амплитуд гармоник
    # aSumDist2 = [] #создание массива для суммирования суммы квадратов расстояний между центрами кластеров
    # MaxDist2 = [] #создание массивов расстояний
    # MinDist2 = []
    # Dist2 = []
    # meanAmplTotal = []
    # KONEZ = []
    # SumForGraphics = []
    Middle = 10
    max_dist, min_dist, midl_dist = [], [], []
    for clust in range(1, clusters + 1): #внешний цикл для перебора кластеров
        print(clust)
        arr_dist_clust = []  # расстояние между кластерами для каждого clust
        for i in range(numIter):  # цикл для перебора зашумленных сигналов
            print(i)
            sum_lenght_clust = 0
            copy_mtrx = copy.deepcopy(arrayAmplt)
            aAmplt = generate_noise(copy_mtrx, error, metrics="swing")
            fftRes = [np.fft.fft(x) for x in aAmplt] #применение БПФ
            res = np.absolute(fftRes) #формируем матрицу распределения амплитуд гармоник по частотам
            # отрисовка результатов восстановления графика функции
            # a = len(fftRes[0])
            # freqNp = np.fft.fftfreq(a)
            # for c in res:
            #     plt.plot(freqNp, c)
            # plt.ylabel(u'АМПЛИТУДА')
            # plt.xlabel(u'ЧАСТОТА')
            # plt.title(u'БПФ')
            # plt.show()
            # if i > 0:
            #     resTotal[clust - 1] = resTotal[clust - 1] + res #суммируем получившиеся амплитуд
            # else:
            #     resTotal.append(res) # создание массива для суммирования амплитуд и дальнейшего их усреднения
            for j in range(Middle):
                sum_lenght_clust += Clustering(clust, res, 0)# фнкция возвращае в данном случае сумму квадратов расстояний
            arr_dist_clust.append(sum_lenght_clust / Middle) #добавляем эту сумму в массив
        max_dist.append(max(arr_dist_clust))
        min_dist.append(min(arr_dist_clust))
        midl_dist.append(np.mean(np.array(arr_dist_clust)))
        # resTotal[clust - 1] = resTotal[clust - 1] / numIter
        # #resTotal[clust - 1] = [x / numIter for x in resTotal[clust - 1]]
        # #print type(resTotal)
        # #print type(numIter)
        # #print resTotal
        # #print
        # meanAmplTotal.append(resTotal[clust - 1]) # среднее значение аплитуд гармноик
        # KONEZ.append(Clustering(clust, meanAmplTotal[clust - 1], 0))
        if clust > 1:# проверка на существование более 2-х кластеров
            print(clust)
            if midl_dist[clust - 1] > min_dist[clust - 2]:
                return max_dist, midl_dist, min_dist, clust
            # if np.array(Dist2[clust - 1]) - np.array(MinDist2[clust - 2]) > 0:
            #     listClust = Clustering(clust, meanAmplTotal[clust - 2], 0) #функция возращает кластеризованные сигналы
            #     print(clust)
            #     print()
            #     print(MaxDist2)
            #     print()
            #     print(Dist2)
            #     print()
            #     print(MinDist2)
            #     #print KONEZ
            #     print('Output}')
            #     for e in np.arange(2, clust+1):
            #         print(np.array(Dist2[e - 1]) - np.array(MinDist2[e - 2]))
            #     print ('YesMin')
            #     #return MaxDist2, Dist2, MinDist2, KONEZ, clust
            #     return SumForGraphics, MaxDist2, Dist2, MinDist2, clust
            #
            #
            #
            # ЗДЕСЬ ИДЕТ УЛУЧШЕНИЕ АЛГОРИТМА, ЧТО БЫ НЕ ВЫЧИСЛЯТЬ СЛЕДУЮЩИЙ КЛАСТЕР
            # if np.array(MaxDist2[clust - 1]) - np.array(Dist2[clust-2]) > 0:
            #     print clust
            #     print
            #     print MaxDist2
            #     print
            #     print Dist2
            #     print
            #     print MinDist2
            #     #print KONEZ
            #     print
            #     print  np.array(MaxDist2[clust - 1]), np.array(Dist2[clust-2])
            #     print "Output"
            #     for e in np.arange(2, clust+1):
            #         print np.array(MaxDist2[e - 1]) - np.array(Dist2[e - 2])
            #     print 'YesMaxs'
            #     #return MaxDist2, Dist2, MinDist2, KONEZ, clust
            #     return SumForGraphics, MaxDist2, Dist2, MinDist2, clust
            #
            # if clust == clusters:
            #     listClust = [Clustering(x, meanAmplTotal[x-1], 0) for x in np.arange(1,clusters+1)] #функция возращает кластеризованные сигналы
            #     print(clust)
            #     print()
            #     print(Dist2)
            #     print()
            #     print()
            #     print(MaxDist2)
            #     print()
            #     print(MinDist2)
            #     print()
            #     for e in np.arange(1, clust+1):
            #         print(np.array(Dist2[e - 1]) - np.array(MinDist2[e - 2]))
            #     return MaxDist2, Dist2, MinDist2, KONEZ, clust


error = 50
clusters_max = 50
num_iter_for_montecarlo = 1000
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
#plt.show()
# print listClst[0], listClst[1], listClst[2]
#
# plt.plot(o, listClst[0])
plt.ylabel(u'СУММА РАССТОЯНИЙ')
plt.xlabel(u'КОЛИЧЕСТВО КЛАСТЕРОВ')
# fig = plt.figure(1, figsize=(9, 6))
# # Create an axes instance
# ax = fig.add_subplot(111)
#plt.boxplot(listClst[0])
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid()
plt.show()

