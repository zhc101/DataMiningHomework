
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import metrics
#seeds_dateset数据集


#读入数据
def import_data():
    file = os.sep.join(['seeds_dataset.txt'])
    data = pd.read_csv(file, sep="\t", header=None, names=["Area", "Perimeter", "Compactness", "Length of Kernel","Width of Kernel","Asymmetry coefficient"," Length of kernel groove","h"])
    return data

#清洗数据分开data和data_class
def cleanData(dataFrame):
    data_classes = []
    for instance in dataFrame.values:
        data_classes.append(instance[-1])
    dataFrame=dataFrame.drop(columns=['h'])
    dataVals = dataFrame.values

    data = []
    for instance in dataVals:
        data.append(instance.tolist())
    data = np.array(data)
    #print(data)
    return data, data_classes



def kmeans():
    dataFrame = import_data()
    dataList, dataClasses = cleanData(dataFrame)
    t = 0
    kmeans = KMeans(n_clusters=3, random_state=0).fit(dataList)
    #print(kmeans.labels_)


    plt.figure(1)
    plt.clf()
    labels=kmeans.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    cluster_centers = kmeans.cluster_centers_

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(dataList[my_members, 0], dataList[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('K-eans')
    plt.show()
    for i in range(0, len(kmeans.labels_)):
        kmeans.labels_[i] = kmeans.labels_[i] + 1
    for i in range(0, len(kmeans.labels_)):
        if (kmeans.labels_[i] == dataClasses[i]):
            t = t + 1
    print("acc:" + str(t / len(kmeans.labels_)))
    # print(metrics.adjusted_rand_score(dataClasses, kmeans.labels_))

def Dbscan():
    dataFrame = import_data()
    dataList, dataClasses = cleanData(dataFrame)

    db = DBSCAN(eps=0.9, min_samples=13).fit(dataList)

    labels = db.labels_
    print(labels)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(len(labels))

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    for i in range(0, len(labels)):
        labels[i] = labels[i] + 1
    t = 0
    n1=len(labels)
    for i in range(0, len(labels)):
        if (labels[i] == dataClasses[i]):
            t = t + 1
        elif(labels[i]==0):
            n1=n1-1
    print(n1)
    print("acc:" + str(t / n1))

    plt.figure(1)
    plt.clf()
    labels = db.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)


    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k

        plt.plot(dataList[my_members, 0], dataList[my_members, 1], col + '.')

    plt.title('DBSCAN')
    plt.show()

def meanshift():
    dataFrame = import_data()
    dataList, dataClasses = cleanData(dataFrame)
    t = 0
    print(dataClasses)
    #bandwidths = estimate_bandwidth(dataList, quantile=0.33)
    #print(bandwidths)
    PCAX = PCA(n_components=2).fit_transform(dataList)
    #print(PCAX)

    bandwidths = estimate_bandwidth(PCAX, quantile=0.22)
    print(bandwidths)
    cl = MeanShift(bandwidth=bandwidths).fit(PCAX)

    print(cl.labels_)
    for i in range(0, len(cl.labels_)):
        if cl.labels_[i] ==0 and dataClasses[i]==3:
            t=t+1
        else:
            if (cl.labels_[i] == dataClasses[i]):
                t = t + 1

    print("acc:" + str(t / len(cl.labels_)))
    #print(metrics.adjusted_rand_score(dataClasses, kmeans.labels_))
    plt.figure(1)
    plt.clf()
    labels=cl.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    cluster_centers = cl.cluster_centers_

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(PCAX[my_members, 0], PCAX[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Meanshift')
    plt.show()




if __name__ == "__main__":

    kmeans()
    #Dbscan()
    #meanshift()







