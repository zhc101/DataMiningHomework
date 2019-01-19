import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np


data = pd.read_csv("dataset_cleanedF.csv")
datard=data["readmitted"]

data= data.drop(['readmitted','num'],axis=1)


def kmeans():
    PCAX = PCA(n_components=3).fit_transform(data)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(PCAX)

    print(kmeans.labels_)

    t = 0
    for i in range(0, len(datard)):
        if (datard[i] == kmeans.labels_[i]):
            continue
        else:
            t = t + 1
    print("acc:" + str(t / len(datard)))

    plt.figure(1)
    plt.clf()
    labels = kmeans.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    cluster_centers = kmeans.cluster_centers_

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(PCAX[my_members, 0], PCAX[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('K-eans')
    plt.show()


def Dbscan():
    PCAX = PCA(n_components=3).fit_transform(data)
    print(PCAX)

    db = DBSCAN(eps=6, min_samples=10).fit(PCAX)

    labels = db.labels_
    print(labels)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(len(labels))
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    t = 0
    for i in range(0, len(datard)):
        if (datard[i] == labels[i]):
            continue
        else:
            t = t + 1
    print("acc:" + str(t / len(datard)))

    plt.figure(1)
    plt.clf()
    labels = db.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        plt.plot(PCAX[my_members, 0], PCAX[my_members, 1], col + '.')
    plt.title('DBSCAN')
    plt.show()

def meanshift():
    PCAX = PCA(n_components=3).fit_transform(data)
    #print(PCAX)

    #bandwidths = estimate_bandwidth(PCAX, quantile=0.5)
    #print(bandwidths)
    #cl = MeanShift(bandwidth=bandwidths).fit(PCAX)

    cl = MeanShift(bandwidth=15).fit(PCAX)


    labels = cl.labels_
    print(labels)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(len(labels))
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    t = 0
    for i in range(0, len(datard)):
        if (datard[i] == labels[i]):
            continue
        else:
            t = t + 1
    print("acc:" + str(t / len(datard)))

    plt.figure(1)
    plt.clf()
    labels = cl.labels_
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

def birch():

    brc = Birch(n_clusters=3, threshold=0.5, compute_labels=True).fit(data)
    print(brc.labels_)

    t = 0
    for i in range(0, len(datard)):
        if (datard[i] == brc.labels_[i]):
            continue
        else:
            t = t + 1
    print("acc:" + str(t / len(datard)))

if __name__ == "__main__":
    kmeans()
    #Dbscan()
    #meanshift()
    #birch()



