import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle
filename = "dow_jones_index.data"

indicesToRemove = [0,2, 3,4,5,6,7,10,11,12]#用第9，10，14，15，16项数据进行聚类


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
#清洗数据，去除$符号与缺少项的数据
def clean(X):
	result = []
	for x in X:
		result.append(x)

	result = np.array(result)
	result = np.delete(result, indicesToRemove, 1)
	lastColumn = result[:,-1]
	finalResult = []
	for row in result:
		#print(row)
		rows=[]
		for i in row:
			if i == '' :

				break
			else:
				rows.append(i)
		if len(rows)>2:

			finalResult.append(rows)

	#print(len(finalResult))


	badIndices = []

	for index, dataPoint in enumerate(result):		
		temp = []
		isValidDataPoint = True
		for feature in dataPoint:
			if is_number(feature.replace("$", "")):
				temp.append(float(feature.replace("$", "")))
			else:
				isValidDataPoint = False

		if isValidDataPoint:
			finalResult.append(temp)
		else:
			badIndices.append(index)

	#print(finalResult)

	return finalResult




def loadData(filename):
	lines = [line.rstrip('\n') for line in open(filename)]
	X = []
	for line in lines:
		X.append([x.strip() for x in line.split(',')])
	return clean(X[1:])


#依据每支股票聚类得到的那个种类别最多的，进行类别确定
def classfy(stocks,stocksname,n):
	name=[]
	for x in stocksname:

		if name==[]:
			name.append(x)
		else:
			if x in name:
				continue
			else:
				name.append(x)
	#print(name)
	count=np.zeros((30,n))
	index = 0
	for n in name:

		for row in stocks:
			if row[0]==n:

				count[index][row[1]]=count[index][row[1]]+1
			else:
				continue
		index=index+1
	#print(count)
	#print(len(count))
	stocksclass=[]
	i=0
	for row in count:
		cl=0
		if row[1]>row[0]:
			if row[2]>row[1]:
				cl=2
			else:
				cl =1
		else:
			if row[2]>row[0]:
				cl=2
		nn = []
		nn.append(name[i])
		i=i+1
		nn.append(cl)
		stocksclass.append(nn)
	#print(stocksclass)
	return stocksclass







def kmeans():
	X = loadData(filename)
	stocksname = [];
	for row in X:
		stocksname.append(row[0])
	# print(stocksname)
	newX = np.delete(X, [0], 1)  # 将股票名去除
	# print(newX.shape)
	n=3#分为n类
	PCAX = PCA(n_components=3).fit_transform(newX)

	kmeans = KMeans(n_clusters=n, random_state=0).fit(PCAX)
	#print(kmeans.labels_)
	stocks = []
	for i in range(len(X)):
		a = []
		a.append(stocksname[i])
		a.append(kmeans.labels_[i])
		stocks.append(a)
	#print(stocks)
	print(classfy(stocks, stocksname,n))
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
	plt.title('K-means')
	plt.show()




def Dbscan():
	X = loadData(filename)
	stocksname = [];
	for row in X:
		stocksname.append(row[0])
	# print(stocksname)
	newX = np.delete(X, [0], 1)  # 将股票名去除
	# print(newX.shape)
	#print(newX)
	PCAX = PCA(n_components=3).fit_transform(newX)
	#print(PCAX)

	db = DBSCAN(eps=6.5, min_samples=10).fit(PCAX)

	labels = db.labels_
	print(labels)

	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)
	print(len(labels))
	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)

	stocks = []
	for i in range(len(X)):
		if labels[i]!=-1:
			a = []
			a.append(stocksname[i])
			a.append(labels[i])
			stocks.append(a)
		else:
			continue
	#print(stocks)
	print(classfy(stocks, stocksname, n_clusters_))

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
	X = loadData(filename)
	stocksname = [];
	for row in X:
		stocksname.append(row[0])
	newX = np.delete(X, [0], 1)  # 将股票名去除

	PCAX = PCA(n_components=3).fit_transform(newX)
	#print(PCAX)

	bandwidths = estimate_bandwidth(PCAX, quantile=0.5)
	print(bandwidths)
	cl = MeanShift(bandwidth=bandwidths).fit(PCAX)

	labels = cl.labels_
	print(labels)

	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)
	print(len(labels))
	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)

	stocks = []
	for i in range(len(X)):
		if labels[i] != -1:
			a = []
			a.append(stocksname[i])
			a.append(labels[i])
			stocks.append(a)
		else:
			continue
	# print(stocks)
	print(classfy(stocks, stocksname, n_clusters_))

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

if __name__ == "__main__":

	kmeans()
	#Dbscan()
	#meanshift()


