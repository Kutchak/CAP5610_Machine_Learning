# hw_6.py


from kmeans import loadCSV
from kmeans import kmeans
from kmeans import printTable

dataset = loadCSV('iris.data')
clustering = kmeans(dataset, 3, False)
printTable(clustering["centroids"])
print (clustering["withinss"])

