import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import warnings
from sklearn.preprocessing import normalize
import math

warnings.filterwarnings("ignore")
dataframe = pd.read_excel("./lab_iad_vars.xls", sheet_name="кластеризація")
dataframe.drop(columns=dataframe.columns[:2], axis=1, inplace=True)
inertias = []


class Point:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.neighbours = []

class Edge:
    def __init__(self, point_A, point_B):
        length = 0
        for point_A_coord, point_B_coord in zip(point_A.coordinates, point_B.coordinates):
            length = length + math.pow(point_A_coord - point_B_coord, 2)
        self.length = math.sqrt(length)
        self.points = [point_A, point_B]

class Graph:
    def __init__(self, points):
        self.points = points
        self.edges = []
        self.build_full_graph()
    def build_full_graph(self):
        for point_A in self.points:
            for point_B in self.points:
                if point_A != point_B and point_B not in point_A.neighbours:
                    point_A.neighbours.append(point_B)
                    point_B.neighbours.append(point_A)
                    self.edges.append(Edge(point_A, point_B))

def kmeans_clustering(data):
    inertias = []

    print('X1 X2 X3 X4 X5 X6 X7 X8')
    for i in range(1, data.shape[0]):
        kmeans = KMeans(n_clusters=i)
        model = kmeans.fit(data)
        predicted_clusters = model.predict(data)
        for cluster in predicted_clusters:
            print(cluster, end=" " * 2)
        print("")
        inertias.append(model.inertia_)

    plt.plot(range(1, data.shape[0]), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()
def prepare_data(data):
    prepared_data = normalize([data.iloc[:, 0].to_numpy()[1:]])
    for i in range(1, data.shape[1]):
        prepared_data = np.append(prepared_data, normalize([data.iloc[:, i].to_numpy()[1:]]), axis=0)

    return prepared_data

def initialize_points(vectors):
    points = []
    for vector in vectors:
        point = Point(vector)
        points.append(point)

    return points
def KRAB(graph):
    pass

vectors = prepare_data(dataframe)
points = initialize_points(vectors)
graph = Graph(points)
print(len(graph.edges))
#kmeans_clustering(data)

