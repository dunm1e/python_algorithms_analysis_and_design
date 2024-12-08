# import numpy
# import sklearn
# from sklearn import datasets
# import numpy as np
# from sklearn.cluster import KMeans
# from matplotlib import pyplot as plt
# from itertools import cycle, islice
#
# # 生成一个图
# def makeTwoCircles(n_sample):
#     x, label = datasets.make_circles(n_sample, factor=0.5, noise=0.05)
#     return x, label
#
#
# # 几何距离
# def distance(x1, x2):
#     res = np.sqrt(np.sum((x1 - x2) ** 2))
#     return res
#
# #距离矩阵
# def distanceMatrix(X):
#     M = np.array(X)
#     S = np.zeros((len(M), len(M)))
#     for i in range(len(M)):
#         for j in range(i + 1, len(M)):
#             S[i][j] = 1.0 * distance(X[i], X[j])
#             S[j][i] = S[i][j]
#     return S
#
#
# def adjacencyMatrix(Z, k, sigma=1.0):
#     N = len(Z)
#     A = np.zeros((N, N))
#     for i in range(N):
#         dist = zip(Z[i], range(N))  # 将Z矩阵的一行内的元素进行逐个标识下标(从小到大)
#         dist_index = sorted(dist, key=lambda x: x[0])  # 根据一行的元素进行从小到大排序
#         neibours_id = [dist_index[m][1] for m in range(k + 1)]  # 挑出排序前11的元素的下标索引
#         for index in neibours_id:
#             A[i][index] = np.exp(-Z[i][index] / (2 * sigma * sigma))
#             A[index][i] = A[i][index]
#     return A
#
#
# # 计算拉普拉斯矩阵及其特征矩阵
# def laplacianMatrix(A):
#     # 计算度矩阵D
#     D = np.sum(A, axis=1)
#     # 计算拉普拉斯矩阵
#     L = np.diag(D) - A
#     # 计算标准化之后的拉普拉斯矩阵
#     squareD = np.diag(1.0 / (D ** (0.5)))
#     standardization_L = np.dot(np.dot(squareD, L), squareD)
#     # 计算标准化拉普拉斯矩阵的特征值和特征向量
#     k, f = np.linalg.eig(standardization_L)
#
#     # 将特征值进行排序
#     k = list(zip(k, range(len(k))))
#     k = sorted(k, key=lambda k: k[0])
#
#     # 按特征值排序后的索引将对应的特征向量重新组合
#     F = np.vstack(f[:, i] for (v, i) in k[:500]).T
#     return F
#
#
# # 绘图函数
# def plot(X, y_sp, y_km):
#     colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
#                                          '#f781bf', '#a65628', '#984ea3',
#                                          '#999999', '#e41a1c', '#dede00']),
#                                   int(max(y_km) + 1))))
#     plt.subplot(121)
#     # 绘制散点图
#     plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_sp])
#     plt.title("Spectral Clustering")
#     plt.subplot(122)
#     plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_km])
#     plt.title("Kmeans Clustering")
#     # plt.savefig(r"F:\DataBuilding\spectral_clustering.png")
#     plt.show()
#
#
# data, label = makeTwoCircles(500)
# print(np.shape(np.array(data)))
# print(np.array(data)[2])
# print(data[:5])
# S = distanceMatrix(data)
# W = adjacencyMatrix(S, 10)
# F = laplacianMatrix(W)
# print(F)
# # 谱聚类
# sp_Kmeans = KMeans(n_clusters=3).fit(F)
# # K-means聚类
# kmeans = KMeans(n_clusters=3).fit(data)
# print(f"sp_Kmeans_labels:{sp_Kmeans.labels_}\nkmeans_labels:{kmeans.labels_}")  # 标签
# # 根据标签来填充散点的颜色，以达到标识分类效果
# plot(data, sp_Kmeans.labels_, kmeans.labels_)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from scipy.linalg import eigh

# 生成数据集
def generate_data(n_samples=300, n_features=2, centers=3, cluster_std=1.0):
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=42)
    return X, y

# 构建相似度矩阵（高斯核函数）
def build_similarity_matrix(X, sigma=1.0):
    n_samples = X.shape[0]
    W = np.exp(-0.5 * (np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)) / sigma**2)
    np.fill_diagonal(W, 0)
    return W

# 计算度矩阵
def compute_degree_matrix(W):
    D = np.diag(np.sum(W, axis=1))
    return D

# 构建拉普拉斯矩阵
def compute_laplacian_matrix(D, W):
    L = D - W
    return L

# 特征值分解
def compute_normalized_laplacian(L, D):
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    L_normalized = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)
    return L_normalized

# 提取特征向量
def extract_eigenvectors(L_normalized, n_clusters):
    eigvals, eigvecs = eigh(L_normalized)
    return eigvecs[:, :n_clusters]

# 归一化特征向量
def normalize_eigenvectors(eigenvectors):
    norm_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1)[:, np.newaxis]
    return norm_eigenvectors

# 谱聚类主函数
def spectral_clustering(X, n_clusters=3, sigma=1.0):
    W = build_similarity_matrix(X, sigma)
    D = compute_degree_matrix(W)
    L = compute_laplacian_matrix(D, W)
    L_normalized = compute_normalized_laplacian(L, D)
    eigenvectors = extract_eigenvectors(L_normalized, n_clusters)
    norm_eigenvectors = normalize_eigenvectors(eigenvectors)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(norm_eigenvectors)
    labels = kmeans.labels_
    return labels

# 可视化聚类结果
def plot_clusters(X, labels, title='Spectral Clustering'):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

if __name__ == '__main__':
    # 生成数据集
    X, y = generate_data(n_samples=300, centers=3)

    # 执行谱聚类
    labels = spectral_clustering(X, n_clusters=3, sigma=1.0)

    # 可视化聚类结果
    plot_clusters(X, labels, title='Spectral Clustering Result')
