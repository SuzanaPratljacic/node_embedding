

from __future__ import division
import numpy as np
import numpy.linalg as linalg
import scipy as sc
import scipy.optimize as sco
import itertools
import sys
from sklearn.preprocessing import normalize
import networkx as nx
from typing import Callable 

class Embedding :
    def __init__(self, train_graph: nx.Graph, similarity_function: Callable[[int,int,nx.Graph,nx.Graph],float], dim = 2, eps = 0.3):
        self.N = train_graph.number_of_nodes()
        self.train_graph = train_graph
        self.dim = dim
        self.eps = eps
        self.similarity_function = similarity_function

    def train(self):
        W = np.array([1 if (self.similarity_function(i,j,self.train_graph,self.train_graph) > self.eps) else 0
            for i in range (self.N) for j in range (self.N)]).reshape(self.N,self.N)

        D = np.diag(np.sum(W, axis=1))
        self.D_pow = sc.linalg.fractional_matrix_power(D, -0.5)

        L = np.subtract(D, W) 
        normalized_laplacian = np.dot(np.dot(self.D_pow, L), self.D_pow) 
       
        eigenValues1, eigenVectors1 = linalg.eig(normalized_laplacian) 
        idx = eigenValues1.argsort()[::-1]
        eigenValues = eigenValues1[idx]
        eigenVectors = eigenVectors1[:, idx]

        A = eigenVectors[:, self.N - self.dim - 1 : self.N - 1]  
        A = np.real(A)
        self.data_embedding = normalize(A, axis=1, norm='l2')  

    def predict(self, node: int, graph: nx.Graph):
        X_sim = np.array([1 if  (self.similarity_function(i, node, self.train_graph, graph) > self.eps) else 0 
            for i in range (self.N)])
        X_sim = np.dot(self.D_pow, X_sim)

        emb = np.zeros(self.dim)
        for i in range(self.dim):
            for j in range(self.N):
                emb[i] += X_sim[j] * self.data_embedding[j, i]

        emb = normalize(emb.reshape(1, self.dim), axis=1, norm='l2').reshape(self.dim)
        return emb

    def predict_all(self, graph: nx.Graph):
        return np.array([self.predict(i,graph) for i in range(graph.number_of_nodes())])
      
    def get_train_embedding (self):
        return self.data_embedding
