import numpy as np
import collections
import networkx as nx
from typing import List
from dtw import dtw
import math

def get_rings (start_node: int, graph: nx.Graph, neigh_size) -> List[List[int]]:
    '''
    Performs BFS from the start node and returns a list of lists 
    containing nodes degrees, sorted by distance from the start node.
    '''
    q = collections.deque()
    q.appendleft((start_node, 0))

    visited = [False for i in range(graph.number_of_nodes())]
    visited[start_node] = True

    rings = []
    while len(q) > 0:
        (curr, dist) = q.pop()

        if dist>neigh_size:
            return rings

        if len(rings) <= dist:
            rings.append([])

        rings[dist].append(graph.degree(curr))

        for neigh in graph.neighbors(curr):
            if not visited[neigh]:
                visited[neigh] = True
                q.appendleft((neigh, dist+1))

    return rings

def similarity_function (i : int, j: int, G1: nx.Graph, G2: nx.Graph, neigh_size) -> float:
    '''
    Calculate similarity between node i from graph G1 and node j from graph G2.
    '''

    rings_i = [sorted(ring) for ring in get_rings(i,G1,neigh_size)]
    rings_j = [sorted(ring) for ring in get_rings(j,G2,neigh_size)]
    
    rings_i = rings_i[:min(len(rings_i), len(rings_j))]
    rings_j = rings_j[:min(len(rings_i), len(rings_j))]

    distance = [dtw(x, y, lambda x, y: np.abs(x-y))[0]/k for (k, x, y) in zip(range(1, len(rings_i)+1), rings_i, rings_j)]

    return math.exp(-sum(distance))
