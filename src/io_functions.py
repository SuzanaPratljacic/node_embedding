import networkx as nx
import matplotlib.pyplot as plt

def load_graph(path):
    graph = nx.read_edgelist(path, nodetype=int)
    return graph

def draw_graph(graph,ax=plt.gca()):
    nx.draw(graph, pos=nx.spring_layout(graph), with_labels=True,ax=ax,
        node_size=100, node_color='r', font_size=7)

def draw_2d_embedding(embedding,ax=plt.gca()):
    plt.scatter(embedding[:,0], embedding[:,1])
    for i in range(len(embedding)):
        plt.annotate(str(i), (embedding[i,0], embedding[i,1]))

def save_embedding(embedding,path):
    with open(path,"w+") as writer:
        writer.write(str(len(embedding))+" "+str(len(embedding[0]))+"\n")
        for i in range(len(embedding)):
            writer.write(str(i)+" ")
            for j in range(len(embedding[i])):
                writer.write(str(embedding[i,j])+" ")
            writer.write("\n")
