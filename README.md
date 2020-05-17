# node_embedding
A Python library for learning the embedding of nodes in graphs. Embedding is continuous, and often low dimensional, representation of discrete features. The basic purpose of embedding is as an input of highly structured data into machine learning algorithms.

This library learns the embedding of nodes in a graph using Laplacian eigenmaps. Associated embeddings of nodes are close to each other in the embedding space if the nodes have a similar graph structure around them (the degrees of the vertices in the rings around the nodes). 

This library, unlike existing methods like struc2vec and node2vec,
learns embedding on a training graph and predicts embedding (using projection on embedding space) for a node belonging to any other graph. In this way, it is possible to compare the embeddings of nodes belonging to different graphs and train some downstream machine learning algorithms that will have the ability to generalize on new graphs.

## Input
A graph in the form of an edge list. It is assumed that the graph is undirected and unweighted. 

## Output 
The output file has n+1 lines (where n is the number of nodes in the graph). The first line contains two numbers: the number of nodes and the dimension of embedding. The remaining lines of the output contain the node_id and d-dimensional embedding vector.  

## Usage
You can check the basic usage:
    
    python src/main.py --help 
