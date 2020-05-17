'''
Demonstration program for node embedding that captures the structural similarity 
and enables generalization between different graphs. 

Author: Suzana Pratljacic
'''
import matplotlib.pyplot as plt
import click
import io_functions as io 
import numpy as np
from embedding import *
import utils 
from pathlib import Path
import dill

@click.group()
def main():
    pass

@main.command()
@click.argument('input_filename', type=click.Path(exists=True))
@click.option('--o', type=click.Path(exists=True), help="Output folder", default="output")
def draw_graph(input_filename,o):
    '''
	Load graph from an edge list and draw to output_folder/graph_example.
	'''
    o = Path(o)
    fig, ax = plt.subplots()
    graph = io.load_graph(input_filename)
    io.draw_graph(graph,ax)
    fig.savefig(o/(input_filename.split("/")[-1]+"_graph"))


@main.command()
@click.argument('input_filename', type=click.Path(exists=True))
@click.argument('obj_filename', type=click.Path(),default="obj.pickle")
@click.option('--dim', help="Dimension of embedding vector", default=2)
@click.option('--eps', help="Treshold for similarity", default=0.3)
@click.option('--neigh', help="The size of the neighborhood", default=2)
@click.option('--draw', is_flag=True, help="A flag indicating whether train embedding should be drawn")
@click.option('--o', type=click.Path(exists=True), help="Output folder", default="output")
def train (input_filename, obj_filename, draw, o, dim, eps, neigh):
    '''
	Method for learning embeddings of nodes in the training graph.
	'''
    o = Path(o)

    emb = Embedding(io.load_graph(input_filename),
        lambda i,j,G1,G2 : utils.similarity_function(i,j,G1,G2,neigh), dim, eps)
    emb.train()
    
    with open(obj_filename, 'wb') as handle:
        dill.dump(emb, handle)

    if (draw):
        fig, ax = plt.subplots()
        io.draw_2d_embedding(emb.get_train_embedding())
        fig.savefig(o/(input_filename.split("/")[-1]+"_emb_graph"))

    io.save_embedding(emb.get_train_embedding(),o/(input_filename.split("/")[-1]+"_emb"))

@main.command()
@click.argument('input_filename', type=click.Path(exists=True))
@click.argument('obj_filename', type=click.Path(exists=True),default="obj.pickle")
@click.option('--draw', is_flag=True, help="A flag indicating whether predicted embedding should be drawn")
@click.option('--o', type=click.Path(exists=True), help="Output folder", default="output")
def predict (input_filename,obj_filename,draw,o):
    ''' 
	Method for predicting embeddings of nodes in the given graph.
	'''
    o = Path(o)
    with open(obj_filename, 'rb') as handle:
        emb = dill.load(handle)

    predicted = emb.predict_all(io.load_graph(input_filename))

    if (draw):
        fig, ax = plt.subplots()
        io.draw_2d_embedding(predicted,ax)
        fig.savefig(o/(input_filename.split("/")[-1]+"_emb_graph"))

    io.save_embedding(predicted,o/(input_filename.split("/")[-1]+"_emb"))

if __name__ == '__main__':
    main()

