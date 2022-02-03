import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

RED = [.5, 0, 0]
GREEN = [0, .5, 0]
BLUE = [0, .5, .5]

def build_graph_from_xls(path: str, verbose=False):
    """
    Builds bipartite graph from .xls
    """
    data_frame = pd.read_excel (path, header = 0)
    matrix = data_frame.to_numpy()

    Graph = nx.Graph()
    pollinators = []
    plants = []
    if verbose == True:
        print(f'\nADDING POLLINATORS\n')
    for i in range(1,len(matrix)):
        pollinators.append(matrix[i][0])
        if verbose == True:
            print(f'{matrix[i][0]} added to nodes')
    Graph.add_nodes_from(pollinators, bipartite=0)

    if verbose == True:
        print(f'\nADDING PLANTS\n')
    for i in range(1,len(matrix[0])):
        plants.append(matrix[0][i])
        if verbose == True:
            print(f'{matrix[0][i]} added to nodes')
    Graph.add_nodes_from(plants, bipartite=1)

    if verbose == True:
        print(f'\nADDING EDGES\n')
    for i in range (1,len(matrix)):
        for j in range (1,len(matrix[0])):
            if matrix[i][j]:
                Graph.add_edge(matrix[i][0], matrix[0][j], weight=matrix[i][j])
                if verbose == True:
                    print(f'Added edge {matrix[i][0]}----{matrix[i][j]}---{matrix[0][j]}')
    
    return Graph, pollinators, plants


def plot_bipartite_graph(G: nx.Graph(), pollinators, node_colours=['blue', 'green'], figure_size=(15,10)):
    """
    Plots the bipartite graph
    """
    mapping = {0: node_colours[0], 1: node_colours[1]}
    nodes = G.nodes
    colours = [mapping[nodes[n]['bipartite']] for n in nodes]
    plt.figure(figsize=figure_size)
    plt.title('Plant-pollinator network')
    nx.draw_networkx(G, pos = nx.drawing.layout.bipartite_layout(G, pollinators), node_color=colours, node_size=50, font_size=5)
    plt.show()
    
    
def compute_n_radom_graphs(num_rand_graphs, node_first_set, node_last_set, edge_number):
    """
    Computes n random graphs following  
    Erdős-Renyi algorithm
    """
    random_graphs = []
    for i in range(num_rand_graphs):
        G = nx.algorithms.bipartite.generators.gnmk_random_graph(node_first_set,node_last_set,edge_number)
        random_graphs.append(G)
    return random_graphs


def compute_centralities(G: nx.Graph(), dist=None, w=None):
    """
    Computes closeness and betweenness centralities for a graph G
    """
    cc = nx.closeness_centrality(G, distance=dist)
    bc = nx.betweenness_centrality(G, weight=w)
    return cc, bc


def plot_centrality_graph(G: nx.Graph(), pollinators, centrality: dict, node_colours=[RED, GREEN], figure_size=(15,10), max_node_size=500, size=True, opacity=True, title='plot'):
    mapping = {0: node_colours[0], 1: node_colours[1]}
    nodes = G.nodes()
    if opacity == True:
        colors = [tuple(mapping[nodes[n]['bipartite']] + [centrality[n]]) for n in nodes]
    else:
        colors = [tuple(mapping[nodes[n]['bipartite']]) for n in nodes]
    if size == True:
        sizes = [max_node_size * centrality[n] for n in nodes]
    else:
        sizes = [max_node_size for n in nodes]
    plt.figure(figsize=figure_size)
    plt.title(title)
    nx.draw_networkx(G, pos = nx.drawing.layout.bipartite_layout(G, pollinators), node_color=colors, node_size=sizes, font_size=5)
    plt.show()