import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import birankpy

RED = [.5, 0, 0]
GREEN = [0, .5, 0]
BLUE = [0, .5, .5]


def load_paths(data_folder):
    """
    Loads all paths in data_folder
    """
    paths = []
    files = os.listdir(data_folder)
    for file_name in files:
        path = os.path.join(data_folder, file_name)
        paths.append(path)
    return paths, files


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


def build_all_graphs(paths):
    """
    Builds all graphs
    """
    Graphs = []
    pollinators = []
    plants = []
    for path in paths:
        G, pol, pla = build_graph_from_xls(path, verbose=False)
        Graphs.append(G)
        pollinators.append(pol)
        plants.append(pla)
    return Graphs, pollinators, plants


def plot_bipartite_graph(G: nx.Graph(), pollinators, node_colours=[RED, GREEN], figure_size=(20,15), showFig = True, title='Plan-pollinator network'):
    """
    Plots the bipartite graph
    """
    mapping = {0: node_colours[0], 1: node_colours[1]}
    nodes = G.nodes
    colours = [mapping[nodes[n]['bipartite']] for n in nodes]
    f = plt.figure(figsize=figure_size)
    plt.title(title)
    pos = nx.drawing.layout.bipartite_layout(G, pollinators)
    thickness = nx.get_edge_attributes(G, 'weight')
    thickness = {key : value / max(thickness.values()) for key, value in zip(thickness.keys(), thickness.values())}

    nx.draw_networkx_nodes(G,
                           pos=pos,
                           nodelist=nodes,
                           node_size=2000,
                           node_color=colours,
                           alpha=0.7,
                           node_shape='s')

    if (any(thickness)):
        nx.draw_networkx_edges(G,
                               pos=pos,
                               edgelist = thickness.keys(),
                               width=list(thickness.values()),
                               edge_color='black')
    else:
        nx.draw_networkx_edges(G,
                               pos=pos,
                               width=5,
                               edge_color='black')
    nx.draw_networkx_edges(G,
                           pos=pos,
                           edgelist=thickness.keys(),
                           width=list(thickness.values()),
                           edge_color='black')

    nx.draw_networkx_labels(G, 
                            pos=pos,
                            labels=dict(zip(nodes,nodes)),
                            font_size=15)

    if showFig:
        plt.box(False)
        plt.show()
    return f
    
    
def compute_n_radom_graphs(num_rand_graphs, node_first_set, node_last_set, edge_number):
    """
    Computes n random graphs following  
    Erdős-Renyi algorithm
    """
    random_graphs = []

    i=0
    while (i < num_rand_graphs):
        G = nx.algorithms.bipartite.generators.gnmk_random_graph(node_first_set,node_last_set,edge_number)
        if (nx.is_connected(G)):
            random_graphs.append(G)
            i=i+1

    return random_graphs


def compute_centralities(G: nx.Graph(), dist=None, w=None):
    """
    Computes closeness and betweenness centralities for a graph G
    """
    cc = nx.closeness_centrality(G, distance=dist)
    bc = nx.betweenness_centrality(G, weight=w)
    return cc, bc


def degree_centrality(G):
    """
    Returns a dictionary containing nodes of G and their degree
    """
    return {node: G.degree(node) for node in G.nodes()}


def top_K_nodes(centrality, K, all_nodes=False):
    """
    Given a dict with centrality score for each node, returns top K nodes with highest score
    """
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True )
    if K >= len(centrality) or all_nodes==True:
        return sorted_centrality
    return sorted_centrality[0:K]


def plot_centrality_graph(G: nx.Graph(), pollinators, centrality: dict, node_colours=[RED, GREEN], figure_size=(20,15), max_node_size=1500, size=True, opacity=True, title='plot'):
    """
    Plots the bipartite graph, where the nodes' alpha and size can depend on their centrality value
    if the opacity=True and size=True
    """
    mapping = {0: node_colours[0], 1: node_colours[1]}
    nodes = G.nodes()
    if opacity == True:
        colours = [tuple(mapping[nodes[n]['bipartite']] + [centrality[n]]) for n in nodes]
    else:
        colours = [tuple(mapping[nodes[n]['bipartite']]) for n in nodes]
    if size == True:
        sizes = [max_node_size * centrality[n] for n in nodes]
    else:
        sizes = [max_node_size for n in nodes]
    plt.figure(figsize=figure_size)
    plt.title(title)

    plt.title(title)
    pos = nx.drawing.layout.bipartite_layout(G, pollinators)
    thickness = nx.get_edge_attributes(G, 'weight')
    thickness = {key : value / max(thickness.values()) for key, value in zip(thickness.keys(), thickness.values())}

    nx.draw_networkx_nodes(G,
                           pos=pos,
                           nodelist=nodes,
                           node_size=sizes,
                           node_color=colours,
                           node_shape='s')

    nx.draw_networkx_edges(G,
                           pos=pos,
                           edgelist = thickness.keys(),
                           width=list(thickness.values()),
                           edge_color='black')

    nx.draw_networkx_labels(G, 
                            pos=pos,
                            labels=dict(zip(nodes,nodes)),
                            font_size=15)
    
    plt.box(False)
    plt.show()


def GraphToAdjacencyMatrix(G):
    G_adj = nx.to_numpy_array(G)
    for i in range(G_adj.shape[0]):
        for j in range (G_adj.shape[1]):
            if G_adj[i][j]:
                G_adj[i][j] = 1
    return G_adj


def top_K_nodes_df(G, centralities, centralities_names, K, all_nodes=False, show_value=False, only_pollinators=False, only_plants=False):
    """
    Returns a pandas dataframe cointaining the top K nodes according to different centralities
    """
    K_centralities = []
    for c in centralities:
        if only_pollinators and not only_plants:
            c = {node: c[node] for node in nx.bipartite.sets(G)[0]}
        if only_plants and not only_pollinators:
            c = {node: c[node] for node in nx.bipartite.sets(G)[1]}
        Kc = top_K_nodes(c, K, all_nodes)
        if show_value == False:
            Kc = [x[0] for x in Kc]
        K_centralities.append(Kc)

    d = {c_name : c for c_name, c in zip(centralities_names, K_centralities)}
    if all_nodes or K >= len(K_centralities[0]):
        r = len(K_centralities[0])
    else:
        r = K
    df = pd.DataFrame(d, index=range(1, r+1))
    return df

def compute_cc (graph):
    return nx.algorithms.bipartite.cluster.robins_alexander_clustering(graph)
    
def compute_z_score_for_cc(graph, num_random_graphs, verbose = False):
    cc_graph = compute_cc(graph)
    top_nodes, bottom_nodes = nx.algorithms.bipartite.basic.sets(graph)
    t_len = len(top_nodes)
    b_len = len(bottom_nodes)
    random_graphs = compute_n_radom_graphs(num_random_graphs, t_len, b_len, len(graph.edges()))

    #Compute clustering coefficient for random graphs
    cc_random = []
    for G in random_graphs:
        cc_random.append(compute_cc(G))
    mean = np.mean(cc_random)
    std = np.std(cc_random)
    if verbose:
        print(f"The clustering coefficient for the graph is: {cc_graph}")
        print(f"The mean = {mean}, std = {std}")
    z_score = (cc_graph - mean)/std

    return z_score

def create_edgelist_dataframe(G):
    edges = G.edges()
    pollinators = []
    plants = []
    for i, edge in enumerate(edges):
        pollinators.append(edge[0])
        plants.append(edge[1])
    d = {'pollinators': pollinators, 'plants': plants}
    df = pd.DataFrame(data=d)
    return df

#The other normalizer is 'CoHITS'
def compute_birank_centrality(G, normalizer = 'HITS'):
    bn = birankpy.BipartiteNetwork()
    bn.set_edgelist(
        create_edgelist_dataframe(G),
        top_col='pollinators', bottom_col='plants',
        weight_col=None
    )
    #Create dataframes
    pollinators_birank_df, plants_birank_df = bn.generate_birank(normalizer=normalizer)
    return pollinators_birank_df, plants_birank_df
