import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import utils

data_folder_path = '../dataset/'
dataset = 'bezerra-et-al-2009_MOD.xls'
path = data_folder_path + dataset

# plot a bipartite graph
G, pol, pla = utils.build_graph_from_xls(path, verbose=True)
utils.plot_bipartite_graph(G, pol)

# calculate the centralities
cc = nx.closeness_centrality(G, distance=None)
print("\nCloseness centrality:")
print(cc)
bc = nx.betweenness_centrality(G, weight=None)
print("\nBetwenness centrality:")
print(bc)
