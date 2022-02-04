import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import utils
import motifs

data_folder_path = '../dataset/'
dataset = 'bezerra-et-al-2009_MOD.xls'
path = data_folder_path + dataset

# plot a bipartite graph
G, pol, pla = utils.build_graph_from_xls(path, verbose=True)
utils.plot_bipartite_graph(G, pol)

# calculate the centralities
closeness_centrality, betweenness_centrality = utils.compute_centralities(G)
utils.plot_centrality_graph(G, pol, closeness_centrality, title='Closeness Centrality', max_node_size=1000, size=True, opacity=True)
utils.plot_centrality_graph(G, pol, betweenness_centrality, title='Betweenness Centrality', max_node_size=1000, size=False, opacity=True)


##############################################################

k = 4
graphlets, counts = motifs.ESU_bipartite_version(G, k)

for i in range(len(graphlets)):
	pol_current_temp, pla_current_temp = nx.algorithms.bipartite.sets(graphlets[i])
	pol_current = list(pol_current_temp)
	utils.plot_bipartite_graph(graphlets[i], pol_current)
	print("Graphlet n°" + str(i) + " has " + str(counts[i]) + " occurrences.")


