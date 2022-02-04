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
# generate the adjacency matrix
G_adj = utils.GraphToAdjacencyMatrix(G)
#np.savetxt('adj.txt', pd.DataFrame(G_adj).values, fmt='%d') # to check if G_adj is actually an adjacency matrix

# ESU Algorithm - First Phase
k = 4
print("ESU Algorithm First Phase starts...")
subgraphs = motifs.EnumerateSubgraphs(G_adj, k, True)
#ans, subgraphs = motifs.iterative_ESU(G_adj, k)

# ESU Algorithm - Second Phase
# the case of isomorphic graphlets and with the same ratio of plants and pollinators is not handled correctly!!!!
print("ESU Algorithm Second Phase starts...")
graphlets, pols, plas, counts = motifs.ESU_second_phase(G_adj, k, subgraphs, pol)

for i in range(len(graphlets)):
	utils.plot_bipartite_graph(graphlets[i], pols[i])
	print("Graphlet n°" + str(i) + " has " + str(counts[i]) + " occurrences.")
