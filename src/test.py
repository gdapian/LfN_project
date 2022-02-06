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
G, pol, pla = utils.build_graph_from_xls(path, verbose=False)
#utils.plot_bipartite_graph(G, pol)

# calculate the centralities
closeness_centrality, betweenness_centrality = utils.compute_centralities(G)
#utils.plot_centrality_graph(G, pol, closeness_centrality, title='Closeness Centrality', max_node_size=1000, size=True, opacity=True)
#utils.plot_centrality_graph(G, pol, betweenness_centrality, title='Betweenness Centrality', max_node_size=1000, size=False, opacity=True)


'''
paths = utils.load_paths(data_folder_path)
Graphs, pollinators, plants = utils.build_all_graphs(paths)
for i in range(len(Graphs)):
	utils.plot_bipartite_graph(Graphs[i], pollinators[i])

'''

ll = utils.top_K_nodes(closeness_centrality, 10)
print(ll)

''' to fix, infinite loop with esu algorithm
##############################################################
# generate the adjacency matrix
G_adj = nx.to_numpy_array(G)
for i in range(G_adj.shape[0]):
	for j in range (G_adj.shape[1]):
		if G_adj[i][j]:
			G_adj[i][j] = 1


#np.savetxt('adj.txt', pd.DataFrame(G_adj).values, fmt='%d') # to check if G_adj is actually an adjacency matrix
k = 2
print("ESU:")
subgraphs = motifs.EnumerateSubgraphs(G_adj, k)
#subgraphs = motifs.iterative_ESU(G_adj, k)


# plot a graphlet
index = 0
bi_partition_index = 12 # maximum node of the first partition of the bipartite graph

g = nx.Graph()
g.add_nodes_from(subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)], bipartite=0)
g.add_nodes_from(subgraphs[index][np.where(subgraphs[index]>bi_partition_index)], bipartite=1)
for i in range(k):
	for j in range(k):
		if (i<j) and (G_adj[subgraphs[index][i]][subgraphs[index][j]] == 1):
			g.add_edge(subgraphs[index][i], subgraphs[index][j])
utils.plot_bipartite_graph(g, subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)])
'''