import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import utils
import motifs

data_folder_path = '../dataset/'
dataset = [	'bezerra-et-al-2009_MOD.xls', 
			'olesen_aigrettes_MOD.xls',
			'olesen_flores_MOD.xls']
path = data_folder_path + dataset[0]

# plot a bipartite graph
G, pol, pla = utils.build_graph_from_xls(path, verbose=False)
#utils.plot_bipartite_graph(G, pol)

# calculate the centralities
# closeness_centrality, betweenness_centrality = utils.compute_centralities(G)
# utils.plot_centrality_graph(G, pol, closeness_centrality, title='Closeness Centrality', max_node_size=1000, size=True, opacity=True)
# utils.plot_centrality_graph(G, pol, betweenness_centrality, title='Betweenness Centrality', max_node_size=1000, size=False, opacity=True)


'''
paths = utils.load_paths(data_folder_path)
Graphs, pollinators, plants = utils.build_all_graphs(paths)
for i in range(len(Graphs)):
	utils.plot_bipartite_graph(Graphs[i], pollinators[i])

'''
dc = utils.degree_centrality(G)
cc, bc = utils.compute_centralities(G)

centralities = [dc, cc, bc]

df = utils.top_K_nodes_df(G, centralities, ['degree centrality', 'closeness centrality', 'betweenness centrality'], K=5, all_nodes=False, only_pollinators=True, only_plants=False)
print(df)


##############################################################
# compute the counting of all subgraphs (#nodes = k) of graph G 
k = 5
print("Execute ESU algorithm on the real network.")
graphlets, counts = motifs.ESU_bipartite_version(G, k)

for i in range(len(graphlets)):
	#pol_current_temp, pla_current_temp = nx.algorithms.bipartite.sets(graphlets[i])
	#pol_current = list(pol_current_temp)
	#utils.plot_bipartite_graph(graphlets[i], pol_current)
	print("Graphlet n°" + str(i) + " has " + str(counts[i]) + " occurrences.")

num_random_graphs = 6
z_score, p_value = motifs.compute_graphlets_scores(G, k, graphlets, counts, num_random_graphs)

print ("z-score: " + str(z_score))
print ("p-value: " + str(p_value))