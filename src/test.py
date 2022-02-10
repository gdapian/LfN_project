import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import utils
import motifs

data_folder_path = '../dataset/'
dataset = [	'bezerra-et-al-2009_MOD.xls', 
			'olesen_aigrettes_MOD.xls',
			'olesen_flores_MOD.xls']

data_index = 0
path = data_folder_path + dataset[data_index]

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
#                          Motifs                            #
##############################################################
'''
# define the maximum size of the a possible graphlet. The minimum size of the graphlet is setted to 3 by default
max_k = 6

total_graphlets = []
total_z_score = []
total_p_value = []

for k in range(3, max_k+1):
	# compute the counting of all subgraphs (#nodes = k) of graph G 
	print("Execute ESU algorithm on the real network. Value k = " + str(k))
	graphlets, counts = motifs.ESU_bipartite_version(G, k)

	for i in range(len(graphlets)):
		#pol_current_temp, pla_current_temp = nx.algorithms.bipartite.sets(graphlets[i])
		#utils.plot_bipartite_graph(graphlets[i], list(pol_current_temp))
		print("Graphlet n°" + str(i) + " has " + str(counts[i]) + " occurrences.")

	z_score, p_value = motifs.compute_graphlets_scores(G, k, graphlets, counts, num_random_graphs=100)

	# compute the motifs
	top_graphlets, top_z_score, top_p_value = motifs.find_top_graphlets(graphlets, z_score, p_value, num=-1)

	newpath = "results/motifs/" + dataset[data_index] + "/k=" + str(k)

	# remove all the old contents from the folder
	try:
	    shutil.rmtree(newpath)
	except OSError as e:
	    print("Warning: %s - %s." % (e.filename, e.strerror))
	    print("Folder " + e.filename + " will be created.")

	# (re)create the folder
	if not os.path.exists(newpath):
	    os.makedirs(newpath)

	for j in range(len(top_graphlets)):
		# save the plot of the current graphlets
		pol_current_temp, pla_current_temp = nx.algorithms.bipartite.sets(top_graphlets[j])
		fig = utils.plot_bipartite_graph(top_graphlets[j], list(pol_current_temp), showFig = False)
		fig.savefig(newpath + "/" + str(j) +".png")
		plt.close(fig)
		# save the scores of the current graphlets
		file = open(newpath + "/" + str(j) +".txt", "w+")
		file.write("z-score: " + str(top_z_score[j]) + "\n" + "p-value: " + str(top_p_value[j]))
		file.close()

	total_graphlets = total_graphlets + top_graphlets
	total_z_score = total_z_score + top_z_score
	total_p_value = total_p_value + top_p_value

	print ("Top z-scores: " + str(top_z_score))
	print ("Top p-values: " + str(top_p_value))


top_total_graphlets, top_total_z_score, top_total_p_value = motifs.find_top_graphlets(total_graphlets, total_z_score, total_p_value, num=-1)

newpath = "results/motifs/" + dataset[data_index] + "/total"

# remove all the old contents from the folder
try:
    shutil.rmtree(newpath)
except OSError as e:
    print("Warning: %s - %s." % (e.filename, e.strerror))
    print("Folder " + e.filename + " will be created.")

# (re)create the folder
if not os.path.exists(newpath):
    os.makedirs(newpath)

for j in range(len(top_total_graphlets)):
	# save the plot of the current graphlets
	pol_current_temp, pla_current_temp = nx.algorithms.bipartite.sets(top_total_graphlets[j])
	fig = utils.plot_bipartite_graph(top_total_graphlets[j], list(pol_current_temp), showFig = False)
	fig.savefig(newpath + "/" + str(j) +".png")
	plt.close(fig)
	# save the scores of the current graphlets
	file = open(newpath + "/" + str(j) +".txt", "w+")
	file.write("z-score: " + str(top_total_z_score[j]) + "\n" + "p-value: " + str(top_total_p_value[j]))
	file.close()
'''