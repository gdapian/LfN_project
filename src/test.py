import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel ('bezerra-et-al-2009_MOD.xls', header = 0)
matrix = df.to_numpy()
print ("Loaded dataset: \"" + matrix[0][0] + "\"\n")

G = nx.Graph()

print("Loading pollinators ...")
pol = []
for i in range(1,len(matrix)):
	pol.append(matrix[i][0])
	print("\"" + matrix[i][0] + "\" added to nodes")
G.add_nodes_from(pol, bipartite=0)

print("\nLoading plants ...")
pla = []
for i in range(1,len(matrix[0])):
	pla.append(matrix[0][i])
	print("\"" + matrix[0][i] + "\" added to nodes")
G.add_nodes_from(pla, bipartite=1)

print("\nLoading edges ...")
for i in range (1,len(matrix)):
	for j in range (1,len(matrix[0])):
		if matrix[i][j]:
			G.add_edge(matrix[i][0], matrix[0][j], weight=matrix[i][j])
			print("\"" + str(matrix[i][0]) + "\" to \"" + str(matrix[0][j]) + "\" (weight = " + str(matrix[i][j]) + ") added to edges")

# plot a bipartite graph
nx.draw_networkx(G, pos = nx.drawing.layout.bipartite_layout(G, pol))
plt.show()


# calculate the centralities
cc = nx.closeness_centrality(G, distance=None)
print("\nCloseness centrality:")
print(cc)
bc = nx.betweenness_centrality(G, weight=None)
print("\nBetwenness centrality:")
print(bc)
