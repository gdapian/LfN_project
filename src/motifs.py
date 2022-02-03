import numpy as np

# https://tobigs.gitbook.io/tobigs-graph-study/chapter3./python-code-roix-and-esu-tree
def iterative_ESU(G, k):

  subgraphs = []

  N = len(G)
  visited = [False] * N
  queue = []
  ans = 0
  sub = {}
  sub_graph = {}
  for i in range(N):
    queue.append([i])
  while queue:
    front = queue[0]
    queue = queue[1:]
    if len(front)==k:
      degree = [0] * N
      for f in front:
        for j in front:
          if G[f][j]==1:
            degree[f] += 1
            degree[j] += 1
      if len(set(front)) != len(front):
        continue
      if str(sorted(front)) in sub_graph:
        continue
      sub_graph[str(sorted(front))] = 1
      if str(sorted(degree)) in sub:
        sub[str(sorted(degree))] += 1 
      else:
        sub[str(sorted(degree))] = 1
      print(front)
      subgraphs.append(np.array(front))
      continue
    for i in range(N):
      if G[front[-1]][i]==1 and i > front[0]:
        queue.append(front+[i])
  for s in sub:
    ans += sub[s]
  return ans, subgraphs

#####################################################################

import numpy as np
import networkx as nx
import utils

subgraphs = []

def EnumerateSubgraphs(G, k, verbose = False):
    n = len(G)

    for v in range(n):
      V_ext = []
      for u in range(n):
        if (u > v) and (G[v][u] == 1): 
          V_ext.append(u)
      ExtendSubgraph([v], V_ext, v, G, k, verbose)

    return subgraphs

def ExtendSubgraph(V_sub, V_ext, v, G, k, verbose):
  n = len(G)

  if len(V_sub) == k:
    if verbose:
      print("V_sub (len == k) : " + str(V_sub))
    subgraphs.append(V_sub) 
    return

  while len(V_ext) != 0:
    w = V_ext.pop(0)
    
    N_w = []
    for i in range(n):
      if (G[w][i] == 1): 
        N_w.append(i)

    N_V_sub = []
    for i in V_sub:
      for j in range(n):
        if (G[i][j] == 1) and (j not in N_V_sub): 
          N_V_sub.append(j)

    N_excl = np.setdiff1d(N_w, (np.union1d(V_sub, N_V_sub)))

    V_ext_prime = V_ext.copy()
    for u in N_excl:
      if (u > v) and (u not in V_ext_prime):
        V_ext_prime.append(u)

    '''
    print("v : " + str(v))
    print("w : " + str(w))
    print("V_sub : " + str(V_sub))
    print("N_w : " + str(N_w))
    print("N_V_sub : " + str(N_V_sub))
    print("N_excl : " + str(N_excl))
    print("V_ext : " + str(V_ext))
    print("V_ext_prime : " + str(V_ext_prime))
    print("---------")
    '''
    ExtendSubgraph(np.union1d(V_sub, w), V_ext_prime, v, G, k, verbose)
  return


def ESU_second_phase(G_adj, k, subgraphs, pol):
  bi_partition_index = len(pol)-1 # maximum node of the first partition of the bipartite graph

  graphlets = []
  pols = []
  plas = []
  counts = []
  ratios = []

  index = 0
  g = nx.Graph()
  g.add_nodes_from(subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)], bipartite=0)
  g.add_nodes_from(subgraphs[index][np.where(subgraphs[index]>bi_partition_index)], bipartite=1)
  for i in range(k):
    for j in range(k):
      if (i<j) and (G_adj[subgraphs[index][i]][subgraphs[index][j]] == 1):
        g.add_edge(subgraphs[index][i], subgraphs[index][j])

  graphlets.append(g)
  pols.append(subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)])
  plas.append(subgraphs[index][np.where(subgraphs[index]>bi_partition_index)])
  counts.append(1)
  ratios.append(len(subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)]) / len(subgraphs[index][np.where(subgraphs[index]>bi_partition_index)]))
  #this is the ratio between the nodes of one class and those of the other. it is necessary to correct the problem of isomorphic graphs in case of bipartite graphs

  for index in range(1,len(subgraphs)): # skip the first, already added above
    g_current = nx.Graph()
    g_current.add_nodes_from(subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)], bipartite=0)
    g_current.add_nodes_from(subgraphs[index][np.where(subgraphs[index]>bi_partition_index)], bipartite=1)
    for i in range(k):
      for j in range(k):
        if (i<j) and (G_adj[subgraphs[index][i]][subgraphs[index][j]] == 1):
          g_current.add_edge(subgraphs[index][i], subgraphs[index][j])
    flag = True
    for i in range(len(graphlets)):
      GM = nx.algorithms.isomorphism.GraphMatcher(g_current, graphlets[i])
      ratio_current = len(subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)]) / len(subgraphs[index][np.where(subgraphs[index]>bi_partition_index)])
      if(GM.is_isomorphic() and (ratio_current == ratios[i])):
        flag = False
        counts[i] = counts[i]+1
        break
    if flag:
      graphlets.append(g_current)
      pols.append(subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)])
      plas.append(subgraphs[index][np.where(subgraphs[index]>bi_partition_index)])
      counts.append(1)
      ratios.append(ratio_current)

  return graphlets, pols, plas, counts