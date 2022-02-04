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
  counts = []
  degrees = []

  index = 0
  g = nx.Graph()
  g.add_nodes_from(subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)], bipartite=0)
  g.add_nodes_from(subgraphs[index][np.where(subgraphs[index]>bi_partition_index)], bipartite=1)
  for i in range(k):
    for j in range(k):
      if (i<j) and (G_adj[subgraphs[index][i]][subgraphs[index][j]] == 1):
        g.add_edge(subgraphs[index][i], subgraphs[index][j])

  graphlets.append(g)
  counts.append(1)
 
  deg_pol = []
  for j in subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)]:
    deg_pol.append(g.degree()[j])
  deg_pla = []
  for j in subgraphs[index][np.where(subgraphs[index]>bi_partition_index)]:
    deg_pla.append(g.degree()[j])
  degrees.append([set(deg_pol), set(deg_pla)])
  # "degrees" represent, for each graphlet, the set of degrees (of every node) with the respect of pollinators and plants. it is necessary to correct the problem of isomorphic graphs in case of bipartite graphs

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
      
      deg_pol = []
      for j in subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)]:
        deg_pol.append(g_current.degree()[j])
      deg_pla = []
      for j in subgraphs[index][np.where(subgraphs[index]>bi_partition_index)]:
        deg_pla.append(g_current.degree()[j])
      deg_current = [set(deg_pol), set(deg_pla)]
      
      if(GM.is_isomorphic() and (deg_current == degrees[i])):
        flag = False
        counts[i] = counts[i]+1
        break

    if flag:
      graphlets.append(g_current)
      counts.append(1)
      degrees.append(deg_current)

  return graphlets, counts


def ESU_bipartite_version(G, k, verbose=False): 

  pol_temp, pla_temp = nx.algorithms.bipartite.sets(G)
  pol = list(pol_temp)

  # generate the adjacency matrix
  G_adj = utils.GraphToAdjacencyMatrix(G)
  #np.savetxt('adj.txt', pd.DataFrame(G_adj).values, fmt='%d') # to check if G_adj is actually an adjacency matrix

  # ESU Algorithm - First Phase
  print("ESU Algorithm First Phase starts...")
  subgraphs = EnumerateSubgraphs(G_adj, k, verbose)

  # ESU Algorithm - Second Phase
  print("ESU Algorithm Second Phase starts...")
  graphlets, counts = ESU_second_phase(G_adj, k, subgraphs, pol)

  return graphlets, counts
