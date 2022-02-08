import numpy as np
import networkx as nx
import utils

def EnumerateSubgraphs(G, k, verbose = False):
    subgraphs = []
    n = len(G)

    for v in range(n):
      V_ext = []
      for u in range(n):
        if (u > v) and (G[v][u] == 1): 
          V_ext.append(u)
      ExtendSubgraph([v], V_ext, v, G, k, subgraphs, verbose)

    return subgraphs

def ExtendSubgraph(V_sub, V_ext, v, G, k, subgraphs, verbose):
  n = len(G)

  if len(V_sub) == k:
    if verbose:
      print("Added subgraph : " + str(V_sub))
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

    ExtendSubgraph(np.union1d(V_sub, w), V_ext_prime, v, G, k, subgraphs, verbose)
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
 
  deg_pol = []
  for j in subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)]:
    deg_pol.append(g.degree()[j])
  deg_pla = []
  for j in subgraphs[index][np.where(subgraphs[index]>bi_partition_index)]:
    deg_pla.append(g.degree()[j])

  graphlets.append(g)
  counts.append(1)
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

    deg_pol = []
    for j in subgraphs[index][np.where(subgraphs[index]<=bi_partition_index)]:
      deg_pol.append(g_current.degree()[j])
    deg_pla = []
    for j in subgraphs[index][np.where(subgraphs[index]>bi_partition_index)]:
      deg_pla.append(g_current.degree()[j])
    deg_current = [set(deg_pol), set(deg_pla)]

    flag = True
    for i in range(len(graphlets)):
      GM = nx.algorithms.isomorphism.GraphMatcher(g_current, graphlets[i])
      
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

  # generate the adjacency matrix
  G_adj = utils.GraphToAdjacencyMatrix(G)
  #np.savetxt('adj.txt', pd.DataFrame(G_adj).values, fmt='%d') # to check if G_adj is actually an adjacency matrix

  # ESU Algorithm - First Phase
  print("ESU Algorithm First Phase starts...")
  subgraphs = EnumerateSubgraphs(G_adj, k, verbose)

  # ESU Algorithm - Second Phase
  print("ESU Algorithm Second Phase starts...")
  pol_temp, pla_temp = nx.algorithms.bipartite.sets(G)
  pol = list(pol_temp)
  graphlets, counts = ESU_second_phase(G_adj, k, subgraphs, pol)

  return graphlets, counts

# Monte-Carlo approach
def compute_graphlets_scores(G, k, graphlets, counts, num_random_graphs):


  # generate the random graphs
  top_nodes, bottom_nodes = nx.algorithms.bipartite.basic.sets(G)
  t_len = len(top_nodes)
  b_len = len(bottom_nodes)
  random_graphs = utils.compute_n_radom_graphs(num_random_graphs, t_len, b_len, len(G.edges()))
  print("Generated " + str(num_random_graphs) + " random graphs.")

  # perform ESU on all random graphs
  graphlets_random_graphs = []
  counts_random_graphs = []
  for i in range(num_random_graphs):
    print("Execute ESU algorithm on the " + str(i) + "° random graph.")
    rg_graphlets, rg_counts = ESU_bipartite_version(random_graphs[i], k)
    graphlets_random_graphs.append(rg_graphlets)
    counts_random_graphs.append(rg_counts)

  # for each graphlet of the real network, save the number of occurrencies of it for each random graph
  counts_random_graphs_reduced = []
  for i in range(len(graphlets)):

    pol_temp_set, pla_temp_set = nx.algorithms.bipartite.sets(graphlets[i])
    pol_temp = list(pol_temp_set)
    pla_temp = list(pla_temp_set)

    deg_pol = []
    for jj in pol_temp:
      deg_pol.append(graphlets[i].degree(jj))
    deg_pla = []
    for jj in pla_temp:
      deg_pla.append(graphlets[i].degree(jj))
    real_graph_degrees = [set(deg_pol), set(deg_pla)]

    counts_random_graphs_reduced.append([])

    for j in range(len(graphlets_random_graphs)):
      for k in range(len(graphlets_random_graphs[j])):

        pol_temp_set, pla_temp_set = nx.algorithms.bipartite.sets(graphlets_random_graphs[j][k])
        pol_temp = list(pol_temp_set)
        pla_temp = list(pla_temp_set)

        deg_pol = []
        for jj in pol_temp:
          deg_pol.append(graphlets_random_graphs[j][k].degree(jj))
        deg_pla = []
        for jj in pla_temp:
          deg_pla.append(graphlets_random_graphs[j][k].degree(jj))
        random_graph_degrees = [set(deg_pol), set(deg_pla)]

        GM = nx.algorithms.isomorphism.GraphMatcher(graphlets[i], graphlets_random_graphs[j][k])
        
        if(GM.is_isomorphic() and (real_graph_degrees == random_graph_degrees)):
          counts_random_graphs_reduced[i].append(counts_random_graphs[j][k])
          break

  counts_rgr_mean = []
  counts_rgr_std = []
  p_value = []

  for i in range(len(counts_random_graphs_reduced)):
    counts_rgr_mean.append(np.mean(counts_random_graphs_reduced[i]))
    counts_rgr_std.append(np.std(counts_random_graphs_reduced[i]))
    
    # compute p-value
    k = 0
    for j in range(len(counts_random_graphs_reduced[i])):
      if counts_random_graphs_reduced[i][j]>=counts[i]:
        k = k+1 
    p = k/len(counts_random_graphs_reduced[i])
    p_value.append(p)

  # compute z-score
  z_score = []
  for i in range(len(graphlets)):
    z_score.append((counts[i]-counts_rgr_mean[i])/counts_rgr_std[i])

  return z_score, p_value