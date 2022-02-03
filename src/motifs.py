# https://tobigs.gitbook.io/tobigs-graph-study/chapter3./python-code-roix-and-esu-tree
def iterative_ESU(G, k):
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
      continue
    for i in range(N):
      if G[front[-1]][i]==1 and i > front[0]:
        queue.append(front+[i])
  for s in sub:
    ans += sub[s]
  return ans

#####################################################################

import numpy as np

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


