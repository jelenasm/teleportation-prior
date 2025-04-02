# This script creates a fully connected posterior network that should be used with a standard map equation for directed networks
# To run the code from the command line: python3 posterior_directed_network_with_metadata.py network_edgelist.dat metadata.dat $number_of_nodes
# It assumes that the input network is directed. If the input network is undirected, the input file should contain both (source, target) and (target, source) links.

import numpy as np
import sys

network_file = sys.argv[1] # input network, edgelist format (source,target,weight)
metadata_file = sys.argv[2] # metadata (node,label)
N = int(sys.argv[3]) # network size

prior_uniform = np.log(N) / N

#read metadata
nodes, labels = [], []
f_metadata = open(metadata_file, "r")
for line in f_metadata:
    node, label = line.split()
    nodes.append(int(node) - 1)
    labels.append(int(label) - 1)
f_metadata.close()        
nodes = np.array(nodes)
labels = np.array(labels)   
M = np.max(labels) # number of labels

    
#read edgelist and create adjacency matrix   
f_network = open(network_file, "r")

A = np.zeros(N * N).reshape(N, N) # adjacency matrix
W = np.zeros(N * N).reshape(N, N) # weighted network
for line in f_network:
    node_i, node_j, w = np.array(line.split()).astype(int)
    A[int(node_i)-1][int(node_j)-1] = 1
    W[int(node_i)-1][int(node_j)-1] += w
f_network.close()    

#create prior network based on metadata

s_in, k_in, mass_in, s_out, k_out, mass_out = np.zeros(N), np.zeros(N), np.ones(N), np.zeros(N), np.zeros(N), np.ones(N)
for node in range(N):
    s_out[node] = np.sum(W[node, :])    
    k_out[node] = np.sum(A[node, :])
    if k_out[node] > 0:
        mass_out[node] = np.float(s_out[node]) / k_out[node]
    s_in[node] = np.sum(W[:, node])
    k_in[node] = np.sum(A[:, node])
    if k_in[node] > 0:
        mass_in[node] = np.float(s_in[node]) / k_in[node]
norm = (np.sum(k_out) + np.sum(k_in)) / (np.sum(s_out) + np.sum(s_in))        

Aprior = np.zeros(N * N).reshape(N, N) # prior adjacency matrix
for m in np.unique(labels):
    nodes_in_m = nodes[labels == m]
    Nm = len(nodes_in_m)
    alpha = np.log(Nm) / Nm # link probability between two nodes with the same label "m"
    for node_i in nodes_in_m:
        for node_j in nodes_in_m:
            if node_i == node_j:
                continue
            Aprior[node_i][node_j] = alpha * norm * mass_out[node_i] * mass_in[node_j]  


# print posterior network
fout = open("infomap_network_metadata.net", "w")
fout.write("*Vertices " + str(N) + "\n")
for node in range(1, N + 1):
    fout.write(str(node) + ' ' + '"' + str(node) + '"\n')
fout.write("*Edges " + str(N*(N-1)) + "\n")
for node_i in range(N):
    for node_j in range(N):
        if node_i == node_j:
            continue        
        weight = W[node_i][node_j] + Aprior[node_i][node_j] + prior_uniform * norm * mass_out[node_i] * mass_in[node_j]
        fout.write(str(node_i + 1) + " " + str(node_j + 1) + " " + str(weight) + "\n")
fout.close()
