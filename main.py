import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def createPlots(num, graph):
    if graph != set():
        nx.draw(graph)
        # 4.1 Degree centrality plot
        plt.hist(list(nx.degree_centrality(graph).values()))
        plt.title('Degree Centrality')
        plt.savefig('Degree%dist' % num)
        plt.clf()
        # 4.2 Degree centrality plot
        plt.hist(list(nx.in_degree_centrality(graph).values()))
        plt.title('In Degree Centrality')
        plt.savefig('InDegree%dist' % num)
        plt.clf()
        # 4.3 Degree centrality plot
        plt.hist(list(nx.out_degree_centrality(graph).values()))
        plt.title('Out Degree Centrality')
        plt.savefig('OutDegree%dist' % num)
        plt.clf()
        # 4.4 Degree centrality plot
        plt.hist(list(nx.closeness_centrality(graph).values()))
        plt.title('Closeness Centrality')
        plt.savefig('Closeness%dist' % num)
        plt.clf()
        # 4.5 Degree centrality plot
        plt.hist(list(nx.betweenness_centrality(graph).values()))
        plt.title('Betweeness Centrality')
        plt.savefig('Betweeness%dist' % num)
        plt.clf()
        # 4.6 Degree centrality plot
        plt.hist(list(nx.eigenvector_centrality_numpy(graph).values()))
        plt.title('Eigenvector Centrality')
        plt.savefig('Eigenvector%dist' % num)
        plt.clf()
        # 4.7 Degree centrality plot
        plt.hist(list(nx.katz_centrality(graph).values()))
        plt.title('Katz Centrality')
        plt.savefig('Katz%dist' % j)
        plt.clf()
        print("All centralities have been plotted and saved !!")


def e_star__a_calc(N, nodes):
    e_star = []
    for idx in range(1, N):
        try:
            e_star.append(set(nodes[idx - 1]).intersection(nodes[idx]))  # find the common values
        except:
            break

    return e_star


def e_star__b_calc(N, nodes):
    e_star = []
    for idx in range(0, N - 1):
        try:
            e_star.append(set(nodes[idx]).intersection(nodes[idx + 1]))  # find the common values
        except:
            break

    return e_star


def v_star_calc(N, edges):
    v_star = []
    for idx in range(1, N):
        try:
            v_star.append(set(edges[idx - 1]).intersection(edges[idx]))  # find the common values
        except:
            break

    return v_star


def similarities_matrices_calc(graphs):
    for idx in range(len(graphs)):

        nodes = list(graphs[idx].nodes)
        GD = {}
        CN = {}
        G = graphs[idx].to_undirected()  # graph must be undirected in order for functions to work

        for first_node in nodes:
            for second_node in nodes:
                ## 6.1 find the common neighbors of nodes
                neighbors = []
                temp_neighbors = nx.common_neighbors(G, first_node, second_node)
                for p in temp_neighbors:
                    neighbors.append(p)
                CN[first_node, second_node] = len(neighbors)

                # 6.2 find the graph distance
                try:
                    distance = nx.shortest_path_length(G, first_node, second_node)
                    GD[first_node, second_node] = distance
                except:
                    continue

        # 6.3 find the jaccard coefficient
        jaccard = nx.jaccard_coefficient(G)

        # 6.4 find the adamic adar
        adamic = nx.adamic_adar_index(G)

        # 6.5 find the preferential attachment
        preferential = nx.preferential_attachment(G)

    return CN, GD, jaccard, adamic, preferential


def predict_similarity(CN, GD, jaccard, adamic, preferential, eStar):
    jac_co, dist, pref_attach = {}, {}, {}
    for u, v, p in jaccard:
        jac_co[u, v] = p
    for u, v, p in adamic:
        dist[u, v] = p
    for u, v, p in preferential:
        pref_attach[u, v] = p

    distance = GD
    neighbors = CN

    # TODO: SINEXEIA APO EDO gia erotima 7 it just needs the prediction function applied to each matrix

    return


# Load data
filepath = './sx-stackoverflow.txt'

print('== StackOverflow Dataset ============================')
print('Reading first 1000 rows...')
df = pd.read_csv(filepath, sep=' ', header=None, nrows=1000)
df.columns = ['source_id', 'target_id', 'timestamp']

# 1st part
print('\n-- 1st part --------------------------------------')
t_min = df['timestamp'].min()
t_max = df['timestamp'].max()

print(f't_min: {t_min}')
print(f't_max: {t_max}')

# 2nd part
print('\n-- 2nd part --------------------------------------')
# print('\n Enter partitioning value (it should be an integer):')
N = int(input("\n Enter partitioning value (it should be an integer): "), 10)

t_list = []
for j in range(N + 1):
    t_list.append(t_min + j * ((t_max - t_min) // N))

T_list = []
for j in range(1, N + 1):
    if (1 <= j < N):
        T_list.append([t_list[j - 1], t_list[j] - 1])
    else:
        T_list.append([t_list[j - 1], t_list[j]])

# 3rd part
print('\n-- 3rd part --------------------------------------')

# The list of time dependent graphs
graphs = [nx.DiGraph() for _ in range(len(T_list))]

# For each period create the corresponding graphs
for idx, period in enumerate(T_list):
    min = period[0]
    max = period[1]

    for _, row in df.iterrows():
        source_id = row['source_id']
        target_id = row['target_id']
        timestamp = row['timestamp']

        if (timestamp >= min and timestamp <= max):
            graphs[idx].add_node(source_id)
            graphs[idx].add_node(target_id)
            graphs[idx].add_edge(source_id, target_id)

print('\n Sub-graphs have been created!')

# 4th part
print('\n-- 4th part --------------------------------------')
for idx in range(len(graphs)):
    graphs[idx].remove_edges_from(graphs[idx].selfloop_edges())

    if graphs[idx] != nx.empty_graph:
        createPlots(idx, graphs[idx])

# 5th part
nodes = []
edges = []
print('\n-- 5th part --------------------------------------')
for idx in range(len(graphs)):
    nodes.append(graphs[idx].nodes)
    edges.append(graphs[idx].edges)

v_star = v_star_calc(N, nodes)
e_star__a = e_star__a_calc(N, edges)
e_star__b = e_star__b_calc(N, edges)

# 6th && 7th part
print('\n-- 6th and 7th part --------------------------------------')
for idx in range(N - 1):
    print(idx)
    if e_star__a[idx] != set():
        # 6th part
        CN, GD, jaccard, adamic, preferential = similarities_matrices_calc(graphs)
        # 7th part
        # TODO: SINEXEIA APO EDO gia erotima 7
        predict_similarity(CN, GD, jaccard, adamic, preferential, e_star__a[idx])
