import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import islice


def createPlots(num, graph):
    if graph != set():
        nx.draw(graph)
        plt.title("Subgraph")
        plt.savefig('subgraph%d' % num)
        plt.clf()
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


def e_star__a_calc(nodes, edges):
    e_star = []

    idx = 0
    for i in range(1, len(edges)):
        e_star.append([])

        for tuple in edges[i]:
            if (tuple[0] in nodes[i - 1]) and (tuple[1] in nodes[i - 1]):
                e_star[idx].append(tuple)  # find the common values

        e_star[idx] = list(set(e_star[idx]))
        idx += 1

    return e_star


def e_star__b_calc(nodes, edges):
    e_star = []

    idx = 0
    for i in range(0, len(edges) - 1):
        e_star.append([])

        for tuple in edges[i]:
            if (tuple[0] in nodes[i - 1]) and (tuple[1] in nodes[i - 1]):
                e_star[i].append(tuple)  # find the common values

        e_star[idx] = list(set(e_star[idx]))
        idx += 1

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
    # for idx in range(len(graphs)):

    nodes = list(graphs.nodes)
    GD = {}
    CN = {}
    G = graphs.to_undirected()  # graph must be undirected in order for functions to work

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


def predict_similarity(CN, GD, jaccard, adamic, preferential, e_star, idx):
    JA, A, PA = {}, {}, {}
    for u, v, p in jaccard:
        JA[u, v] = p
    for u, v, p in adamic:
        A[u, v] = p
    for u, v, p in preferential:
        PA[u, v] = p

    param_GD, param_CN, param_JC, param_A, param_PA = -1, -1, -1, -1, -1
    while (
            param_GD < 0 or param_GD >= 1 or param_CN < 0 or param_CN >= 1 or param_JC < 0 or param_JC >= 1 or param_A < 0 or param_A >= 1 or param_PA < 0 or param_PA >= 1):
        param_GD = float(input("Give param_GD. Must be 0 < param_GD <= 1\n"))
        param_CN = float(input("Give param_CN. Must be 0 < param_CN <= 1\n"))
        param_JC = float(input("Give param_JC. Must be 0 < param_JC <= 1\n"))
        param_A = float(input("Give param_A. Must be 0 < param_A <= 1\n"))
        param_PA = float(input("Give param_PA. Must be 0 < param_PA <= 1\n"))

    print(f'++ Result for set of node {idx} +++++++++++++++++++++++++++++++++++++')
    prediction_calc(param_GD, GD, e_star, 'Graph Distance')
    prediction_calc(param_CN, CN, e_star, 'Common Neighbors')
    prediction_calc(param_JC, JA, e_star, 'Jaccard Coefficient')
    prediction_calc(param_A, A, e_star, 'Adamic Adar')
    prediction_calc(param_PA, PA, e_star, 'Preferential Attachment')

    return


def prediction_calc(param, matrix, e_star, name):
    matrix = {key: value for key, value in matrix.items() if value != 0}
    matrix = sorted(matrix.items(), key=lambda kv: kv[1])

    temp = int(round(param * len(matrix)))
    temp_values, temp_total_elements = list(islice(matrix, temp, None)), len(matrix) - temp
    temp_count = 0
    temp_values = [x[0] for x in temp_values]

    for i in range(temp_total_elements):
        if (temp_values[i]) in (list(e_star)):
            temp_count += 1

    try:
        temp_success_rate = (temp_count / float(temp_total_elements)) * 100
        print("The success rate for", name, "is: ", temp_success_rate, "%")

    except:
        print("Cannot divide with 0!!!")


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
N = int(input("\n Enter partitioning value (it must be an integer): "), 10)

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
e_star__a = e_star__a_calc(v_star, edges)
e_star__b = e_star__b_calc(v_star, edges)

# 6th && 7th part
print('\n-- 6th and 7th part --------------------------------------')
for idx in range(N - 1):
    if e_star__a[idx] != set():
        # 6th part
        CN, GD, jaccard, adamic, preferential = similarities_matrices_calc(graphs[idx])

        # 7th part
        predict_similarity(CN, GD, jaccard, adamic, preferential, e_star__a[idx], idx)
