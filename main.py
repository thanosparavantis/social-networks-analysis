import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def createPlots(num, graph):
    if G != set():
        nx.draw(graph)
        # 4.1 Degree centrality plot
        plt.hist(list(nx.degree_centrality(graph).values()))
        plt.title('Degree Centrality')
        plt.savefig('Degree%d' % num)
        plt.clf()
        # 4.2 Degree centrality plot
        plt.hist(list(nx.in_degree_centrality(graph).values()))
        plt.title('In Degree Centrality')
        plt.savefig('InDegree%d' % num)
        plt.clf()
        # 4.3 Degree centrality plot
        plt.hist(list(nx.out_degree_centrality(graph).values()))
        plt.title('Out Degree Centrality')
        plt.savefig('OutDegree%d' % num)
        plt.clf()
        # 4.4 Degree centrality plot
        plt.hist(list(nx.closeness_centrality(graph).values()))
        plt.title('Closeness Centrality')
        plt.savefig('Closeness%d' % num)
        plt.clf()
        # 4.5 Degree centrality plot
        plt.hist(list(nx.betweenness_centrality(graph).values()))
        plt.title('Betweeness Centrality')
        plt.savefig('Betweeness%d' % num)
        plt.clf()
        # 4.6 Degree centrality plot
        plt.hist(list(nx.eigenvector_centrality_numpy(graph).values()))
        plt.title('Eigenvector Centrality')
        plt.savefig('Eigenvector%d' % num)
        plt.clf()
        # 4.7 Degree centrality plot
        plt.hist(list(nx.katz_centrality(graph).values()))
        plt.title('Katz Centrality')
        plt.savefig('Katz%d' % j)
        plt.clf()
        print("All centralities have been plotted and saved !!")


filepath = './sx-stackoverflow.txt'

print('StackOverflow Dataset')
print(f'Reading first 1000 rows...')
df = pd.read_csv(filepath, sep=' ', header=None, nrows=1000)
df.columns = ['source_id', 'target_id', 'timestamp']

# 1o erotima
print()
print('1o erotima')
print()
t_min = df['timestamp'].min()
t_max = df['timestamp'].max()

print(f't_min: {t_min}')
print(f't_max: {t_max}')

# 2o erotima
print()
print('2o erotima')
print()

N = 5
n = N + 1

t_list = [t_min + j * (t_max - t_min) // (n - 1) for j in range(n)]

[print(f'Bound t{idx}={item}') for idx, item in enumerate(t_list)]
print()

T_list = [[t_list[j - 1], t_list[j]] for j in range(1, n)]

[print(f'Period T{idx + 1}={item}') for idx, item in enumerate(T_list)]

# 3o erotima

print()
print('3o erotima')
print()

# The list of time dependent graphs
graphs = [nx.Graph() for _ in range(len(T_list))]

num = 0
for _, row in df.iterrows():
    source_id = row['source_id']
    target_id = row['target_id']
    timestamp = row['timestamp']
    G = nx.DiGraph()  # empty graph
    for idx, period in enumerate(T_list):
        min = period[0]
        max = period[1]

        if (idx == len(T_list) - 1 and min <= timestamp <= max) or min <= timestamp < max:
            G = graphs[idx]
            G.add_node(source_id)
            G.add_node(target_id)
            G.add_edge(source_id, target_id)
            break


    # erotima 4
    print()
    print('4o erotima')
    print()
    G.remove_edges_from(G.selfloop_edges())
    if G != nx.empty_graph:
        createPlots(num, G)

    # erotima 5
    nodes = G.nodes
    edges = G.edges

    Vs = []
    Es = []
    for i in range(num):
        try:
            tmp = set(nodes[i + 1]) - set(nodes[i])
            tmp2 = set(edges[i + 1]) - set(edges[i])
            Vs.append(set(nodes[i]) - set(tmp))
            Es.append(set(nodes[i]) - set(tmp2))
            continue
        except:
            break
    print("V star")
    print(Vs)
    print()
    print("E star")
    print(Es)
    num += 1

for graph in graphs:
    nx.draw(graph, **{
        'node_size': 10,
        'width': 1,
    })
    plt.show()
