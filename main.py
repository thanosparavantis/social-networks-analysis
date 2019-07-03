import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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

for _, row in df.iterrows():
    source_id = row['source_id']
    target_id = row['target_id']
    timestamp = row['timestamp']

    for idx, period in enumerate(T_list):
        min = period[0]
        max = period[1]

        if (idx == len(T_list) - 1 and min <= timestamp <= max) or min <= timestamp < max:
            G = graphs[idx]
            G.add_node(source_id)
            G.add_node(target_id)
            G.add_edge(source_id, target_id)
            break

for graph in graphs:
    nx.draw(graph, **{
        'node_size': 10,
        'width': 1,
    })
    plt.show()
