import pandas as pd
import networkx as nx
import numpy as np

filepath = './sx-stackoverflow.txt'

print('StackOverflow Dataset')
print(f'Reading first 1000 rows...')
dataset = pd.read_csv(filepath, sep=' ', header=None, nrows=1000)
dataset.columns = ['source_id', 'target_id', 'timestamp']

# 1o erotima
print()
print('1o erotima')
print()
t_min = dataset['timestamp'].min()
t_max = dataset['timestamp'].max()

print(f't_min: {t_min}')
print(f't_max: {t_max}')

# 2o erotima
print()
print('2o erotima')
print()

N = 5
t_list = [t_min + j * (t_max - t_min) // (N - 1) for j in range(N)]

print(f'N=5')
[print(f't{idx}={item}') for idx, item in enumerate(t_list)]
print()

T_list = [[t_list[j - 1], t_list[j]] for j in range(1, N)]

[print(f'T{idx}-{idx+1}={item}') for idx, item in enumerate(T_list)]

# 3o erotima
print()
print('3o erotima')
print()
