import pandas as pd
import networkx as nx
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
N = input('Enter N parameter: ')
N = int(N)
DT = t_max - t_min
dt = DT // N
print(f'DT: {DT}')
print(f'dt: {dt}')

t_list = []
for j in range(N):
    t_list.append(t_min + j * dt)

print()
print(t_list)

# 3o erotima
print()
print('3o erotima')
print()

