#D-Wave Ocean imports
import dimod
import dwave_networkx as dnx
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite #Para Fazer o embendding no sistema D-wave
from dwave_qbsolv import QBSolv
import minorminer
from dwave.system.composites import FixedEmbeddingComposite

#Outros imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import itertools



#******************************************************************************
#Função auxiliar que calcula a distância entre as cidades(nodos) do problema

def get_distance(route, data):
   
    #Calcula a distância total sem o retorino a cidade de inicio
    total_dist = 0
    for idx, node in enumerate(route[:-1]):
        dist = data[route[idx + 1]][route[idx]]
        total_dist += dist

    print("Distância Total (Sem retorno):", total_dist)

    # Soma a distância entre o ponto de inicio e o ponto final do ciclo completo
    return_distance = data[route[0]][route[-1]]
    print('Distância entre o início e fim:', return_distance)

    # Pega a distância total com o retorno a cidade de inicio
    distance_with_return = total_dist + return_distance
    print("Distância Total (Com Retorno):", distance_with_return)

    return total_dist, distance_with_return

#*******************************************************************************

#[0,4,1,3,2]- solução encontrada que tem a menor distância

#Posição das cidades , dados aleatórios
pos = np.array([[12, 29],[12,  11],[6, 23],[ 1, 15],[35, 25],[25,13],[19,31],[33,12],[28,29]])
c=len(pos)


#Cria a uma Matriz 5x5 com a diagonal principal 0 e o resto como resultado da continha feita abaixo
adj = np.zeros((c,c))
for i in range(c):
    for j in range(c):
        adj[i][j] = np.sqrt((pos[i][0]-pos[j][0])**2 + (pos[i][1]-pos[j][1])**2)
        
# Cria um grafo com os pontos que representam as cidades       
G = nx.from_numpy_matrix(adj)

#print('adj={}'.format(adj))

# Coloca as características do Grafo, pontos e arestas que ligam os pontos (existem n(n-1)/2 rotas)
nodes = G.nodes()
edges = G.edges()
weights = nx.get_edge_attributes(G,'weight')

#Uso da lib matplotlib para desenhar o grafo
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
plt.axis('off')
plt.show()

#Começa o problema TSP

#Inicio com uma rota qualquer e faz o caminho no grafo
Exact_route = [0, 2, 3, 1, 4, 5, 8, 7, 6]
Exact_weights = [(Exact_route[i], (Exact_route[(i+1)%c])) for i in range(c)]
G_best = nx.from_edgelist(Exact_weights)

nx.draw_networkx_nodes(G_best, pos)
nx.draw_networkx_edges(G_best, pos)

# Com esse comando obtemos todos os coeficientes, lineares e quadráticos para um problema QUBO do TSP.
tsp_qubo = dnx.algorithms.tsp.traveling_salesperson_qubo(G)

#print(tsp_qubo)

#Here we will look for the Best Lagrange parameters. The Ocean documentation says that good values can found between 75-150%.
lagrange = None
weight='weight'

# Obtem O QUBO correspondente passo a passo
N = G.number_of_nodes()

if lagrange is None:
    # If no lagrange parameter provided, set to 'average' tour length.
    # Usually a good estimate for a lagrange parameter is between 75-150%
    # of the objective function value, so we come up with an estimate for 
    # tour length and use that.
    if G.number_of_edges()>0:
        lagrange = G.size(weight=weight)*G.number_of_nodes()/G.number_of_edges() # lagrange = N * soma_dos acoplamentos/Nc
    else:
        lagrange = 2

print('Parâmetro de Lagrange Padrão:', lagrange)

# Cria a lista de parâmetros de Lagrange aceitáveis 
lagrange_pars = list(np.arange(int(0.9*lagrange), int(1.2*lagrange), 10))
#print('Parâmetro de Lagrange para o HPO:', lagrange_pars)

# Rodamos o TSP com a rotina padrão do dwave.networkx.algorithms da D-Wave

#Faz o embendding no D-Wave 2000Q
#sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))

#simmulated annealing
sampler = dimod.SimulatedAnnealingSampler()
route = dnx.traveling_salesperson(G, sampler, start=0)
print('Route found with simulated annealing:', route)

# set parameters
num_shots = 2
start_city = 0
best_distance = sum(weights.values())
best_route = [None]*len(G)

#simullated annealing
for lagrange in lagrange_pars:
    print('Rodando o Annealing simulado para o TSP com o parâmetro de Lagrange =', lagrange)
    route = dnx.traveling_salesperson(G, sampler, lagrange=lagrange, 
                                  start=start_city,num_reads=num_shots)
    print('Rota encontrada com D-Wave:', route)
    
    # print distance
    #Usamos a função definida no inicio para calcular a distância percorrida 
    total_dist, distance_with_return = get_distance(route, adj)
    
    # Compara as rotas que têm o menor valor da distância
    if distance_with_return < best_distance:
        best_distance = distance_with_return
        best_route = route

"""#Roda o HPO(hiperparametro de otimização) para encontrar a rota
for lagrange in lagrange_pars:
    print('Rodando o Annealing Quântico para o TSP com o parâmetro de Lagrange =', lagrange)
    route = dnx.traveling_salesperson(G, sampler, lagrange=lagrange, 
                                  start=start_city, num_reads=num_shots, answer_mode="histogram")
    print('Rota encontrada com D-Wave:', route)
    
    # print distance
    #Usamos a função definida no inicio para calcular a distância percorrida 
    total_dist, distance_with_return = get_distance(route, adj)
    
    # Compara as rotas que têm o menor valor da distância
    if distance_with_return < best_distance:
        best_distance = distance_with_return
        best_route = route"""

#Mostra a melhor solução encontrada 
print('***************SOLUÇÃO FINAL**************')
print('Melhor Solução encontrada:', best_route)
print('Distância Total (com restorno):', best_distance)


#Desenha o Grafo para a melhor rota encontrada
best_weights = [(best_route[i], (best_route[(i+1)%c])) for i in range(c)]
G_bestDW = nx.from_edgelist(best_weights)

nx.draw_networkx_nodes(G_bestDW, pos)
nx.draw_networkx_edges(G_bestDW, pos)
plt.show()