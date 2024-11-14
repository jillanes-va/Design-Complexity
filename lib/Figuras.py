import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform

import lib.Tratamiento as trat

plt.rcParams['font.family'] = 'STIXGeneral'


def graf(X, xlabel = '', ylabel = '', save = False, name = '', title = ''):
    plt.figure(figsize = (7,7))
    plt.imshow(X, interpolation = 'nearest')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save and len(name) != 0:
        plt.savefig('./figs/' + name + '.pdf')
    else:
        plt.show()

def Clustering(phi, metodo  = 'complete', save = False, name = ''):
    plt.figure(figsize = (7,7))
    phi_star = np.zeros(phi.shape)
    linkage_matrix = linkage(squareform(1 - phi - np.identity(phi.shape[0])), method = metodo)
    indexes = leaves_list(linkage_matrix)

    for i,i_c in enumerate(indexes[::]):
        for j,j_c in enumerate(indexes[::]):
            phi_star[i,j] = phi[i_c, j_c]

    plt.imshow(phi_star)
    plt.title(r'$\phi_{pq}$')
    plt.xlabel('Categoria')
    plt.ylabel('Categoria')

    if save and len(name) != 0:
        plt.savefig('./figs/' + name + '.pdf')
    else:
        plt.show()

def Density_plot(domain, prob, xlabel = '', ylabel = '', xlim_sup = 0.7, save = False, name = ''):
    plt.bar(domain, prob, width=1 / len(domain), align='edge')
    plt.xlabel(xlabel, fontsize = 'x-large')
    plt.ylabel(ylabel, fontsize = 'x-large')
    plt.xlim([-0.05, xlim_sup])

    if save and len(name) != 0:
        plt.savefig('./figs/' + name + '.pdf')
    else:
        plt.show()

def red(phi, by_com = True, diccionario =None, PCI = None, umbral_enlace = 0.5, save = False, name = ''):
    Red_original = nx.from_numpy_array(phi)
    Red_nueva = nx.maximum_spanning_tree(Red_original)


    pesos_red = []
    for enlace in list(Red_original.edges.data()):
        peso = enlace[2]['weight']
        if peso > umbral_enlace:
            Red_nueva.add_edge(enlace[0], enlace[1], weight = peso)
        pesos_red.append(peso)
    k_degree = np.sum([2 for enlaces in Red_nueva.edges])/Red_nueva.number_of_nodes()
    print(k_degree) #Printea el grado promedio de la red sin peso.
    print( (phi[phi > umbral_enlace].sum() / phi.sum())*100, '%' ) #Contea la reconexion del top %

    posicion_red = nx.kamada_kawai_layout(Red_nueva, weight = None)
    pesos = np.array([enlace[2]['weight'] for enlace in Red_nueva.edges.data()])
    pesos = 2 * smooth(pesos)
    for n, nodo in enumerate(Red_nueva.nodes()):
        Red_nueva.nodes[nodo]['PCI'] = PCI[n]
    nx.write_gexf(Red_nueva, r'./Datos/Resultados/Red_Graphi.gexf')
    fig, ax = plt.subplots()
    if by_com:
        comunidades = nx.community.greedy_modularity_communities(Red_nueva)
        for comunidad, color, color_oscuro in zip(comunidades, ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:brown', 'tab:purple', 'tab:pink', 'tab:cyan'], ['#1e547b', '#2b752b', '#bb7130', '#8e2829', '#6f372c', '#6b3996', '#ad3f8c', 'cyan']):
            nx.draw_networkx_nodes(Red_original, pos=posicion_red, nodelist=comunidad, node_color= color_oscuro, node_size=95)
            nx.draw_networkx_nodes(Red_original, pos=posicion_red, nodelist=comunidad, node_color = color, node_size = 55)
    else:
        coloracion, barra = get_cmap(PCI)
        nx.draw_networkx_nodes(Red_nueva, pos=posicion_red, node_size = 55, node_color=coloracion)
        plt.colorbar(barra, ax=plt.gca())

    nx.draw_networkx_edges(Red_nueva, pos = posicion_red, width = pesos, alpha = 0.3)


    if save and len(name) != 0:
        plt.axis('off')
        plt.savefig('./figs/' + name + '.pdf', transparent = True, bbox_inches = 'tight')
    else:
        plt.tight_layout()
        plt.show()

def grafico_prueba(A):
    fig, ax = plt.subplots()
    ax.hist(A, bins = 30, density = True)
    plt.show()

def get_cmap(PCI, name='inferno'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    normalizacion = plt.Normalize(vmin = PCI.min(), vmax = PCI.max()+1)
    cmap = plt.get_cmap(name)
    lista = np.array([cmap(normalizacion(valores)) for valores in PCI])
    colorbar = plt.cm.ScalarMappable(cmap = cmap, norm = normalizacion)
    colorbar.set_array([])
    return lista, colorbar

def get_cmap_evenly(N, name='inferno'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    normalizacion = plt.Normalize(vmin = 0, vmax = N - 1)
    cmap = plt.get_cmap(name)
    lista = np.array([cmap(normalizacion(valores)) for valores in range(N)])
    return lista

def smooth(x,n=0.5):
    y = ( x - np.min(x)) / (np.max(x) - np.min(x))
    return (y + 0.35)/1.15

def equal_vectors(comunidades):
    np.random.seed(None)
    print(np.random.get_state()[1][0])
    r = 0.3
    N = len(comunidades)
    diff = 2 * np.pi/(N)
    centro_comm = np.zeros((N, 2))
    nodes_pos = {}
    for i in range(0, N - 1):
        centro_comm[i +1,:] = np.random.random(), np.random.random()
    for n, comunidad in enumerate(comunidades):
        for nodo in comunidad:
            posicion = r*centro_comm[n, :] + 0.03* np.random.normal(size = 2)
            nodes_pos.update({nodo : posicion})

    print('')

    return nodes_pos