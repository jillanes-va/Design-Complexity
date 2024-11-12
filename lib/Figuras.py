import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform

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
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([-0.05, xlim_sup])

    if save and len(name) != 0:
        plt.savefig('./figs/' + name + '.pdf')
    else:
        plt.show()

def red(phi, PCI,inicio = 10, max_color = 120, umbral_enlace = 0.5, save = False, name = ''):
    Red_original = nx.from_numpy_array(phi)
    Red_nueva = nx.maximum_spanning_tree(Red_original)

    pesos_red = []
    for enlace in list(Red_original.edges.data()):
        peso = enlace[2]['weight']
        if peso > umbral_enlace:
            Red_nueva.add_edge(enlace[0], enlace[1], weight = peso)
        pesos_red.append(peso)
    k_degree = np.sum([2 for enlaces in Red_nueva.edges])/Red_nueva.number_of_nodes()
    print(k_degree)
    print( (phi[phi > umbral_enlace].sum() / phi.sum())*100, '%' )

    posicion_red = nx.kamada_kawai_layout(Red_nueva)
    pesos = np.array([enlace[2]['weight'] for enlace in Red_nueva.edges.data()])
    pesos = 3*smooth(pesos)
    coloracion = get_cmap(PCI)

    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(Red_nueva, pos = posicion_red, node_size= 50, node_color = coloracion )
    nx.draw_networkx_edges(Red_nueva, pos = posicion_red, width = pesos)

    if save and len(name) != 0:
        plt.axis('off')
        plt.savefig('./figs/' + name + '.pdf', transparent = True, bbox_inches = 'tight')
    else:
        plt.show()

def grafico_prueba(A):
    fig, ax = plt.subplots()
    ax.hist(A, bins = 30, density = True)
    plt.show()

def get_cmap(PCI, name='plasma'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    normalizacion = plt.Normalize(vmin = PCI.min(), vmax = PCI.max())
    cmap = plt.get_cmap(name)
    lista = np.array([cmap(normalizacion(valores)) for valores in PCI])
    return lista

def smooth(x,n=0.5):
    y = ( x - np.min(x)) / (np.max(x) - np.min(x))
    return (y + 0.35)/1.35