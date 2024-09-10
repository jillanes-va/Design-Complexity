import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform

def graf(X, save = False, name = ''):
    plt.figure(figsize = (7,7))
    plt.imshow(X, interpolation = 'nearest', cmap = 'afmhot')
    plt.xlabel('Categoria')
    plt.ylabel('Paises')
    if save and len(name) != 0:
        plt.savefig('./figs/' + name + '.pdf')

def Clustering(phi, metodo  = 'complete', save = True):
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
    plt.show()
