import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform

def Clustering(phi):

    metodos = ['single', 'complete', 'average']
    plt.figure(figsize = (16,4))
    for k,metodo in enumerate(metodos):
        phi_star = np.zeros(phi.shape)
        linkage_matrix = linkage(squareform(1 - phi - np.identity(phi.shape[0])), method = metodo)
        indexes = leaves_list(linkage_matrix)

        for i,i_c in enumerate(indexes[::]):
            for j,j_c in enumerate(indexes[::]):
                phi_star[i,j] = phi[i_c, j_c]

        plt.subplot(1,3, k + 1)
        plt.imshow(phi_star)
        plt.title(metodo)
    plt.show()