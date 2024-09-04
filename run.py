import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs

import matplotlib.pyplot as plt
import numpy as np

# nombre_archivo = r'data_wipo_2010-2023.csv'
# columnas = ['country_name', 'subclass_name', 'designs']

nombre_archivo = r'wrd_04_all-data.csv'
columnas = ['designer_country', 'award_category', 'award_period' , 'award_score']

datos_premio = imp.carga(nombre_archivo, columnas)
lista_de_cosas, cantidad_info = trat.domain_of_data(datos_premio)
diccionarios = trat.dictionaries(datos_premio)

X_cpt = trat.X_matrix(datos_premio, time = True)
X_cp = X_cpt.sum(axis = 2) /13
R_cp, M_cp, diccionarios = calc.Matrices_ordenadas(X_cp, lista_de_cosas, diccionarios)


phi = calc.Similaridad(M_cp)
n,m = phi.shape
phi_star = np.zeros((n,m))

# Algoritmo de modularidad para matrices
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform


linkage_matrix = linkage(squareform(1 - phi - np.identity(n)), method = 'ward')
indexes = leaves_list(linkage_matrix)

for i,i_c in enumerate(indexes[::]):
    for j,j_c in enumerate(indexes[::]):
        phi_star[i,j] = phi[i_c, j_c]

plt.imshow(phi_star)
plt.show()
