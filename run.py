from sympy.physics.control.control_plots import matplotlib

import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#nombre_archivo = r'data_wipo_2010-2023.csv'
#columnas = ['country_name', 'subclass_name', 'designs']

nombre_archivo = r'wrd_04_all-data.csv'
columnas = ['designer_country', 'award_category', 'award_period' , 'award_score']

datos_premio = imp.carga(nombre_archivo, columnas)
lista_de_cosas, cantidad_info = trat.domain_of_data(datos_premio)

diccionaries = trat.dictionaries(datos_premio)

X_cpt = trat.X_matrix(datos_premio)
X_cp = trat.Promedio_temporal(X_cpt)

Informacion = test.categorias_presentes(X_cpt, diccionaries)
plt.scatter([i[0] for i in Informacion], [i[1] for i in Informacion])
plt.ylabel('Frecuencia absoluta')
plt.title('Categorias presentes por año')
plt.xticks(rotation = 90)
plt.show()

figs.graf(np.log(X_cp + 1), xlabel = 'Categorias', ylabel = 'Paises', title = '$X_{cp}$')
#sisi = datos_premio.groupby(['designer_country']).sum()
#print(sisi)
#print(sisi[sisi['award_score'] > 5.0])

R_cp, M_cp, X_cp = calc.Matrices_ordenadas(X_cp, diccionaries, 1)
figs.graf(np.log(R_cp + 1), xlabel = 'Categorias', ylabel = 'Paises', title = '$R_{cp}$')

figs.graf(M_cp, xlabel = 'Categorias', ylabel = 'Paises',title = '$M_{cp}$')

ECI = calc.Z_transf( calc.Complexity_measures(M_cp, 2 * 7)[0] )
plt.scatter([i for i in range(len(ECI))], ECI)
plt.title('Indice de Complejidad Economica')
plt.show()

phi = calc.Similaridad(M_cp)
omega_cp = calc.Similarity_Density(R_cp)

figs.Clustering(phi)

test.Relatedness_density_test(X_cpt, diccionaries, N_bins = 15)






