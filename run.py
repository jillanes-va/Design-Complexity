from sympy.physics.control.control_plots import matplotlib

import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


#----------------------------------Tipo de archivo a importar----------------------------------------------
nombre_archivo = r'wipo_design.csv'
columnas = ['country_name', 'subclass_name', 'wipo_year_from', 'n']

#nombre_archivo = r'wrd_04_all-data.csv'
#columnas = ['designer_country', 'award_category', 'award_period' , 'award_score']

#----------------------------------Importar y limpiar----------------------------------------------

datos_premio = imp.carga(nombre_archivo, columnas)
lista_de_cosas, cantidad_info = trat.domain_of_data(datos_premio)
diccionaries = trat.dictionaries(datos_premio)

X_cpt = trat.X_matrix(datos_premio)

X_cpt = trat.Promedio_temporal(X_cpt, Awards= False)
X_cp = X_cpt[:,:,2]

#Informacion = test.categorias_presentes(X_cpt, diccionaries)
# plt.scatter([i[0] for i in Informacion], [i[1] for i in Informacion])
# plt.ylabel('Frecuencia absoluta')
# plt.title('Categorias presentes por año')
# plt.tight_layout()
# plt.xticks(rotation = 45)
# plt.show()

#
#sisi = datos_premio.groupby(['designer_country']).sum()
#print(sisi)
#print(sisi[sisi['award_score'] > 5.0])

#------------------------------------Calculo de RCA y demás--------------------------------------------

R_cp, M_cp, X_cp = calc.Matrices_ordenadas(X_cp, diccionaries, 1, 2 )
figs.graf(np.log(X_cp + 1), xlabel = 'Subclases', ylabel = 'Paises', title = 'log-$X_{cp}$')

figs.graf(np.log(R_cp + 1), xlabel = 'Subclases', ylabel = 'Paises', title = 'log-$RCA_{cp}$')

figs.graf(M_cp, xlabel = 'Subclases', ylabel = 'Paises',title = '$M_{cp}$')

#-------------------------------------Calculo de ECI y PCI-------------------------------------------

k_0 =  calc.Complexity_measures(M_cp, 0)[0]
k_1 =  calc.Complexity_measures(M_cp, 1)[0]

plt.scatter(k_0, k_1)
plt.xlabel('k_0')
plt.ylabel('k_1')
plt.show()
#
#


ECI = calc.Z_transf(calc.Complexity_measures(M_cp, 2 * 9)[0])
PCI = calc.Z_transf(calc.Complexity_measures(M_cp, 2 * 9)[1])

num_paises = trat.inv_dict(diccionaries[0])
num_cat = trat.inv_dict(diccionaries[1])

#
ECI_paises = [ (ECI[n], num_paises[n]) for n in range(len(ECI)) ]
PCI_cat = [ (PCI[n], num_cat[n]) for n in range(len(PCI)) ]
#
ECI_paises = sorted(ECI_paises, key=lambda A: A[0], reverse=1)
PCI_cat = sorted(PCI_cat, key=lambda A: A[0], reverse=1)
#
plt.scatter([i for i in range(len(ECI))], ECI)
plt.title('Indice de Complejidad Economica')
plt.show()
#
plt.scatter([i for i in range(len(PCI))], PCI)
plt.title('Indice de Complejidad Economica de los Productos')
plt.show()
#
#
#-------------------------------------Calculo de Similitud y densidad de Similitud-------------------------------------------
phi = calc.Similaridad(M_cp)
omega_cp = calc.Similarity_Density(R_cp)
figs.red(phi, PCI = PCI, diccionario = diccionaries, by_com = False, name = 'Espacio_productos_Comunidades', save = False, umbral_enlace = 0.4)
#
figs.Clustering(phi, save = False)
#
dom_phi, relatedness = test.Relatedness_density_test(X_cpt, diccionaries,n_t = 1, N_bins = 15, Awards = False)
figs.Density_plot(dom_phi, relatedness, xlabel = r'Densidad de similaridad', ylabel = 'Probabilidad de transicionar en alguna categoría', xlim_sup= 0.83, name = 'PrincipleOfRelatedness', save = False)