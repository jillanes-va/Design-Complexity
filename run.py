from sympy.physics.control.control_plots import matplotlib

import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs

import numpy as np
import matplotlib.pyplot as plt

#nombre_archivo = r'data_wipo_2010-2023.csv'
#columnas = ['country_name', 'subclass_name', 'designs']

nombre_archivo = r'wrd_04_all-data.csv'
columnas = ['designer_country', 'award_category', 'award_period' , 'award_score']

datos_premio = imp.carga(nombre_archivo, columnas)
lista_de_cosas, cantidad_info = trat.domain_of_data(datos_premio)

diccionaries = trat.dictionaries(datos_premio)

X_cpt = trat.X_matrix(datos_premio)
X_cp = trat.Promedio_temporal(X_cpt)

figs.graf(np.log(X_cp + 1))
#sisi = datos_premio.groupby(['designer_country']).sum()
#print(sisi)
#print(sisi[sisi['award_score'] > 5.0])

R_cp, M_cp, X_cp = calc.Matrices_ordenadas(X_cp, diccionaries, 1)
figs.graf(np.log(X_cp + 1))

phi = calc.Similaridad(M_cp)
omega_cp = calc.Similarity_Density(R_cp)

test.Relatedness_density_test(X_cpt)






