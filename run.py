import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs

import matplotlib.pyplot as plt
import numpy as np

nombre_archivo = r'data_wipo_2010-2023.csv'
columnas = ['country_name', 'class_name', 'designs']

#nombre_archivo = r'wrd_04_all-data.csv'
#columnas = ['designer_country', 'award_category', 'award_period' , 'award_score']

datos_premio = imp.carga(nombre_archivo, columnas)
lista_de_cosas, cantidad_info = trat.domain_of_data(datos_premio)
diccionarios = trat.dictionaries(datos_premio)

X_cpt = trat.X_matrix(datos_premio, time = False)
X_cp = X_cpt /14
R_cp, M_cp, diccionarios = calc.Matrices_ordenadas(X_cp, lista_de_cosas, diccionarios)


phi = calc.Similaridad(M_cp)


figs.Clustering(phi)
print(len(lista_de_cosas[1]))
