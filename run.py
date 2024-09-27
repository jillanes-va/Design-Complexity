import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

#--------------------------------Carga de datos--------------------------------
str_wipo = r'data_wipo_2010-2023.csv'
columnas_wipo = ['country_name', 'subclass_name', 'designs']

str_premio = r'wrd_04_all-data.csv'
columnas_premios = ['designer_country', 'award_category', 'award_period' , 'award_score']

datos_premios = imp.carga(str_premio, columnas_premios)
lista_de_cosas_premios, cantidad_de_cosas_premios = trat.domain_of_data(datos_premios)
diccionarios_premios = trat.dictionaries(datos_premios)

X_premios_cpt = trat.X_matrix(datos_premios)[:, :, :12] #El periodo 2022 - 2023 solo tiene 3 registros
X_premios_cp = trat.Promedio_temporal(X_premios_cpt)

datos_wipo = imp.carga(str_wipo, columnas_wipo)
lista_de_cosas_wipo, cantidad_de_cosas_wipo = trat.domain_of_data(datos_wipo)
diccionarios_wipo = trat.dictionaries(datos_wipo)

X_wipo_cpt = trat.X_matrix(datos_wipo) #El periodo 2022 - 2023 solo tiene 3 registros
X_wipo_cp = trat.Promedio_temporal(X_wipo_cpt,14)
paises_pareados, paises_no_pareados = trat.pareo_listas(lista_de_cosas_premios[0], lista_de_cosas_wipo[0])

premios_por_pais = sorted(X_wipo_cpt.sum(axis = 1))
print(premios_por_pais)
N_paises = np.arange(1, len(premios_por_pais) + 1, 1)
plt.plot(premios_por_pais,  marker = '.')
plt.xlabel('Paises')
plt.ylabel('Cantidad de patentes')
plt.xscale('log')
plt.yscale('log')
plt.show()
# for i in np.linspace(0.1, 2, 20):
#
# #--------------------------------Calculo premios--------------------------------
#     R_premios_cp, M_premios_cp, diccionarios_premios = calc.Matrices_ordenadas(X_premios_cp, lista_de_cosas_premios, diccionarios_premios, i)
#     R_wipo_cp, M_wipo_cp, diccionarios_wipo = calc.Matrices_ordenadas(X_wipo_cp, lista_de_cosas_wipo, diccionarios_wipo, i)
#
#     N = 18
#
#     ECI_premios = calc.Complexity_measures(M_premios_cp, N)[0]
#     ECI_wipo = calc.Complexity_measures(M_wipo_cp, N)[0]
# #phi_premios = calc.Similaridad(M_cp)
# #omega_premios_cp = calc.Similarity_Density(R_cp)
#
# #--------------------------------Calculo WIPO--------------------------------
#
#
#
#
#
# #phi = calc.Similaridad(M_cp)
# #omega_cp = calc.Similarity_Density(R_cp)
#
# #--------------------------------Categorias presentes--------------------------------
# # cate = test.categorias_presentes(X_premios_cpt, diccionarios_wipo)
# # print([i[1] for i in cate])
#
# #--------------------------------Paises similares--------------------------------
#
#
#
# #--------------------------------Calculo ECI---------------------------------------------
#
#
# #--------------------------------Filtracion--------------------------------
#
#     ECI_premios_presente = np.array([ ECI_premios[ diccionarios_premios[0][pais] ] for pais in paises_pareados[0]])
#     ECI_wipo_presente = np.array([ ECI_wipo[ diccionarios_wipo[0][pais] ] for pais in paises_pareados[1]])
#
# #--------------------------------Enmascarado (!=0)--------------------------------
#     Mascara = (ECI_wipo_presente != 0)
#
#     ECI_premios_presente = calc.Z_transf(ECI_premios_presente[Mascara])
#     ECI_wipo_presente = calc.Z_transf(ECI_wipo_presente[Mascara])
#
# #--------------------------------Regresion lineal--------------------------------
#     res = linregress(ECI_wipo_presente, ECI_premios_presente)
#
# #--------------------------------Graficos--------------------------------
# plt.scatter(ECI_wipo_presente, ECI_premios_presente)
# plt.plot(ECI_wipo_presente, res.intercept + res.slope * ECI_wipo_presente, 'b', alpha = 0.8)
# print(res.rvalue**2)
# # plt.plot(ECI_wipo_presente, marker = '.')
# plt.show()
#

