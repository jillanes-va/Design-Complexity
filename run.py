from sympy.physics.control.control_plots import matplotlib

import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs

#import numpy as np
#import matplotlib.pyplot as plt

str_wipo = r'data_wipo_2010-2023.csv'
columnas_wipo = ['country_name', 'subclass_name', 'designs']

str_premio = r'wrd_04_all-data.csv'
columnas_premios = ['designer_country', 'award_category', 'award_period' , 'award_score']

# Calculo premios

datos_premios = imp.carga(str_premio, columnas_premios)
lista_de_cosas_premios, cantidad_de_cosas_premios = trat.domain_of_data(datos_premios)
diccionarios_premios = trat.dictionaries(datos_premios)

X_premios_cpt = trat.X_matrix(datos_premios)[:, :, :12] #El periodo 2022 - 2023 solo tiene 3 registros
#X_premios_cp = trat.Promedio_temporal(X_premios_cpt)
#R_premios_cp, M_premios_cp, diccionarios = calc.Matrices_ordenadas(X_premios_cp, lista_de_cosas_premios, diccionarios_premios)
#phi_premios = calc.Similaridad(M_cp)
#omega_premios_cp = calc.Similarity_Density(R_cp)

# Calculo WIPO

datos_wipo = imp.carga(str_wipo, columnas_wipo)
lista_de_cosas_wipo, cantidad_de_cosas_wipo = trat.domain_of_data(datos_wipo)
diccionarios_wipo = trat.dictionaries(datos_wipo)

X_cpt = trat.X_matrix(datos_wipo) #El periodo 2022 - 2023 solo tiene 3 registros
#X_cp = trat.Promedio_temporal(X_cpt)
#R_cp, M_cp, diccionarios = calc.Matrices_ordenadas(X_cp, lista_de_cosas, diccionarios)
#phi = calc.Similaridad(M_cp)
#omega_cp = calc.Similarity_Density(R_cp)


#cate = test.categorias_presentes(X_cpt, diccionarios_wipo)

paises_wipo = lista_de_cosas_wipo[0].copy()
paises_premios = lista_de_cosas_premios[0].copy()
paises_wipo_presentes = []
paises_premios_presentes = []

i = 0

while i <= len(paises_premios):
    j = 0
    while j < len(paises_wipo):
        if not (paises_premios[i] in paises_wipo[j] ):
            j+=1
        else:
            paises_premios_presentes.append(paises_premios.pop(i))
            paises_wipo_presentes.append(paises_wipo.pop(j))
            break
    if j == len(paises_wipo):
        i += 1

nombres = ['paises_wipo_pareados.txt','paises_premio_pareados.txt', 'paises_wipo_no_pareados.txt','paises_premio_no_pareados.txt', ]
listados = [paises_wipo_presentes, paises_premios_presentes, paises_wipo, paises_premios]

for i in range(4):
    with open(r'./Datos/Paises/Clasificados/' + nombres[i], 'w' ) as text:
        for paises in listados[i]:
            text.writelines(paises + '\n')
