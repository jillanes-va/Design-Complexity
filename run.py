from idlelib.autocomplete import TRY_A

import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs
import lib.Diccionariacion as dicc

import os

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import pandas as pd

from lib.Testeo_estadistico import mi_pais_a_ganado

#----------------------------------
#--------- Carga de datos ---------
#----------------------------------
is_award = True
save_things = False
thre = 1


population_file = r'World_Population.xls'
population_columns = ['Country Name', 'mean_awards']

wipo_file = r'wipo_design.csv'
wipo_columns = ['country_name','subclass_name', 'wipo_year_to', 'n']

awards_file = r'wrd_04_all-data.csv'
awards_columns = ['designer_country', 'award_category', 'award_period' , 'award_score']

gdp_file = r'IMF_GDP_per_PPA_April_2024.xlsx'
awards_gdp_columns = ['Country'] + [2011 + i for i in range(13)] + ['mean_awards'] #Para awards
wipo_gdp_columns = ['Country'] + ['mean_1', 'mean_2', 'mean_3'] + ['mean_wipo'] #Pawa WIPO

print('Importando...')

if is_award:
    data_DCI = imp.carga(awards_file, awards_columns)#***
    data_gdp = imp.carga_excel(gdp_file, awards_gdp_columns)
else:
    data_DCI = imp.carga(wipo_file, wipo_columns)#***
    data_gdp = imp.carga_excel(gdp_file, wipo_gdp_columns)

data_pop = imp.carga_excel(population_file, population_columns)

#----------------------------------------
#--------- Tratamiento de datos ---------
#----------------------------------------

print('Generando dicts')

for thre in np.arange(0.4, 1.2, 0.1):
    dicts_DCI = trat.dictionaries(data_DCI)
    dict_country_gdp = trat.dictionaries(data_gdp)[0]

    if is_award:
        dict_country_pop = trat.interchange_dict(dicc.awards_pop, trat.direct_dict(data_pop, population_columns))
    else:
        dict_country_pop = trat.interchange_dict(dicc.wipo_pop, trat.direct_dict(data_pop, population_columns))
    gdp_array = trat.gdp_matrix(data_gdp)



    print('Calculando Matriz X')
    X_cpt = trat.X_matrix(data_DCI) #Los datos wipo van en 3 periodos de 5 años cada uno. Los datos awards solo consideran 12 periodos (el último no tiene nada)

    if is_award:
        X_cpt = trat.sum_files(X_cpt, dicts_DCI, dicc.partida_award_llegada_wipo)
        X_cpt = trat.agregado_movil_(X_cpt)

    #----------------------------------------
    #--------- Calculo de cosas -------------
    #----------------------------------------

    #============== RCA, M_cp ================

    # print('Matrices RCA y demás')
    #
    # print('Suma sobre productos', np.max(X_cpt.sum(axis = 1)))
    # print('Suma sobre paises', np.max(X_cpt.sum(axis = 0)))
    if is_award:
        X_cpt, RCA_cpt, M_cpt = calc.Matrices_ordenadas(X_cpt, dicts_DCI, dict_country_pop, threshold=thre, pop_min=1_000_000,
                                                        c_min=10,
                                                        p_min=10)  # c min significa cantidad de premios minima de un país
    else:
        X_cpt, RCA_cpt, M_cpt = calc.Matrices_ordenadas(X_cpt, dicts_DCI, dict_country_pop, threshold=thre, pop_min=1_000_000,
                                                    c_min=10,
                                                    p_min=10)  # c min significa cantidad de premios minima de un país
    figs.graf(M_cpt[:, :, -1], xlabel='Categorias', ylabel='Paises', title=f'Umbral = {thre} ', save=False, name='M_cp_awards')
#test.mi_pais_a_ganado(X_cpt, 'Italy', dicts_DCI, -1)

#c_min =
#        10 para awards

#=============== Relatedness ===============

#print('Calculando Relatedness...')
#phi_t = calc.Relatedness(M_cpt)

# #=============== Design Complexity Index ===============

print('Calculando DCI...')
DCI, PCI = calc.Eigen_method(M_cpt, last = True)
N = len(DCI)

# for i in range(len(DCI)):
#     print(DCI[i], trat.inv_dict(dicts_DCI[0])[i])
#
# anios = ['2015-2019', '2020-2023','']

if save_things:
    print('Guardado...')
    if is_award:
        imp.guardado_ranking(DCI, dicts_DCI[0], 'awards', ['2015-2019', '2020-2023'], 'Ranking_DCI_awards', 'DCI')
        imp.guardado_ranking(PCI, dicts_DCI[1], 'awards', ['2015-2019', '2020-2023'], 'Ranking_PCI_awards', 'PCI')
    else:
        imp.guardado_ranking(DCI, dicts_DCI[0], 'wipo', ['2010-2014','2015-2019','2020-2024'], 'Ranking_DCI_wipo', 'DCI')
        imp.guardado_ranking(PCI, dicts_DCI[1], 'wipo', ['2010-2014','2015-2019','2020-2024'], 'Ranking_PCI_wipo', 'PCI')

# for i in range(3):
if is_award:
    DCI_vs_GDP, paises = test.punteo_especifico(DCI[:, -1], gdp_array[:, -1], dicts_DCI[0], dict_country_gdp, dicc.awards_gdp, dicc.awards_iso)
else:
    DCI_vs_GDP, paises = test.punteo_especifico(DCI[:, -1], gdp_array[:, -1], dicts_DCI[0], dict_country_gdp, dicc.wipo_gdp, dicc.wipo_iso)

#---- Graficos -----
slope, intercept, rho, pvalue = figs.scatter_lm(DCI_vs_GDP, paises, log = True, param = ['DCI awards', 'log mean GDP per capita PPA', ''], save = False, name = 'DCI_awards_GDP_regression')

# for i in range(0):
#     figs.graf(np.log(X_cpt[:,:, i] + 1), xlabel = 'Categorias', ylabel = 'Paises', title = r'log-$X_{cp}$', save = False, name = 'logRCA_awards')
#
# for i in range(0):
#     figs.graf(np.log(RCA_cpt[:,:, i] + 1), xlabel = 'Categorias', ylabel = 'Paises', title = r'log-$RCA_{cp}$', save = False, name = 'logRCA_awards')
#
# for i in range(0):
#     figs.graf(np.log(M_cpt[:,:, i] + 1), xlabel = 'Categorias', ylabel = 'Paises', title = r'log-$M_{cp}$', save = False, name = 'logRCA_awards')
#
# figs.graf(M_cpt[:,:, -1], xlabel = 'Categorias', ylabel = 'Paises',title = '$M_{cp}$', save = False, name = 'M_cp_awards')
#
# figs.Clustering(phi_t[:, :, -1], save = False, name = 'Relatedness_awards')
#
# figs.k_density(phi_t[:, :, -1], save = False, name = 'k_density_awards')
#
# figs.red(phi_t[:, :, -1], by_com = True, save = False, umbral_enlace = 0.45, name = 'Design_space_awards_communitites')
#
# figs.red(phi_t[:, :, -1], by_com = False, save = False, umbral_enlace = 0.45, PCI = PCI, diccionario = dicts_DCI, name = 'Design_space_awards_PCI')
#
#omega_cp = calc.relatedness_density(M_cpt, False)
#dom_phi, relatedness, dict_trans = test.Relatedness_density_test(X_cpt, threeshold = thre, M_inicial = None, phi_inicial = None, mid_index = 1, N_bins = 15)
#figs.Density_plot(dom_phi, relatedness, param = ['Relatedness density', 'Probability of developing RCA in a award category', ''], name = 'PrincipleOfRelatedness_awards', save = False)