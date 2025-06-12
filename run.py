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

#----------------------------------
#--------- Carga de datos ---------
#----------------------------------

trade_file = r'/datasets/BACI_HS22_Y2023_V202501.csv'
trade_columns = ['i', 'k','t','v']

dict_country_trade_file = r'/datasets/country_codes_V202501.csv'

dict_product_trade_file = r'/datasets/product_codes_HS22_V202501.csv'

wipo_file = r'wipo_design.csv'
wipo_columns = ['country_name','subclass_name', 'wipo_year_to', 'n']

awards_file = r'wrd_04_all-data.csv'
awards_columns = ['designer_country', 'award_category', 'award_period' , 'award_score']

gdp_file = r'IMF_GDP_per_PPA_April_2024.xlsx'
awards_gdp_columns = ['Country'] + [2011 + i for i in range(13)] + ['mean_awards'] #Para awards
wipo_gdp_columns = ['Country'] + ['mean_1', 'mean_2', 'mean_3'] + ['mean_wipo'] #Pawa WIPO

data_DCI = imp.carga(awards_file, awards_columns)#***
data_gdp = imp.carga_excel(gdp_file, awards_gdp_columns)

#----------------------------------------
#--------- Tratamiento de datos ---------
#----------------------------------------

dict_country_gdp = trat.dictionaries(data_gdp)[0]
gdp_array = trat.gdp_matrix(data_gdp)

print('Importando...')


print('Calculando Matriz X')

p = np.arange(0, 125, 5)
c = np.arange(0, 500, 10)
Total = len(c) * len(p)
contador = 0

R_sq = np.full(shape = (len(c), len(p)), fill_value = np.nan)
p_value = np.full(shape = (len(c), len(p)), fill_value = np.nan)

for i in range(len(c)):
    for j in range(len(p)):
        contador += 1
        t_0 = time.time()
        dicts_DCI = trat.dictionaries(data_DCI)
        X_cpt = trat.X_matrix(data_DCI) #Los datos wipo van en 3 periodos de 5 años cada uno. Los datos awards solo consideran 12 periodos (el último no tiene nada)
        X_cpt = trat.sum_files(X_cpt, dicts_DCI, dicc.partida_award_llegada_wipo)
        X_cpt = trat.agregado_movil(X_cpt, 5)

    #----------------------------------------
    #--------- Calculo de cosas -------------
    #----------------------------------------

    #============== RCA, M_cp ================

        # print('Matrices RCA y demás')
        #
        # print('Suma sobre productos', np.max(X_cpt.sum(axis = 1)))
        # print('Suma sobre paises', np.max(X_cpt.sum(axis = 0)))

        X_cpt, RCA_cpt, M_cpt = calc.Matrices_ordenadas(X_cpt, dicts_DCI, c_min = c[i], p_min = p[j]) #c min significa cantidad de premios minima de un país
        #c_min =
        #        10 para awards

        #=============== Relatedness ===============

        #print('Relatedness')
        #phi_t = calc.Relatedness(M_cpt)

        # #=============== Design Complexity Index ===============

        #print('DCI')
        try:
            DCI, PCI = calc.Eigen_method(M_cpt, last = True)

        #
        # anios = ['2015-2019', '2020-2023','']

        #print('Guardado...')
        #imp.guardado_ranking(DCI, dicts_DCI[0], '', [''], 'Ranking_DCI_awards')
        #imp.guardado_ranking(PCI, dicts_DCI[1], '', [''], 'Ranking_PCI_awards')

            DCI_vs_GDP, paises = test.punteo_especifico(DCI[:, -1], gdp_array[:, -1], dicts_DCI[0], dict_country_gdp, dicc.awards_gdp, dicc.awards_iso)

        #---- Graficos -----
            R_sq[i, j], p_value[i, j] = figs.scatter_lm(DCI_vs_GDP, paises, log = True, param = ['DCI awards', 'log mean GDP per capita PPA', ''], save = False, name = 'DCI_awards_GDP_regression')
        except:
            pass
        t_1 = time.time()
        print(round(100 * contador / Total, 2))
        print(f'{t_1 - t_0}\n', flush = True)

C, P = np.meshgrid(c, p)


plt.pcolor(C, P, R_sq.T)
plt.colorbar()
plt.xlabel('Umbral de productos (c_min)')
plt.ylabel('Umbral de paises (p_min)')
plt.title('R squared')
plt.show()
plt.close()

plt.pcolor(C, P, np.log(p_value.T))
plt.colorbar()
plt.xlabel('Umbral de productos (c_min)')
plt.ylabel('Umbral de paises (p_min)')
plt.title('log p_value')
plt.show()

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
# omega_cp = calc.relatedness_density(M_cpt, False)
# dom_phi, relatedness, dict_trans = test.Relatedness_density_test(X_cpt, M_inicial = None, phi_inicial = None, mid_index = i, N_bins = 15)
# figs.Density_plot(dom_phi, relatedness, param = ['Relatedness density', 'Probability of developing RCA in a award category', ''], name = 'PrincipleOfRelatedness_awards', save = False)