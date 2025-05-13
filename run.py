from idlelib.autocomplete import TRY_A

import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs
import lib.Diccionariacion as dicc

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

#----------------------------------
#--------- Carga de datos ---------
#----------------------------------

wipo_file = r'wipo_design.csv'
wipo_columns = ['country_name','subclass_name', 'wipo_year_to', 'n']

awards_file = r'wrd_04_all-data.csv'
awards_columns = ['designer_country', 'award_category', 'award_period' , 'award_score']

gdp_file = r'IMF_GDP_per_PPA_April_2024.xlsx'
gdp_columns = ['Country'] + [2011 + i for i in range(13)] + ['mean_awards']

data_DCI = imp.carga(awards_file, awards_columns)
data_gdp = imp.carga_excel(gdp_file, None)

#----------------------------------------
#--------- Tratamiento de datos ---------
#----------------------------------------

dict_country_gdp = trat.dictionaries(data_gdp)[0]
gdp_array = trat.gdp_matrix(data_gdp)

dicts_DCI = trat.dictionaries(data_DCI)

X_cpt = trat.X_matrix(data_DCI)[:,:,:12] #Los datos wipo van en 3 periodos de 5 años cada uno. Los datos awards solo consideran 12 periodos (el último no tiene nada)
X_cpt = trat.sum_files(X_cpt, dicts_DCI, dicc.partida_award_llegada_wipo)



#----------------------------------------
#--------- Calculo de cosas -------------
#----------------------------------------

#============== RCA, M_cp ================

X_cpt, RCA_cpt, M_cpt = calc.Matrices_ordenadas(X_cpt, dicts_DCI, 12, c_min = 0, p_min = 0)

#total time =
#        12 para awards
#        15 para WIPO
#c_min =
#        10 para awards

#=============== Relatedness ===============

phi_t = calc.Similaridad(M_cpt)

# #=============== Design Complexity Index ===============

ECI, PCI = calc.Eigen_method(M_cpt, last = True)

DCI_vs_GDP, paises = test.punteo_especifico(ECI[:,-1], gdp_array[:, -1], dicts_DCI[0], dict_country_gdp, dicc.wipo_gdp, dicc.wipo_iso)



#---- Graficos -----

figs.scatter_lm(DCI_vs_GDP, paises, log = True, param = ['DCI awards', 'log mean GDP per capita PPA', ''], save = False)

figs.graf(np.log(RCA_cpt[:,:,-1] + 1), xlabel = 'Categorias', ylabel = 'Paises', title = r'log-$X_{cp' + f'{2}' + r'}$')

figs.graf(M_cpt[:,:,-1], xlabel = 'Categorias', ylabel = 'Paises',title = '$M_{cp}$')

figs.Clustering(phi_t[:, :, -1], save = False, name = 'Similarity_matrix_awards.pdf')

figs.k_density(phi_t[:, :, -1])

figs.red(phi_t[:, :, -1], by_com = True, save = False, umbral_enlace = 0.45, name = 'Design_space_awards')

figs.red(phi_t[:, :, -1], PCI = PCI, diccionario = dicts_DCI, by_com = True, name = 'Espacio_productos_Comunidades', save = False, umbral_enlace = 0.4)


dom_phi, relatedness = test.Relatedness_density_test(X_cpt, M_cpt, phi_t[:, :, -1], mid_index = 4, N_bins = 30)
print(relatedness)
figs.Density_plot(dom_phi, relatedness, param = ['Relatedness density', 'Probability of developing RCA in a award category', ''], name = 'PrincipleOfRelatedness_wipo', save = False)

