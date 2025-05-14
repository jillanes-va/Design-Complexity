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
#gdp_columns = ['Country'] + [2011 + i for i in range(13)] + ['mean_awards'] #Para awards
gdp_columns = ['Country'] + ['mean_1', 'mean_2', 'mean_3'] + ['mean_wipo'] #Pawa WIPO

data_DCI = imp.carga(wipo_file, wipo_columns)
data_gdp = imp.carga_excel(gdp_file, gdp_columns)

#----------------------------------------
#--------- Tratamiento de datos ---------
#----------------------------------------

dict_country_gdp = trat.dictionaries(data_gdp)[0]
gdp_array = trat.gdp_matrix(data_gdp)


dicts_DCI = trat.dictionaries(data_DCI)

X_cpt = trat.X_matrix(data_DCI)[:,:,:12]#Los datos wipo van en 3 periodos de 5 años cada uno. Los datos awards solo consideran 12 periodos (el último no tiene nada)
#X_cpt = trat.agregado_movil(X_cpt, 4)
#X_cpt = trat.sum_files(X_cpt, dicts_DCI, dicc.partida_award_llegada_wipo)



#----------------------------------------
#--------- Calculo de cosas -------------
#----------------------------------------

#============== RCA, M_cp ================

X_cpt, RCA_cpt, M_cpt = calc.Matrices_ordenadas(X_cpt, dicts_DCI, 15, c_min = 1, p_min = 1)

#total time =
#        12 para awards
#        15 para WIPO
#c_min =
#        10 para awards

#=============== Relatedness ===============

phi_t = calc.Similaridad(M_cpt)

# #=============== Design Complexity Index ===============

ECI, PCI = calc.Eigen_method(M_cpt, last = False)
for i in range(len(ECI[0, :])):
    DCI_vs_GDP, paises = test.punteo_especifico(ECI[:,i], gdp_array[:, i], dicts_DCI[0], dict_country_gdp, dicc.wipo_gdp, dicc.wipo_iso)



    #---- Graficos -----

    figs.scatter_lm(DCI_vs_GDP, paises, log = True, param = ['DCI awards', 'log mean GDP per capita PPA', ''], save = False, name = 'DCI_awards_GDP_regression')

    figs.graf(np.log(RCA_cpt[:,:,i] + 1), xlabel = 'Categorias', ylabel = 'Paises', title = r'log-$X_{cp}$', save = False, name = 'logRCA_awards')

    figs.graf(M_cpt[:,:,i], xlabel = 'Categorias', ylabel = 'Paises',title = '$M_{cp}$', save = False, name = 'M_cp_awards')

    figs.Clustering(phi_t[:, :, i], save = False, name = 'Relatedness_awards')

    figs.k_density(phi_t[:, :, i], save = False, name = 'k_density_awards')

    figs.red(phi_t[:, :, i], by_com = True, save = False, umbral_enlace = 0.45, name = 'Design_space_awards_communitites')

#    figs.red(phi_t[:, :, i], by_com = False, save = False, umbral_enlace = 0.45, PCI = PCI, diccionario = dicts_DCI, name = 'Design_space_awards_PCI')
#
#
# dom_phi, relatedness = test.Relatedness_density_test(X_cpt, M_cpt, phi_t[:, :, -1], N_bins = 30)
# print(relatedness)
# figs.Density_plot(dom_phi, relatedness, param = ['Relatedness density', 'Probability of developing RCA in a award category', ''], name = 'PrincipleOfRelatedness_awards', save = False, xlim_sup = 0.8)

