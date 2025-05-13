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
import textalloc as ta

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

for i in range(15):
   dicts_DCI = trat.dictionaries(data_DCI)

   X_cpt = trat.X_matrix(data_DCI)[:,:,:12] #Los datos wipo van en 3 periodos de 5 años cada uno. Los datos awards solo consideran 12 periodos (el último no tiene nada)

   X_cpt = trat.sum_files(X_cpt, dicts_DCI, dicc.partida_award_llegada_wipo)



#----------------------------------------
#--------- Calculo de cosas -------------
#----------------------------------------

#============== RCA, M_cp ================

   X_cpt, RCA_cpt, M_cpt = calc.Matrices_ordenadas(X_cpt, dicts_DCI, 12, c_min = i, p_min = 10)

   #12 para awards
   #15 para WIPO
   #=============== Relatedness ===============

   #phi_t = calc.Similaridad(M_cpt)

   # #=============== Design Complexity Index ===============

   ECI_e, PCI_e = calc.Eigen_method(M_cpt, last = True)

   fig, ax = plt.subplots(figsize = (6,4))
   DCI_vs_GDP, paises = test.punteo_especifico(ECI_e[:,-1], gdp_array[:, -1], dicts_DCI[0], dict_country_gdp, dicc.awards_gdp, dicc.awards_iso)
   DCI = DCI_vs_GDP[:, 0]
   log_GDP = np.log(DCI_vs_GDP[:, 1])

   b_1, b_0, r, p, se = sc.linregress(DCI, log_GDP)

   ax.scatter(DCI, log_GDP, color='tab:blue', s=4 ** 2)
   ta.allocate(
      ax, DCI, log_GDP,
      paises, x_scatter = DCI, y_scatter = log_GDP, textsize = 6,
      draw_lines = False
   )

   reg_x = np.array([np.min(DCI), np.max(DCI)])
   reg_y = b_1 * reg_x + b_0

   ax.plot(reg_x, reg_y, color = 'tab:red', alpha = 0.6, label = f'n={i}\nm={b_1:.3f}\np={p:.3f}\n' + r'$r^2$' + f'={r**2:.3f}', linestyle = '--')
   ax.set_xlabel('DCI awards')
   ax.set_ylabel('mean log GDP per capita PPA')
   ax.set_title('A\'Design Awards')

   ax.legend(loc = 'lower right')
   if i < 11:
      plt.savefig(r'./figs/DCI_vs_PIB/awards_gdp_0' + f'{i}.png')
   else:
      plt.savefig(r'./figs/DCI_vs_PIB/awards_gdp_' + f'{i}.png')
   plt.close()

#---- Graficos -----

#figs.graf(np.log(RCA_cpt[:,:,-1] + 1), xlabel = 'Categorias', ylabel = 'Paises', title = r'log-$X_{cp' + f'{i}' + r'}$')
#
# figs.graf(np.log(R_cp + 1), xlabel = 'Categorias', ylabel = 'Paises', title = 'log-$RCA_{cp}$', name = 'log_RCA_awards', save = False)
#
# figs.graf(M_cp, xlabel = 'Categorias', ylabel = 'Paises',title = '$M_{cp}$')

#figs.Clustering(phi_t[:, :, -1], save = False, name = 'Similarity_matrix_awards.pdf')

# figs.k_density(phi)
#
# figs.red(phi, by_com = True, save = False, umbral_enlace = 0.45, name = 'Design_space_awards')
# #
# # figs.red(phi, PCI = PCI, diccionario = diccionaries, by_com = True, name = 'Espacio_productos_Comunidades', save = False, umbral_enlace = 0.4)
# #
#
# #dom_phi, relatedness = test.Relatedness_density_test(X_cpt, diccionaries, N_bins = 15, Awards=False, n_t = 1)
# #figs.Density_plot(dom_phi, relatedness, xlabel = r'Relatedness density', ylabel = 'Probability of developing RCA in a WIPO subclass', xlim_sup= 0.8, name = 'PrincipleOfRelatedness_wipo', save = False)

