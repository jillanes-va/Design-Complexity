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
gdp_columns = ['Country', 'mean_awards']

data_DCI = imp.carga(awards_file, awards_columns)
data_gdp = imp.carga_excel(gdp_file, gdp_columns)

#----------------------------------------
#--------- Tratamiento de datos ---------
#----------------------------------------

dicts_DCI = trat.dictionaries(data_DCI)

dict_country_gdp = trat.dictionaries(data_gdp)[0]
gdp_array = trat.gdp_matrix(data_gdp)

X_cpt = trat.X_matrix(data_DCI)[:,:,:] #Los datos wipo van en 3 periodos de 5 años cada uno. Los datos awards solo consideran 12 periodos (el último no tiene nada)

X_cpt = trat.sum_files(X_cpt, dicts_DCI, dicc.partida_award_llegada_wipo)



#----------------------------------------
#--------- Calculo de cosas -------------
#----------------------------------------

#============== RCA, M_cp ================

X_cpt, RCA_cpt, M_cpt = calc.Matrices_ordenadas(X_cpt, dicts_DCI, 15)

#=============== Relatedness ===============

phi_t = calc.Similaridad(M_cpt)

#=============== Design Complexity Index ===============

#ECI_e, PCI_e = calc.Eigen_method(M_cpt, last = True)
ECI_m, PCI_m = calc.Reflextion_method(M_cpt, 40, last = True)

last_ECI = ECI_m[:, -1]

DCI_vs_GDP, paises = test.punteo_especifico(last_ECI, gdp_array, dicts_DCI[0], dict_country_gdp, dicc.awards_gdp, dicc.awards_iso)

DCI = DCI_vs_GDP[:, 0]
log_GDP = np.log(DCI_vs_GDP[:, 1])


fig, ax = plt.subplots(figsize = (6,6))

ax.scatter(DCI, log_GDP, color = 'tab:blue', s = 4**2)
ta.allocate(
    ax, DCI, log_GDP,
    paises, x_scatter = DCI, y_scatter = log_GDP, textsize = 6,
    draw_lines = False
)

plt.xscale('linear')
plt.show()


# paises_ECI_e = { num_paises[n]:ECI_e[n] for n in range(len(ECI_e)) }
# paises_ECI_m = { num_paises[n]:ECI_m[n] for n in range(len(ECI_e)) }
#
# cat_PCI = { num_cat[n]: PCI_e[n] for n in range(len(PCI_e)) }
# cat_PCI = { num_cat[n]: PCI_e[n] for n in range(len(PCI_e)) }
#
# importantes = datos_gdp.loc[:, gdp_columns]
# paises_GDP = {tupla[0]:tupla[1] for tupla in importantes.values}
#
# print('Autovalores')
# print(sorted(tuple(paises_ECI_e), key = lambda X: X[1], reverse=0), '\n')
# print('Reflexiones')
# print(sorted(tuple(paises_ECI_m), key = lambda X: X[1], reverse=0), '\n')
#
# print(paises_ECI_m['Chile'])
#
# points = []
# dentro = []
# fuera = []
# labels = []

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

# diccionario de paises -> ECI
# diccionario de paises_x -> paises_gdp
# diccionario paises_gdp -> GDP

# fig, ax = plt.subplots(figsize = (10, 7))
# for c_prod, ECI in paises_ECI_m.items(): #cambiar el dicc por cada wea
#     try:
#         c_gdp = dicc.wipo_gdp[c_prod]
#         xy = np.array([ECI, np.log(paises_GDP[c_gdp])])
#         points.append(
#             xy
#         )
#         ax.annotate(dicc.wipo_iso[c_prod], xy + np.array([0.02, 0]), fontsize = 'x-small')
#         dentro.append([c_prod, c_gdp])
#     except:
#         fuera.append(c_prod)
#         pass
#
#
# points = np.array(points)[~np.isnan(points).any(axis = 1)]
#
# print(fuera)
#
# m, c, low_slope, high_slope = sc.theilslopes(points[:, 1], points[:,0])
# res = sc.spearmanr(points[:,0], points[:,1], nan_policy = 'omit')
# print('r y p para award vs gdp',res.statistic, res.pvalue)
# #
# X = [ min(points[:,0]), max(points[:,0]) ]
# Y = [ m * X[0] + c, m * X[1] + c ]
# ax.plot(X, Y, alpha=1, linestyle='--', color='red', label = f'$\\rho = {res.statistic:.2f}$\n' + r'$p-value<10^{-8}$')
# ax.scatter(points[:,0], points[:, 1], s = 5**2, color = 'tab:blue')
#
# ax.set_xlabel('DCI design from WIPO 2010-2024', size = 15)
# ax.set_ylabel('log mean GDP per capita PPA 2010-2024', size = 15)
# plt.legend()
# #plt.show()
# plt.savefig(r'figs/Regresion_DCI_wipo_PIB_per_capita.pdf')

# plt.scatter(ECI_e, ECI_m)
# plt.scatter(PCI_e, PCI_m)
# plt.xlabel('Metodo de autovalores')
# plt.ylabel('Metodo de las reflexiones')
# plt.show()
# # (pais_award, ECI) -> (pais_award, pais_gdp) -> (pais_gdp, GDP)
#
# # llave = lambda A: A[0]
# #
# # ECI_paises = sorted(ECI_paises, key = llave, reverse = 1)
# # PCI_cat = sorted(PCI_cat, key = llave, reverse = 1)
#
#
#
# # with open('./data/results/awards/Ranking_ECI_Design_Awards.csv', 'w+', encoding = 'utf-8') as f:
# #     f.writelines('ranking,country,ECI\n')
# #     for n, data in enumerate(ECI_paises):
# #         f.writelines(str(n + 1) + ',' + data[1] + ',' + str(data[0]) + '\n')
# #
# # with open('./data/results/awards/Ranking_PCI_Design_Awards.csv', 'w+', encoding = 'utf-8') as f:
# #     f.writelines('ranking,product,PCI\n')
# #     for n, data in enumerate(PCI_cat):
# #         f.writelines(str(n + 1) + ',' + data[1] + ',' + str(data[0]) + '\n')
# #
# # plt.scatter([i for i in range(len(ECI))], [ECI_paises[i][0] for i in range(len(ECI))])
# #
# # plt.scatter([i for i in range(len(PCI))], [PCI_cat[i][0] for i in range(len(PCI))])
# # plt.show()
# #
# # print(ECI_paises)
# #
# # figs.red(phi, PCI = PCI, diccionario = diccionaries, by_com = True, name = 'Espacio_productos_Comunidades', save = False, umbral_enlace = 0.4)
# #
#
# #dom_phi, relatedness = test.Relatedness_density_test(X_cpt, diccionaries, N_bins = 15, Awards=False, n_t = 1)
# #figs.Density_plot(dom_phi, relatedness, xlabel = r'Relatedness density', ylabel = 'Probability of developing RCA in a WIPO subclass', xlim_sup= 0.8, name = 'PrincipleOfRelatedness_wipo', save = False)
#
# # ECI_d = points[:, 0]
# # log_GDP = np.log(points[:, 1])
# #
# # m, c, low_slope, high_slope = sc.theilslopes(log_GDP, ECI_d)
# # #
# # X = [ min(ECI_d), max(ECI_d) ]
# # Y = [ m * X[0] + c, m * X[1] + c ]
# # #
# # # print(len(ECI_horizontal), len(ECI_vertical))
# # #
# # plt.scatter(ECI_d, log_GDP)
# # plt.plot(X, Y, alpha = 0.5, linestyle  = '--', color = 'red')
# # plt.xlabel('ECI del Diseño Awards 2011-2023')
# # plt.ylabel('log Promedio PIB per capita PPA 2011-2023')
# # plt.savefig('./figs/log_PIB_vs_ECI_awards.pdf')
# # plt.clf()
# #
# awards_WIPO = imp.dictionary_from_csv(r'dictionaries/dict_country_awards_wipo.csv')
# dict_ECI_d_awards = imp.dictionary_from_csv(r'results/awards/Ranking_ECI_Design_Awards.csv', ranking = True)
# dict_ECI_d_wipo = imp.dictionary_from_csv(r'results/wipo/Ranking_ECI_Design_wipo.csv', ranking = True)
# #
# #
# fig, ax = plt.subplots(figsize = (12,12))
# points = []
# for award, wipo in awards_WIPO.items():
#     try:
#         xy = np.array([dict_ECI_d_awards[award], dict_ECI_d_wipo[wipo]])
#         points.append(
#             xy
#         )
#         ax.annotate(dicc.wipo_iso[wipo], xy + np.array([-0.1, -0.1]), fontsize='x-small')
#     except:
#         pass
#
# points = np.array(points)
# ECI_d_awards = points[:, 0]
# ECI_d_wipo = points[:, 1]
#
# m, c, low_slope, high_slope = sc.theilslopes(ECI_d_awards, ECI_d_wipo)
# res = sc.spearmanr(points[:,0], points[:,1])
# print('r y p para award vs wipo',res.statistic, res.pvalue)
# #
# X = [ min(ECI_d_wipo), max(ECI_d_wipo) ]
# Y = [ m * X[0] + c, m * X[1] + c ]
# #
# # print(len(ECI_horizontal), len(ECI_vertical))
# #
# plt.scatter(ECI_d_awards, ECI_d_wipo)
# plt.plot(X, Y, alpha = 0.5, linestyle  = '--', color = 'red', label = '$\\rho=$0.48\n$p-value=$5.3e-6')
# plt.xlabel('DCI from Awards 2011-2023')
# plt.ylabel('DCI from WIPO 2010-2024')
# plt.legend()
# plt.show()
# # plt.savefig('./figs/Regresion_ECI_diseño_award_wipo.pdf')
# # plt.clf()
# #
# points = []
# for pais, ECI in dict_ECI_d_wipo.items():
#     try:
#         points.append(
#             [
#                 ECI, paises_GDP[pais]
#             ]
#         )
#     except:
#         pass
# points = np.array(points)
#
# m, c, low_slope, high_slope = sc.theilslopes(np.log(points[:, 1]), points[:, 0])
# #
# X = [ min(points[:, 0]), max(points[:, 0]) ]
# Y = [ m * X[0] + c, m * X[1] + c ]
#
# plt.scatter(points[:, 0], np.log(points[:, 1]))
# plt.plot(X, Y, alpha = 0.5, linestyle  = '--', color = 'red')
# plt.xlabel('ECI design from WIPO 2011-2023')
# plt.ylabel('log mean of GDP per capita PPA 2010-2024')
# plt.ylim([2,14])
# #plt.savefig('./figs/Regresion_ECI_diseño_wipo_PIB_per_capita.pdf')
