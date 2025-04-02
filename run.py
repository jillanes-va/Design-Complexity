import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs
import lib.Diccionariacion as dicc

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

from lib.Importacion import carga_excel

#awards_str = r'wipo_design.csv'
#awards_columns = ['country_name','subclass_name', 'wipo_year_to', 'n']
#
awards_str = r'wrd_04_all-data.csv'
awards_columns = ['designer_country', 'award_category', 'award_period' , 'award_score']

#export_str = r'wtf00.dta'
#export_columns = ['exporter', 'sitc4', 'value']

# Informacion = test.categorias_presentes(X_cpt[:,:,:12], diccionaries)
# plt.scatter([i[0] for i in Informacion], [i[1] for i in Informacion])
# plt.ylabel('Frecuencia absoluta')
# plt.title('Categorias presentes por año')
# plt.tight_layout()
# plt.xticks(rotation = 45)
# plt.show()
#
#
# sisi = datos_premio.groupby(['designer_country']).sum()
# print(sisi)
# print(sisi[sisi['award_score'] > 5.0])

datos = imp.carga(awards_str, awards_columns)
diccionaries = trat.dictionaries(datos)

X_cpt = trat.X_matrix(datos)
datos_gdp = carga_excel('IMF_GDP_per_PPA.xls')
datos_gdp['media_wipo'] = datos_gdp.mean(axis = 1, numeric_only = True)
datos_gdp['media_awards'] = datos_gdp.loc[:, 2011:2023].mean(axis = 1, numeric_only = True)

diccionaries_gdp = trat.dictionaries(datos_gdp)[0]
gdp_by_country = trat.gdp_matrix(datos_gdp, last = 0)

X_cp = trat.Promedio_temporal(X_cpt, Awards= True, n_time = None) #Los datos wipo van en 3 periodos de 5 años cada uno.
#X_cp = X_cpt[:,:,0]

R_cp, M_cp, X_cp = calc.Matrices_ordenadas(X_cp, diccionaries, time = False)

X_cp, diccionaries[0] = trat.sum_files(X_cp, dicc.partida_award_llegada_wipo, diccionaries[0])

# figs.graf(np.log(X_cp + 1), xlabel = 'Categorias', ylabel = 'Paises', title = 'log-$X_{cp}$')
#
# figs.graf(np.log(R_cp + 1), xlabel = 'Categorias', ylabel = 'Paises', title = 'log-$RCA_{cp}$', name = 'log_RCA_awards', save = False)
#
# figs.graf(M_cp, xlabel = 'Categorias', ylabel = 'Paises',title = '$M_{cp}$')
#
# phi = calc.Similaridad(M_cp)
# figs.Clustering(phi, save = False, name = 'Similarity_matrix_awards.pdf')
#
#
#
#
# figs.k_density(phi)
#
# figs.red(phi, by_com = True, save = False, umbral_enlace = 0.45, name = 'Design_space_awards')

num_paises = trat.inv_dict(diccionaries[0])
num_cat = trat.inv_dict(diccionaries[1])


ECI_e, PCI_e= calc.Alt_Complexity_measures(M_cp)
ECI_m, PCI_m= calc.Complexity_measures(M_cp, 60)

paises_ECI_e = { num_paises[n]:ECI_e[n] for n in range(len(ECI_e)) }
paises_ECI_m = { num_paises[n]:ECI_m[n] for n in range(len(ECI_e)) }

cat_PCI = { num_cat[n]: PCI_e[n] for n in range(len(PCI_e)) }
cat_PCI = { num_cat[n]: PCI_e[n] for n in range(len(PCI_e)) }

importantes = datos_gdp.loc[:, ['GDP per capita, current prices\n (U.S. dollars per capita)', 'media_awards']]
paises_GDP = {tupla[0]:tupla[1] for tupla in importantes.values}

print('Autovalores')
print(sorted(tuple(paises_ECI_e), key = lambda X: X[1], reverse=1))
print('Reflexiones')
print(sorted(tuple(paises_ECI_m), key = lambda X: X[1], reverse=1))

points = []
labels = []

fig, ax = plt.subplots(figsize = (12, 12))
for c_awards, c_gdp in dicc.awards_gdp.items(): #cambiar el dicc por cada wea
    try:
        xy = np.array([paises_ECI_e[c_awards], np.log(paises_GDP[c_gdp])])
        points.append(
            xy
        )
        ax.annotate(dicc.award_iso[c_awards], xy + np.array([0.02, 0]), fontsize = 'x-small')
    except:
        pass


points = np.array(points)
print(points.shape)
m, c, low_slope, high_slope = sc.theilslopes(points[:, 1], points[:,0])
res = sc.spearmanr(points[:,0], points[:,1])
print('r y p para award vs gdp',res.statistic, res.pvalue)
#
X = [ min(points[:,0]), max(points[:,0]) ]
Y = [ m * X[0] + c, m * X[1] + c ]
ax.plot(X, Y, alpha=1, linestyle='--', color='red', label = '$\\rho = 0.35$\n$p-value=0.0015$')
ax.scatter(points[:,0], points[:, 1], s = 5**2, color = 'tab:blue')

#ax.set_ylim([4,14])
ax.set_xlabel('DCI design from Awards 2011-2023')
ax.set_ylabel('log mean GDP per capita PPA 2011-2023')
plt.legend()
plt.show()

plt.scatter(ECI_e, ECI_m)
plt.scatter(PCI_e, PCI_m)
plt.xlabel('Metodo de autovalores')
plt.ylabel('Metodo de las reflexiones')
plt.show()
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
