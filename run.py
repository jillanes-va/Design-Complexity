#from sympy.physics.control.control_plots import matplotlib

import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

# wipo_str = r'wipo_design.csv'
# wipo_columns = ['country_name','subclass_name', 'wipo_year_to', 'n']
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

X_cp = trat.Promedio_temporal(X_cpt, Awards= True)
#X_cp = X_cpt[:,:,0]

R_cp, M_cp, X_cp = calc.Matrices_ordenadas(X_cp, diccionaries)
print(M_cp.shape)
# figs.graf(np.log(X_cp + 1), xlabel = 'Categorias', ylabel = 'Paises', title = 'log-$X_{cp}$')
#
# figs.graf(np.log(R_cp + 1), xlabel = 'Categorias', ylabel = 'Paises', title = 'log-$RCA_{cp}$')
#
# figs.graf(M_cp, xlabel = 'Categorias', ylabel = 'Paises',title = '$M_{cp}$')
#
# k_0 =  calc.Complexity_measures(M_cp, 0)[0]
# k_1 =  calc.Complexity_measures(M_cp, 1)[0]
#
# plt.scatter(k_0, k_1)
# plt.xlabel('k_0')
# plt.ylabel('k_1')
# plt.show()


phi = calc.Similaridad(M_cp)


# ECI = calc.Z_transf(calc.Complexity_measures(M_cp, 18 )[0])
# PCI = calc.Z_transf(calc.Complexity_measures(M_cp, 18 )[1])
#
# num_paises = trat.inv_dict(diccionaries[0])
# num_cat = trat.inv_dict(diccionaries[1])
#
#
# ECI_paises = [ (ECI[n], num_paises[n]) for n in range(len(ECI)) ]
# PCI_cat = [ (PCI[n], num_cat[n]) for n in range(len(PCI)) ]
#
# llave = lambda A: A[0]
#
# ECI_paises = sorted(ECI_paises, key = llave, reverse = 1)
# PCI_cat = sorted(PCI_cat, key = llave, reverse = 1)



# with open('./data/results/awards/Ranking_ECI_Design_Awards.csv', 'w+', encoding = 'utf-8') as f:
#     f.writelines('ranking,country,ECI\n')
#     for n, data in enumerate(ECI_paises):
#         f.writelines(str(n + 1) + ',' + data[1] + ',' + str(data[0]) + '\n')
#
# with open('./data/results/awards/Ranking_PCI_Design_Awards.csv', 'w+', encoding = 'utf-8') as f:
#     f.writelines('ranking,product,PCI\n')
#     for n, data in enumerate(PCI_cat):
#         f.writelines(str(n + 1) + ',' + data[1] + ',' + str(data[0]) + '\n')
#
# plt.scatter([i for i in range(len(ECI))], [ECI_paises[i][0] for i in range(len(ECI))])
#
# plt.scatter([i for i in range(len(PCI))], [PCI_cat[i][0] for i in range(len(PCI))])
# plt.show()
#
# print(ECI_paises)
#
# figs.red(phi, PCI = PCI, diccionario = diccionaries, by_com = True, name = 'Espacio_productos_Comunidades', save = False, umbral_enlace = 0.4)
#
# figs.Clustering(phi, save = False)

dom_phi, relatedness = test.Relatedness_density_test(X_cpt, diccionaries, N_bins = 15)
figs.Density_plot(dom_phi, relatedness, xlabel = r'Relatedness density', ylabel = 'Probability of developing RCA in a design category', xlim_sup= 0.83, name = 'PrincipleOfRelatedness', save = False)

# str_ECI_wipo = r'results/wipo/Ranking_ECI_Design_wipo.csv'
# str_ECI_awards = r'results/awards/Ranking_ECI_Design_Awards.csv'
# str_awards_wipo_dict = r'dictionaries/dict_country_awards_wipo.csv'
#
# ECI_wipo = imp.carga(str_ECI_wipo, columnas_importantes = ['country', 'ECI'])
# ECI_awards = imp.carga(str_ECI_awards, columnas_importantes = ['country', 'ECI'])
# c_award_c_wipo = imp.carga(str_awards_wipo_dict, columnas_importantes = ['country_award','country_wipo'])
#
# print(dict(c_award_c_wipo.values))

# c_award_c_wipo = dict(imp.csv_to_list(str_awards_wipo_dict))
#
# ECI_horizontal = []
# ECI_vertical = []
#
#
# for pais_award, eci_award in ECI_awards.items():
#     try:
#         pais_wipo = c_award_c_wipo[ pais_award ]
#         y = ECI_wipo[pais_wipo]
#     except:
#         continue
#     ECI_vertical.append(y)
#     ECI_horizontal.append(eci_award)
#
# result = sc.linregress(ECI_horizontal, ECI_vertical)
#
# m = result.slope
# c = result.intercept
# r2 = result.rvalue**2
#
# X = [ min(ECI_horizontal), max(ECI_horizontal) ]
# Y = [ m * X[0] + c, m * X[1] + c ]
#
# print(len(ECI_horizontal), len(ECI_vertical))
#
# plt.scatter(ECI_horizontal, ECI_vertical)
# plt.plot(X, Y, alpha = 0.5, linestyle  = '--', color = 'red', label = f'$r^2 = {r2:.2f}$')
# plt.xlabel('ECI del Diseño Awards')
# plt.ylabel('ECI del Diseño WIPO 2010-2024')
# plt.legend()
# plt.show()