from sympy.physics.control.control_plots import matplotlib

import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

#wipo_str = r'wipo_design.csv'
#wipo_columns = ['country_name','subclass_name', 'wipo_year_to', 'n']

#awards_str = r'wrd_04_all-data.csv'
#awards_columns = ['designer_country', 'award_category', 'award_period' , 'award_score']

# Informacion = test.categorias_presentes(X_cpt[:,:,:12], diccionaries)
# plt.scatter([i[0] for i in Informacion], [i[1] for i in Informacion])
# plt.ylabel('Frecuencia absoluta')
# plt.title('Categorias presentes por año')
# plt.tight_layout()
# plt.xticks(rotation = 45)
# plt.show()

#
#sisi = datos_premio.groupby(['designer_country']).sum()
#print(sisi)
#print(sisi[sisi['award_score'] > 5.0])

# R_cp, M_cp, X_cp = calc.Matrices_ordenadas(X_cp, diccionaries, 1, 2 )
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
#
#
# phi = calc.Similaridad(M_cp)
# omega_cp = calc.Similarity_Density(R_cp)
#
# ECI = calc.Z_transf(calc.Complexity_measures(M_cp, 2 * 9)[0])
# PCI = calc.Z_transf(calc.Complexity_measures(M_cp, 2 * 9)[1])
#
# num_paises = trat.inv_dict(diccionaries[0])
# num_cat = trat.inv_dict(diccionaries[1])
#
#
# ECI_paises = [ (ECI[n], num_paises[n]) for n in range(len(ECI)) ]
# PCI_cat = [ (PCI[n], num_cat[n]) for n in range(len(PCI)) ]
#
# ECI_paises = sorted(ECI_paises, key=lambda A: A[0], reverse=1)
# PCI_cat = sorted(PCI_cat, key=lambda A: A[0], reverse=1)
#
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
# plt.scatter([i for i in range(len(ECI))], ECI)
# plt.title('Indice de Complejidad Economica')
# plt.show()
#
# plt.scatter([i for i in range(len(PCI))], PCI)
# plt.title('Indice de Complejidad Economica de los Productos')
# plt.show()
#
#
# figs.red(phi, PCI = PCI, diccionario = diccionaries, by_com = True, name = 'Espacio_productos_Comunidades', save = False, umbral_enlace = 0.4)
#
# figs.Clustering(phi, save = False)

#dom_phi, relatedness = test.Relatedness_density_test(X_cpt, diccionaries, N_bins = 15)
#figs.Density_plot(dom_phi, relatedness, xlabel = r'Densidad de similitud', ylabel = 'Probabilidad de transicionar en alguna categoría', xlim_sup= 0.83, name = 'PrincipleOfRelatedness', save = False)

str_ECI_wipo = r'./data/results/wipo/Ranking_ECI_Design_wipo.csv'
str_ECI_awards = r'./data/results/awards/Ranking_ECI_Design_Awards.csv'
str_awards_wipo_dict = r'./data/dictionaries/dict_country_awards_wipo.csv'

ECI_wipo = dict(imp.csv_to_list(str_ECI_wipo))
ECI_awards = dict(imp.csv_to_list(str_ECI_awards))

c_award_c_wipo = list(imp.csv_to_dict(str_awards_wipo_dict).items())

ECI_horizontal = []
ECI_vertical = []

print(c_award_c_wipo)
for dupla in c_award_c_wipo:
    try:
        ECI_vertical.append(ECI_wipo[dupla[1]])
        ECI_horizontal.append( ECI_awards[ dupla[0] ] )

    except:
        pass

result = sc.linregress(ECI_horizontal, ECI_vertical)

m = result.slope
c = result.intercept
r = result.rvalue

X = [ min(ECI_horizontal), max(ECI_horizontal) ]
Y = [ m * X[0] + c, m * X[1] + c ]

print(r**2)
plt.scatter(ECI_horizontal, ECI_vertical)
plt.plot(X, Y, alpha = 0.5, linestyle  = '--', color = 'red', label = r'$r^2 = 0.21$')
plt.xlabel('ECI del Diseño Awards')
plt.ylabel('ECI del Diseño WIPO 2010-2024')
plt.legend()
plt.show()