import lib.Importacion as imp
import lib.Tratamiento as trat
import lib.Calculo as calc
import lib.Testeo_estadistico as test
import lib.Figuras as figs

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

from lib.Importacion import carga_excel

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
datos_gdp = carga_excel('IMF_GDP_per_PPA.xls')
datos_gdp['media'] = datos_gdp.mean(axis = 1, numeric_only = True)

diccionaries_gdp = trat.dictionaries(datos_gdp)[0]
gdp_by_country = trat.gdp_matrix(datos_gdp, last = 0)


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


#phi = calc.Similaridad(M_cp)


ECI = calc.Z_transf(calc.Complexity_measures(M_cp, 18 )[0])
PCI = calc.Z_transf(calc.Complexity_measures(M_cp, 18 )[1])
#
num_paises = trat.inv_dict(diccionaries[0])
num_cat = trat.inv_dict(diccionaries[1])
#
#
paises_ECI = { num_paises[n]:ECI[n] for n in range(len(ECI)) }
cat_PCI = { num_cat[n]: PCI[n] for n in range(len(PCI)) }

importantes = datos_gdp.loc[:, ['GDP per capita, current prices\n (U.S. dollars per capita)', 'media']]
paises_GDP = {tupla[0]:tupla[1] for tupla in importantes.values}

country_mapping = {
    "United States": "United States",
    "Italy": "Italy",
    "United Kingdom": "United Kingdom",
    "Germany": "Germany",
    "Turkey": "Türkiye, Republic of",
    "South Korea": "Korea, Republic of",
    "Canada": "Canada",
    "Hungary": "Hungary",
    "Russia": "Russian Federation",
    "Spain": "Spain",
    "Australia": "Australia",
    "Netherlands": "Netherlands",
    "India": "India",
    "Czechia": "Czech Republic",
    "Iran": "Iran",
    "France": "France",
    "Israel": "Israel",
    "Brazil": "Brazil",
    "Singapore": "Singapore",
    "Portugal": "Portugal",
    "Austria": "Austria",
    "Hong Kong": "Hong Kong SAR",
    "Switzerland": "Switzerland",
    "Belgium": "Belgium",
    "Denmark": "Denmark",
    "Sweden": "Sweden",
    "Thailand": "Thailand",
    "China": "China, People's Republic of",
    "Egypt": "Egypt",
    "Lithuania": "Lithuania",
    "Poland": "Poland",
    "Finland": "Finland",
    "Serbia": "Serbia",
    "Greece": "Greece",
    "Japan": "Japan",
    "Mexico": "Mexico",
    "Bulgaria": "Bulgaria",
    "Ireland": "Ireland",
    "Ukraine": "Ukraine",
    "Romania": "Romania",
    "United Arab Emirates": "United Arab Emirates",
    "Latvia": "Latvia",
    "New Zealand": "New Zealand",
    "Argentina": "Argentina",
    "Croatia": "Croatia",
    "Malaysia": "Malaysia",
    "Vietnam": "Vietnam",
    "Slovenia": "Slovenia",
    "Taiwan": "Taiwan Province of China",
    "Chinese Taipei": "Taiwan Province of China",
    "Indonesia": "Indonesia",
    "Philippines": "Philippines",
    "Slovakia": "Slovak Republic",
    "Saudi Arabia": "Saudi Arabia",
    "Chile": "Chile",
    "Estonia": "Estonia",
    "Norway": "Norway",
    "South Africa": "South Africa",
    "Jordan": "Jordan",
    "Lebanon": "Lebanon",
    "Peru": "Peru",
    "Colombia": "Colombia",
    "Cyprus": "Cyprus",
    "Guatemala": "Guatemala",
    "Kazakhstan": "Kazakhstan",
    "Qatar": "Qatar",
    "Armenia": "Armenia",
    "Georgia": "Georgia",
    "Iceland": "Iceland",
    "Moldova": "Moldova",
    "Pakistan": "Pakistan",
    "Belarus": "Belarus",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Bangladesh": "Bangladesh",
    "Dominican Republic": "Dominican Republic",
    "Ecuador": "Ecuador",
    "Kuwait": "Kuwait",
    "Uruguay": "Uruguay",
    "Macau (China)": "Macao SAR",
}

points = []
for c_awards, c_gdp in country_mapping.items():
    points.append(
        [paises_ECI[c_awards], paises_GDP[c_gdp]]
    )

points = np.array(points)
# (pais_award, ECI) -> (pais_award, pais_gdp) -> (pais_gdp, GDP)

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

#dom_phi, relatedness = test.Relatedness_density_test(X_cpt, diccionaries, N_bins = 15)
#figs.Density_plot(dom_phi, relatedness, xlabel = r'Relatedness density', ylabel = 'Probability of developing RCA in a design category', xlim_sup= 0.83, name = 'PrincipleOfRelatedness', save = False)


ECI_d = points[:, 0]
log_GDP = np.log(points[:, 1])

m, c, low_slope, high_slope = sc.theilslopes(log_GDP, ECI_d)
#
X = [ min(ECI_d), max(ECI_d) ]
Y = [ m * X[0] + c, m * X[1] + c ]
#
# print(len(ECI_horizontal), len(ECI_vertical))
#
plt.scatter(ECI_d, log_GDP)
plt.plot(X, Y, alpha = 0.5, linestyle  = '--', color = 'red')
plt.xlabel('ECI del Diseño Awards 2011-2023')
plt.ylabel('log Promedio PIB per capita PPA 2011-2023')
plt.savefig('./figs/log_PIB_vs_ECI_awards.pdf')

awards_WIPO = imp.dictionary_from_csv(r'dictionaries/dict_country_awards_wipo.csv')
ECI_d_awards = imp.dictionary_from_csv(r'results/awards/Ranking_ECI_Design_Awards.csv', ranking = True)
ECI_d_wipo = imp.dictionary_from_csv(r'results/wipo/Ranking_ECI_Design_wipo.csv', ranking = True)

print(len(awards_WIPO), len(ECI_d_awards), len(ECI_d_wipo))
points = []
for award, wipo in awards_WIPO.items():
    try:
        points.append(
            [
                ECI_d_awards[award],
                ECI_d_wipo[wipo]
            ]
        )
    except:
        pass

points = np.array(points)
ECI_d_awards = points[:, 0]
ECI_d_wipo = points[:, 1]

m, c, low_slope, high_slope = sc.theilslopes(ECI_d_awards, ECI_d_wipo)
#
X = [ min(ECI_d_wipo), max(ECI_d_wipo) ]
Y = [ m * X[0] + c, m * X[1] + c ]
#
# print(len(ECI_horizontal), len(ECI_vertical))
#
plt.scatter(ECI_d_wipo, ECI_d_awards)
plt.plot(X, Y, alpha = 0.5, linestyle  = '--', color = 'red')
plt.xlabel('ECI del Diseño Awards 2011-2023')
plt.ylabel('ECI de la WIPO 2010-2024')
plt.savefig('./figs/Regresion_ECI_diseño_award_wipo.pdf')