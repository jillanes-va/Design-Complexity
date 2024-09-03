from Importacion import carga
import numpy as np
import matplotlib.pyplot as plt

def domain_of_data(data):
    '''TO DO'''
    lista_de_elementos = []
    for index in list(data):
        lista_de_elementos.append(sorted(list(set(data[index]))))
    return lista_de_elementos, [len(lista) for lista in lista_de_elementos]

#¿Debería unir a una sola función dictionaries y domain_of_data?

def dictionaries(data):
    '''TO DO'''
    lista_de_elementos, cantidad_de_elementos = domain_of_data(data)
    diccionarios = []
    for index in range(len(cantidad_de_elementos)):
        diccionarios.append(dict( [ (lista_de_elementos[index][n], n) for n in range(cantidad_de_elementos[index])] ))
    return diccionarios

def distributividad(d : list, l : list):
    assert len(d) == len(l), 'La lista debe tener la misma cantidad de elementos que el diccionario.'
    distribuido = tuple(d[n][l[n]] for n in range(len(l)))
    return distribuido

def X_matrix(data):
    _, numero_de_cosas = domain_of_data(data)
    diccionarios = dictionaries(data)[0:3]
    X_cpt = np.zeros(numero_de_cosas[0:3])
    for number in data.index:
        data_number = list(data.loc[number])
        index = distributividad(diccionarios, data_number[:3] )
        X_cpt[index] += data_number[3]
    return X_cpt

nombre_archivo = r'wrd_04_all-data.csv'
columnas = ['designer_country', 'award_category', 'award_period', 'award_score']

datos = carga(nombre_archivo, columnas)
X_cp = np.sum(X_matrix(datos), axis = 2)/ 13
plt.imshow(np.log(X_cp + 1), interpolation = 'nearest', cmap = 'afmhot')
plt.show()




