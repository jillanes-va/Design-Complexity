from Importacion import carga
import numpy as np

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

def distributividad(d, l):
    pass

def X_matrix(data):
    _, cantidad_elementos = domain_of_data(data)
    diccionarios = dictionaries(data)
    X_cpt = np.zeros(cantidad_elementos[0:3])
    Valores = data.values

    for N in range(len(Valores.shape[0])):
        index = ()
        X_cpt[  ]
    return X_cpt






