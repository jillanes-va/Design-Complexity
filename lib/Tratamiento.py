import numpy as np

def domain_of_data(data):
    '''TO DO'''
    lista_de_elementos = []
    for index in list(data):
        lista_de_elementos.append(sorted(list(set(data[index]))))
    return lista_de_elementos, [len(lista) for lista in lista_de_elementos]

#¿Debería unir a una sola función dictionaries y domain_of_data?

def dictionaries(data):
    '''Genera los diccionarios para asociar los elementos a números'''
    lista_de_elementos, cantidad_de_elementos = domain_of_data(data)
    diccionarios = []
    for index in range(len(cantidad_de_elementos)):
        diccionarios.append(dict( [ (lista_de_elementos[index][n], n) for n in range(cantidad_de_elementos[index])] ))
    return diccionarios

def distributividad(d : list, l : list):
    '''Función que distribuye una lista con keys a una lista con diccionarios y retorna los valores de los diccionarios en una tupla.'''
    assert len(d) == len(l), 'La lista debe tener la misma cantidad de elementos que el diccionario.'
    distribuido = tuple(d[n][l[n]] for n in range(len(l)))
    return distribuido

def X_matrix(data, time = True):
    '''Toma los datos y arma una matriz de volumen con índices 'pais', 'producto' y 'año' '''
    _, numero_de_cosas = domain_of_data(data)
    diccionarios = dictionaries(data)[0:2 + time]
    X_cpt = np.zeros(numero_de_cosas[0:2 + time])
    for number in data.index:
        data_number = list(data.loc[number])
        index = distributividad(diccionarios, data_number[:2 + time] )
        X_cpt[index] += data_number[2 + time]
    return X_cpt




