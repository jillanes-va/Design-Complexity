import numpy as np

def domain_of_data(data):
    '''Toma los datos y retorna dos listas, la primera con todos los elementos existentes en cada columna y la segunda lista devuelve la cantidad de elementos existentes.'''
    lista_de_elementos = []
    for index in list(data):
        lista_de_elementos.append(sorted(list(set(data[index]))))
    return lista_de_elementos, [len(lista) for lista in lista_de_elementos]

#¿Debería unir a una sola función dictionaries y domain_of_data?

def dictionaries(data):
    '''Genera los diccionarios para asociar los elementos a números. Lista de diccionarios***'''
    lista_de_elementos, cantidad_de_elementos = domain_of_data(data)
    diccionarios = []
    for index in range(len(cantidad_de_elementos)):
        diccionarios.append(dict( [ (lista_de_elementos[index][n], n) for n in range(cantidad_de_elementos[index])] ))
    return diccionarios

def re_count(diccionario):
    new_dict = {}
    for n, keys in enumerate(diccionario.keys()):
        new_dict[keys] = n
    return new_dict

def inv_dict(diccionario):
    '''Toma un diccionario e invierte el mapeo de llaves a valores'''
    inv_map = {}
    for k, v in diccionario.items():
        inv_map[v] = k
    return inv_map
def distributividad(d : list, l : list):
    '''Función que distribuye una lista con keys a una lista con diccionarios y retorna los valores de los diccionarios en una tupla.'''
    assert len(d) == len(l), 'La lista debe tener la misma cantidad de elementos que el diccionario.'
    distribuido = tuple(d[n][l[n]] for n in range(len(l)))
    return distribuido

def X_matrix(data):
    '''Toma los datos y arma una matriz de volumen con índices 'pais', 'producto' y 'año' '''
    _, numero_de_cosas = domain_of_data(data)
    Dim = len(numero_de_cosas) - 1
    diccionarios = dictionaries(data)[:Dim]
    X_cpt = np.zeros(numero_de_cosas[:Dim])

    for number in data.index:
        data_number = list(data.loc[number])
        index = distributividad(diccionarios, data_number[:Dim])
        X_cpt[index] += data_number[-1]
    return X_cpt

def Promedio_temporal(X, n_time = None, Awards = True):
    '''Toma la matriz y le realiza el promedio temporal'''
    if Awards:
        if n_time is None:
            _, _, T = X.shape
            return X.sum(axis = 2) / T
        if isinstance(n_time, int) or isinstance(n_time, float):
            return X/ n_time
        else:
            raise TypeError
    else:
        return X/5

def pareo_listas(lista_a, lista_b):
    '''Toma dos listas de strings y entrega dos listas tal que si un string de A es contenido (parcialmente) por un string de B, se guarden en listas distinas pero pareads, entrega ademas aquellos strings sobrantes.'''
    lista_1 = lista_a.copy()
    lista_2 = lista_b.copy()

    nueva_lista_1 = []
    nueva_lista_2 = []

    i = 0

    while i <= len(lista_1):
        j = 0
        while j < len(lista_2):
            if not (lista_1[i] in lista_2[j]):
                j += 1
            else:
                nueva_lista_1.append(lista_1.pop(i))
                nueva_lista_2.append(lista_2.pop(j))
                break
        if j == len(lista_2):
            i += 1

    return [nueva_lista_1, nueva_lista_2], [lista_1, lista_2]
