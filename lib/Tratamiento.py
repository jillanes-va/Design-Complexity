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

def transitividad(dict_1, dict_2):
    '''Crea una lista que asocia las llaves del primer diccionario con los valores del diccionario 3'''
    dict_3 = dict({})
    for llave, valor in dict_1.items():
        dict_3[llave] = dict_2[valor]
    return dict_3

def re_count(diccionario):
    new_dict = {}
    for n, keys in enumerate(diccionario.keys()):
        new_dict[keys] = n
    return new_dict

def inv_dict(diccionario, unique = True):
    '''Toma un diccionario e invierte el mapeo de llaves a valores'''
    inv_map = {}
    if unique:
        for k, v in diccionario.items():
            inv_map[v] = k
    else:
        for k, v in diccionario.items():
            inv_map[v] = inv_map.get(v, []) + [k]
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
            return X.sum(axis = 2)/ n_time
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


def gdp_matrix(data, last = False):
    matriz = data.values
    if last:
        return matriz[:,-1]
    else:
        return matriz[:,1:]

def sum_files(X, partida_llegada, cat_num):
    X_nuevo = np.copy(X)
    paises_partida = partida_llegada.keys()
    paises_llegada = list(set(partida_llegada.values()))

    llegada_partida = inv_dict(partida_llegada, unique = False)
    num_cat = inv_dict(cat_num)
    N, M = len(paises_partida), len(paises_llegada)

    index_array = [[cat_num[pais_partida] for pais_partida in llegada_partida[ pais_llegada ]] for pais_llegada in paises_llegada]
    super_indice = []

    for indices in index_array:
        sub_suma = 0
        i_min = np.min(indices)
        for i in indices:
            sub_suma += X[i, :]
            if i != i_min:
                super_indice += [i]
        X_nuevo[i_min, :] = sub_suma
    np.delete(X_nuevo, super_indice)

    for indices in super_indice:
        _ = cat_num.pop(num_cat[indices])
    new_dict = re_count(cat_num)
    return X_nuevo, new_dict