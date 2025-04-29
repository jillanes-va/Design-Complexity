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

def Promedio_temporal(X, total_time = 1):
    '''
    Args:
        X : np array. Matriz de volumen de producción
        total_time: int. Tiempo total
    Returns:
        X_new: np.array. Promedio temporal sobre el 3er indice
    '''
    if total_time == 1:
        total_time = X.shape[2]
    new_X = X.sum(axis = 2) / total_time

    return new_X

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


def gdp_matrix(data):
    #Diccionarios
    dicc_ctry_num = dictionaries(data)[0]

    #Data
    valores_originales = data.values
    c, n = valores_originales.shape

    #Nueva data
    valores_nuevos = np.zeros((c, n - 1))

    for i in range(c):
        cntry = valores_originales[i, 0]
        num_of_cntry = dicc_ctry_num[cntry]
        valores_nuevos[num_of_cntry, :] = valores_originales[i, 1:]
    return valores_nuevos

def sum_files(X, diccionaries, partida_llegada):
    '''
    Args:
        X : np.array. Matriz representando filas los países y columnas los productos.
        diccionaries: list[dict]. Lista con todos los diccionarios de los indices de X.
        partida_llegada: dict. Diccionario en donde clasifica los elementos de llegada o partida.

    Returns:
        X_nuevo: np.array. Matriz combinando los valores (sumandolos) que especificaba partida_llegada.
        new_dict: dict. Nuevo diccionario desechando los países y reindexando la matriz X_nuevo.
    '''
    if len(partida_llegada) == 0:
        return X
    else:
        cat_num = diccionaries[0]
        X_nuevo = np.copy(X)
        paises_partida = partida_llegada.keys()
        paises_llegada = list(set(partida_llegada.values()))
        paises_eliminados = list(set(paises_partida) - set(paises_llegada))

        num_cat = inv_dict(cat_num)

        num_partida = [cat_num[pais] for pais in paises_partida]
        num_llegada = [cat_num[pais] for pais in paises_llegada]
        num_eliminado = [cat_num[pais] for pais in paises_eliminados]

        llegada_partida = inv_dict(partida_llegada, unique = False)
        indexacion = []
        for pais_llegada in paises_llegada:
            intermedio = []
            for paises in llegada_partida[pais_llegada]:
                intermedio.append(
                    cat_num[paises]
                )
            indexacion.append(intermedio)
        for index in indexacion:
            new_row = X_nuevo[index].sum(axis = 0)
            first_country = partida_llegada[num_cat[index[0]]]
            definitive_index = cat_num[ first_country ]
            X_nuevo[definitive_index, :] = new_row

        X_nuevo = np.delete(X_nuevo, num_eliminado, axis = 0)
        for n in num_eliminado:
            num_cat.pop(n)
        cat_num = inv_dict(num_cat)
        diccionaries[0] = re_count(cat_num)
        return X_nuevo