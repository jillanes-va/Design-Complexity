import numpy as np
import lib.Tratamiento as trat

def Limpieza(X, diccionarios, c_min, p_min, time = False):
    '''Limpia los paises que esten bajo un cierto umbral, elimina de la lista aquellos que no cumplan tal motivo'''
    if time:
        X = X.sum(axis = 2)
    Mask_c = (X.sum(axis = 1) > c_min)
    Mask_p = (X.sum(axis = 0) > p_min)

    for n, valor in enumerate(Mask_c):
        if not valor:
            diccionarios[0].pop(trat.inv_dict(diccionarios[0])[n])
    for n, valor in enumerate(Mask_p):
        if not valor:
            diccionarios[1].pop(trat.inv_dict(diccionarios[1])[n])

    diccionarios[0] = trat.re_count(diccionarios[0])
    diccionarios[1] = trat.re_count(diccionarios[1])
    return X[:, Mask_p][Mask_c, :]


def Matrices(X, diccionario = None, threshold = 1, c_min = 1, p_min = 1, time = False, cleaning = True):
    '''Función que toma una matriz de volumen de país-producción y devuelve la RCA y la matriz de especialización binaria'''
    if cleaning:
        X = Limpieza(X, diccionario, c_min, p_min, time)

    c_len, p_len = X.shape
    RCA = np.zeros(X.shape)
    M = np.zeros(X.shape)

    alpha = np.zeros(X.shape)
    beta = np.zeros(X.shape)

    X_c = np.sum(X, axis=1)
    X_p = np.sum(X, axis=0)
    X_net = np.sum(X)

    for i in range(c_len):
        for j in range(p_len):
            if X_c[i] != 0:
                alpha[i, j] = X[i, j] / X_c[i]
            beta[i, j] = X_p[j] / X_net
            if X_p[j] != 0:
                RCA[i, j] = alpha[i, j] / beta[i, j]

    M = 1 * (RCA >= threshold)
    return RCA, M, X


def Matrices_ordenadas(X, diccionario, c_min = -1, p_min = -1, threshold = 1, change_dict = True, time = False):
    '''Funcion que toma una matriz de especialización y la reordena por ubicuidad... y entrega la matriz reordenada, con el diccionario correspondiente'''
    RCA, M, X = Matrices(X, diccionario, threshold, c_min, p_min, time)
    M_p = np.sum(M, axis=0)
    M_c = np.sum(M, axis=1)

    N_c, N_p = X.shape

    lista_p = [(M_p[i], i) for i in range(N_p)]
    lista_c = [(M_c[i], i) for i in range(N_c)]

    llave = lambda A: A[0]

    Shuffle_p = sorted(lista_p, key= llave, reverse = 1)
    Shuffle_c = sorted(lista_c, key= llave, reverse = 1)

    arrays = [RCA, M, X]
    array_1 = [np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)]
    array_2 = [np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape)]

    for i in range(3):
        for j in range(N_c):
            array_1[i][j, :] = arrays[i][ Shuffle_c[j][1], : ]
    for i in range(3):
        for j in range(N_p):
            array_2[i][:, j] = array_1[i][:, Shuffle_p[j][1]]

    if change_dict:
        llave_c = list(diccionario[0].keys())
        llave_p = list(diccionario[1].keys())
        Nuevo_dict_c_num = dict([(llave_c[Shuffle_c[i][1]], i) for i in range(N_c)])
        Nuevo_dict_p_num = dict([(llave_p[Shuffle_p[i][1]], i) for i in range(N_p)])

        diccionario[0] = Nuevo_dict_c_num
        diccionario[1] = Nuevo_dict_p_num

    return array_2

def Similaridad(M):
    '''De una matriz de especialización binaria, obtiene la metrica de similaridad definida en Hidalgo et al 2009 entre actividades. Mantiene la diagonal igual a cero.'''
    c_len, p_len = M.shape
    phi = np.zeros((p_len, p_len))
    ubicuidad = np.sum(M, axis=0)  # p

    for p in range(p_len):
        for q in range(p_len):
            Maximo = np.max([ubicuidad[p], ubicuidad[q]])
            if p != q and Maximo != 0:
                S = 0
                for c in range(c_len):
                    S += M[c, p] * M[c, q]
                phi[p, q] = S / Maximo
    return phi
def Similarity_Density(RCA):
    '''Toma la matriz RCA y calcula su Densidad de Similaridad calculando la Similaridad y matriz M_cp'''
    M_cp = 1 * (RCA >= 1)
    phi = Similaridad(M_cp)
    Num = np.matmul(M_cp, phi) / np.sum(phi, axis = 0)
    return Num

def Complexity_measures(M_cp, n):
    '''Toma la matriz de especialización binaria y aplica el metodo de las reflexiones n veces devolviendo el vector de las iteración de las localidades y los productos'''
    k_c0 = np.sum(M_cp, axis=1)
    k_p0 = np.sum(M_cp, axis=0)
    C, P = len(k_c0), len(k_p0)

    k_cN = k_c0
    k_pN = k_p0
    for _ in range(n):
        for c in range(C):
            k_cN[c] = (1/k_c0[c]) * np.sum( M_cp[c,:] * k_pN )
        for p in range(P):
            k_pN[p] = (1 / k_p0[p]) * np.sum(M_cp[:, p] * k_cN)
    return k_cN, k_pN

def Z_transf(K):
    '''Aplica la transformada Z sobre un vector K'''
    return (K - np.mean(K)) / np.std(K)

