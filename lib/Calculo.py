import numpy as np

def Matrices(X):
    '''Función que toma una matriz de volumen de país-producción y devuelve la RCA y la matriz de especialización binaria'''
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

    M = np.ones((c_len, p_len)) * (RCA >= 1)
    return RCA, M


def Matrices_ordenadas(X, lista_de_cosas, diccionario):
    '''Funcion que toma una matriz de especialización y la reordena por ubicuidad... y entrega la matriz reordenada, con el diccionario correspondiente'''
    RCA, M = Matrices(X)
    M_p = np.sum(M, axis=0)
    M_c = np.sum(M, axis=1)

    N_c, N_p = M.shape

    lista_p = [(M_p[i], i) for i in range(N_p)]
    lista_c = [(M_c[i], i) for i in range(N_c)]

    Shuffle_p = sorted(lista_p, key=lambda A: A[0], reverse=1)
    Shuffle_c = sorted(lista_c, key=lambda A: A[0], reverse=1)

    M_ordenada = np.zeros(M.shape)
    RCA_ordenada = np.zeros(RCA.shape)

    for i in range(N_c):
        for j in range(N_p):
            M_ordenada[i, j] = M[Shuffle_c[i][1], Shuffle_p[j][1]]
            RCA_ordenada[i, j] = RCA[Shuffle_c[i][1], Shuffle_p[j][1]]

    # Nuevo_dict_num_p = dict([(i, lista_de_cosas[1][Shuffle_p[i][1]]) for i in range(N_p)])
    Nuevo_dict_p_num = dict([(lista_de_cosas[1][Shuffle_p[i][1]], i) for i in range(N_p)])
    # Nuevo_dict_num_c = dict([(i, lista_de_cosas[0][Shuffle_c[i][1]]) for i in range(N_c)])
    Nuevo_dict_c_num = dict([(lista_de_cosas[0][Shuffle_c[i][1]], i) for i in range(N_c)])

    diccionario[0] = Nuevo_dict_c_num
    diccionario[1] = Nuevo_dict_p_num

    return RCA_ordenada, M_ordenada, diccionario

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

def Complexity_measures(M, n):
    '''Toma la matriz de especialización binaria y aplica el metodo de las reflexiones n veces devolviendo el vector de las iteración de las localidades y los productos'''
    M_c = np.sum(M, axis=1)
    M_p = np.sum(M, axis=0)

    K_c = M_c
    K_p = M_p
    for i in range(n):
        K_c = np.matmul(M, K_p) / M_c
        K_p = np.matmul(M.T, K_c) / M_p
    return K_c, K_p

def Z_transf(K):
    '''Aplica la transformada Z sobre un vector K'''
    return (K - np.mean(K)) / np.std(K)
