import numpy as np
import matplotlib.pyplot as plt

import lib.Calculo as calc
import lib.Tratamiento as trat

def Trans_matrix(X, threeshold = 0.5, n_t = None, Awards = True):
    '''Toma la matriz de RCA y retorna dos matrices que reportan aquellos productos que son de transición o no, dependiendo de un umbral.'''
    C, P, T = X.shape
    Transicion = np.zeros((C, P))
    Intransicion = np.zeros((C, P))

    if n_t is None:
        n_t = T//2

    if Awards:
        X_0 = X[:, :, :n_t].sum(axis = 2) / (n_t + 1)
        X_1 = X[:, :, n_t + 1:].sum(axis = 2) / (T - n_t)
    else:
        X_0 = X[:, :, n_t] /5
        X_1 = X[:, :, n_t + 1] /5

    R_cp_0, _, _ = calc.Matrices(X_0, cleaning = False)
    R_cp_1, _, _ = calc.Matrices(X_1, cleaning = False)

    for c in range(C):
        for p in range(P):
            if R_cp_0[c,p] <= threeshold:
                if R_cp_1[c,p] >=1:
                    Transicion[c,p] = 1
                if R_cp_1[c,p] <=threeshold:
                    Intransicion[c,p] = 1
    return Transicion, Intransicion

def Relatedness_density_test(X, diccionario= None, threeshold = 0.5, n_t = None, N_bins = 50, Awards = True):
    '''Testea la similaridad en productos de transición y de intransición. No retorna nada, solo grafica la similaridad vs la frecuencia relativa.'''
    X_total = trat.Promedio_temporal(X, Awards = Awards)
    if Awards:
        RCA, M_cp, X_total = calc.Matrices(X_total, cleaning = False)
    else:
        RCA, M_cp, X_total = calc.Matrices(X_total[:,:,:1].sum(axis =2)/10, cleaning=False)
    phi_0 = calc.Similaridad(M_cp)
    Transicion, _ = Trans_matrix(X, threeshold, n_t, Awards)
    omega_cp_t = np.nan_to_num( np.matmul(M_cp, phi_0) * Transicion / np.sum(phi_0, axis = 0) )

    cantidad_paises, cantidad_categorias = RCA.shape

    Ocurrencias_t = np.zeros(N_bins)
    Total = np.zeros(N_bins)
    Prob_t = np.zeros(N_bins)

    for c in range(cantidad_paises):
        for p in range(cantidad_categorias):

            phi_max = (M_cp[c, :] * phi_0[p, :]).max()
            rebanada = int(np.floor(phi_max * N_bins))
            if rebanada == N_bins:
                rebanada += -1
            Total[rebanada] += 1

            if Transicion[c, p] == 1:
                Ocurrencias_t[rebanada] += 1

    for i in range(N_bins):
        if Total[i] ==0:
            Prob_t[i] = 0
        else:
            Prob_t[i] = Ocurrencias_t[i]/Total[i]
    dom_phi = np.linspace(0, 1, N_bins)
    return dom_phi, Prob_t

def categorias_presentes(X, diccionario):
    '''Para una matriz país-producto-año retorna los productos no registrados en un año'''

    C, P, T = X.shape
    diccionario_inv = [trat.inv_dict(d) for d in diccionario]

    anios = []
    for t in range(T):
        lista_cat = []
        cantidad = 0
        año = diccionario_inv[2][t]
        for p in range(P):
            if X[:, p, t].sum() != 0:
                cantidad += 1
                lista_cat.append(diccionario_inv[1][p])
        informacion = (año, cantidad, lista_cat)
        anios.append(informacion)
    return anios

def punteo_especifico(X, Y, dict_X_num, dict_Y_num, dict_X_Y, dict_X_short = None):
    '''
    Args:
        X: numpy array. Contiene valores de elementos probablemente pareados con Y.
        Y: numpy array. Contiene valores de eleementos probablemente pareados con X.
        dict_X_num: dict. Parea los elementos hacia su índice para X.
        dict_Y_num: dict. Parea los elementos hacia su índice para Y.
        dict_X_Y: dict. Parea los elementos de X hacia Y.
        dict_X_short: dict. Parea elementos de X hacia un acronimo.
    Returns:
        tuple[numpy array, list]. Una tupla con los puntos (X,Y) pareados y una lista con el nombre de los elementos de X.
    '''
    lista_X_incluidos = []
    puntos = []
    for X_name, X_num in dict_X_num.items():
        try:

            Y_name = dict_X_Y[X_name]
            Y_num = dict_Y_num[Y_name]

            X_value = X[X_num]
            Y_value = Y[Y_num]

            if X_value != np.nan and Y_value != np.nan:
                puntos.append(np.array([
                    X[X_num], Y[Y_num]
                ]))

                lista_X_incluidos.append(X_name)
        except:
            pass

    if dict_X_short is not None:
        N = len(lista_X_incluidos)
        for i in range(N):
            lista_X_incluidos[i] = dict_X_short[
                lista_X_incluidos[i]
            ]
    return np.array(puntos), lista_X_incluidos
