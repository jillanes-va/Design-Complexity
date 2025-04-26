import numpy as np
import lib.Tratamiento as trat

def Limpieza(X, diccionarios, c_min = 0, p_min = 0):
    '''
    Args:
        X: numpy array. Es una matriz de volumen (c, p, t).
        diccionarios: list[dict]. Son los diccionarios de (c, p, t).
        c_min: int. Umbral mínimo para que un país tenga producción.
        p_min: int. Umbral mínimo para que un producto tenga producción.
    '''

    X_sum = np.copy(X).sum(axis = 2)
    Mask_c = (X_sum.sum(axis = 1) > c_min)
    Mask_p = (X_sum.sum(axis = 0) > p_min)

    for n, valor in enumerate(Mask_c):
        if not valor:
            diccionarios[0].pop(trat.inv_dict(diccionarios[0])[n])
    for n, valor in enumerate(Mask_p):
        if not valor:
            diccionarios[1].pop(trat.inv_dict(diccionarios[1])[n])

    diccionarios[0] = trat.re_count(diccionarios[0])
    diccionarios[1] = trat.re_count(diccionarios[1])

    return X[:, Mask_p][Mask_c, :]


def Matrices(X_cpt, threshold = 1):
    '''Función que toma una matriz de volumen de país-producción y devuelve la RCA y la matriz de especialización binaria'''
    c_len, p_len = X_cpt.shape

    RCA = np.zeros(X_cpt.shape)
    alpha = np.zeros(X_cpt.shape)
    beta = np.zeros(X_cpt.shape)

    X_c = np.sum(X_cpt, axis = 1)
    X_p = np.sum(X_cpt, axis = 0)
    X_net = np.sum(X_cpt)

    for i in range(c_len):
        for j in range(p_len):
            if X_c[i] != 0:
                alpha[i, j] = X_cpt[i, j] / X_c[i]
            beta[i, j] = X_p[j] / X_net
            if X_p[j] != 0:
                RCA[i, j] = alpha[i, j] / beta[i, j]

    M = 1 * (RCA >= threshold)
    return RCA, M

def shuffles(RCA):
    c_len, p_len = RCA.shape

    R_p = np.sum(RCA, axis = 0)
    R_c = np.sum(RCA, axis = 1)

    lista_c = [(R_c[i], i) for i in range(c_len)]
    lista_p = [(R_p[i], i) for i in range(p_len)]

    llave = lambda A: A[0]

    shuffle_c = sorted(lista_c, key=llave, reverse=1)
    shuffle_p = sorted(lista_p, key=llave, reverse=1)
    return shuffle_c, shuffle_p

def array_shuffler(A, shuffle_c, shuffle_p):
    c_len, p_len = A.shape
    array_ex, array_ante = np.zeros((c_len, p_len)), np.zeros((c_len, p_len))
    for i in range(c_len):
        array_ex[i, :] = A[shuffle_c[i][1], :]
    for j in range(p_len):
        array_ante[:, j] = array_ex[:, shuffle_p[j][1]]
    return array_ante


def Matrices_ordenadas(X_cpt, diccionario, total_time, c_min = 0, p_min = 0, threshold = 1, cleaning = True):
    '''Funcion que toma una matriz de especialización y la reordena por ubicuidad... y entrega la matriz reordenada, con el diccionario correspondiente'''
    if cleaning:
        X_cpt = Limpieza(X_cpt, diccionario, c_min, p_min)

    c_len, p_len, t_len = X_cpt.shape
    X_cp = trat.Promedio_temporal(X_cpt, total_time = total_time)

    RCA, M = Matrices(X_cp, threshold)

    shuffle_c, shuffle_p = shuffles(M)

    X_, RCA_, M_ = [np.zeros(X_cp.shape)]* (t_len + 1) , [np.zeros(X_cp.shape)]* (t_len + 1) , [np.zeros(X_cp.shape)]* (t_len + 1)
    X_t, RCA_t, M_t = [np.zeros(X_cp.shape)]* (t_len + 1) , [np.zeros(X_cp.shape)]* (t_len + 1) , [np.zeros(X_cp.shape)]* (t_len + 1)

    for i in range(t_len):
        X_cp_i = X_cpt[:, :, i]
        RCA_i, M_i = Matrices(X_cp_i, threshold)

        X_[i], RCA_[i], M_[i] = [X_cp_i, RCA_i, M_i]
    X_[-1], RCA_[-1], M_[-1] = [X_cp, RCA, M]

    for i in range(t_len + 1):
        X_t[i] = array_shuffler(X_[i], shuffle_c, shuffle_p)
        RCA_t[i] = array_shuffler(RCA_[i], shuffle_c, shuffle_p)
        M_t[i] = array_shuffler(M_[i], shuffle_c, shuffle_p)


    llave_c = list(diccionario[0].keys())
    llave_p = list(diccionario[1].keys())
    Nuevo_dict_c_num = dict([(llave_c[shuffle_c[i][1]], i) for i in range(c_len)])
    Nuevo_dict_p_num = dict([(llave_p[shuffle_p[i][1]], i) for i in range(p_len)])


    diccionario[0] = Nuevo_dict_c_num
    diccionario[1] = Nuevo_dict_p_num

    return np.stack(X_t, axis = -1), np.stack(RCA_t, axis = -1), np.stack(M_t, axis = -1)

def Similaridad(M):
    '''De una matriz de especialización binaria, obtiene la metrica de similaridad definida en Hidalgo et al 2009 entre actividades. Mantiene la diagonal igual a cero.'''
    c_len, p_len, t_len = M.shape
    phi_t = np.stack([np.zeros((p_len, p_len))] * (t_len + 1), axis = -1)
    for t in range(t_len):
        ubicuidad = np.sum(M[:, :, t], axis = 0)  # p

        for p in range(p_len):
            for q in range(p_len):
                Maximo = np.max([ubicuidad[p], ubicuidad[q]])
                if p != q and Maximo != 0:
                    S = 0
                    for c in range(c_len):
                        S += M[c, p, t] * M[c, q, t]
                    phi_t[p, q, t] = S / Maximo
    return phi_t

def Similarity_Density(RCA):
    '''Toma la matriz RCA y calcula su Densidad de Similaridad calculando la Similaridad y matriz M_cp'''
    M_cp = 1 * (RCA >= 1)
    phi = Similaridad(M_cp)
    Num = np.matmul(M_cp, phi) / np.sum(phi, axis = 0)
    return Num

def Z_transf(K):
    '''Aplica la transformada Z sobre un vector K'''
    return (K - np.mean(K)) / np.std(K)

def Reflextion_method(M_cpt, n):
    '''Toma la matriz de especialización binaria y aplica el metodo de las reflexiones n veces devolviendo el vector de las iteración de las localidades y los productos'''
    c_len, p_len, t_len = M_cpt.shape

    eci_t = np.zeros((c_len, t_len))
    pci_t = np.zeros((p_len, t_len))
    for t in range(t_len):
        M_cp = M_cpt[:, :, t]

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

        s1 = np.sign(np.corrcoef(k_c0, k_cN)[0, 1])
        eci_t[:, t] = Z_transf(s1 * k_cN)
        pci_t[:, t] = Z_transf(s1 * k_pN)

    return (eci_t, pci_t)



def Eigen_method(M_cpt, last = False):
    '''Codigo extraido del modulo de ecomplexity'''
    if last:
        M_cpt = (M_cpt[:, :, -1])[:, :, np.newaxis]

    c_len, p_len, t_len = M_cpt.shape

    eci_t = np.zeros((c_len, t_len))
    pci_t = np.zeros((p_len, t_len))

    for t in range(t_len):
        M_cp = M_cpt[:, :, t]

        M_c = np.sum(M_cp, axis = 1)
        M_p = np.sum(M_cp, axis = 0)

        M_cp1 = M_cp / M_c[:, np.newaxis]
        M_cp2_t = (M_cp / M_p[np.newaxis, :]).T.copy()

        Mcc = M_cp1 @ M_cp2_t
        Mpp = M_cp2_t @ M_cp1


        eigvals, eigvecs = np.linalg.eig(Mpp)
        eigvecs = np.real(eigvecs)
        # Get eigenvector corresponding to second largest eigenvalue
        eig_index = eigvals.argsort()[-2]
        kp = eigvecs[:, eig_index]
        kc = M_cp1 @ kp

        s1 = np.sign(np.corrcoef(M_c, kc)[0, 1])
        eci_t[:, t] = Z_transf(s1 * kc)
        pci_t[:, t] = Z_transf(s1 * kp)

    return (eci_t, pci_t)




