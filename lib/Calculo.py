import numpy as np
import lib.Tratamiento as trat
from scipy.stats import pearsonr

def Limpieza(X, diccionarios, diccionario_pop, pop_min = 1_000_000, c_min = 0, p_min = 0):
    '''
    Args:
        X: numpy array. Es una matriz de volumen (c, p, t).
        diccionarios: list[dict]. Son los diccionarios de (c, p, t).
        diccionario_pop: dict. Es el diccionario entre paises y poblacion.
        pop_min: int. Umbral mínimo de población para un país.
        c_min: int. Umbral mínimo para que un país tenga producción.
        p_min: int. Umbral mínimo para que un producto tenga producción.
    '''
    Mask_pop = []
    for i in range(len(diccionarios[0])):
        try:
            if diccionario_pop[trat.inv_dict(diccionarios[0])[i]] > pop_min:
                Mask_pop.append(True)
            else:
                Mask_pop.append(False)
        except:
            Mask_pop.append(True)

    Mask_pop = np.array(Mask_pop)

    X_sum = np.copy(X).sum(axis = 2)
    Mask_c = (X_sum.sum(axis = 1) > c_min) * Mask_pop
    Mask_p = (X_sum.sum(axis = 0) > p_min)

    for n, valor in enumerate(Mask_c):
        if not valor:
            diccionarios[0].pop(trat.inv_dict(diccionarios[0])[n])
    for n, valor in enumerate(Mask_p):
        if not valor:
            diccionarios[1].pop(trat.inv_dict(diccionarios[1])[n])

    diccionarios[0] = trat.re_count(diccionarios[0])
    diccionarios[1] = trat.re_count(diccionarios[1])

    return X[:, Mask_p, :][Mask_c, :, :]


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


def Matrices_ordenadas(X_cpt, diccionario, diccionario_pop, pop_min = 1_000_000, c_min = 0, p_min = 0, threshold = 1, cleaning = True):
    '''
        Args:
            X_cpt: numpy array. Matriz de volumen de producción en el tiempo
            diccionario: list[dict]. Diccionarios para cada indice de la matriz de producción.
            diccionario_pop: dict: Diccionario entre pais y poblacion
            c_min: int. Premios promedio por país mínimo, es un umbral.
            p_min: int. Cantidad de países promedio por categoría, es un umbral.
            threshold: float. Umbral para calcular RCA.
            cleaning: Bool. Es para limpiar países Y categorías segun c_min y p_min.

        Returns:
            X_cpt: numpy array. Matriz de volumen de producción en el tiempo.
            RCA_cpt: numpy array. Matriz RCA en el tiempo.
            M_cpt: numpy array. Matriz de especialización binaria.

        A todas las matrices se les ordena por ubicuidad y diversidad, así como se les añade un índice extra de tiempo que contiene la matriz promediada.
    '''
    if cleaning:
        X_cpt = Limpieza(X_cpt, diccionario, diccionario_pop, pop_min, c_min, p_min)

    c_len, p_len, t_len = X_cpt.shape
    X_cp = X_cpt.sum(axis = 2)

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

def Relatedness(M, last = True):
    '''De una matriz de especialización binaria, obtiene la metrica de similaridad definida en Hidalgo et al 2009 entre actividades. Mantiene la diagonal igual a cero.'''

    M = M.astype(float)
    if last:
        M = (M[:, :, -1])[:, :, np.newaxis]

    c_len, p_len, t_len = M.shape
    phi_t = np.stack([np.zeros((p_len, p_len))] * (t_len), axis = -1)
    for t in range(t_len):
        M_esimo = M[:, :, t]
        producto = M_esimo.T @ M_esimo
        ubicuidad = np.sum(M_esimo, axis = 0)

        a, b = np.meshgrid(ubicuidad, ubicuidad)
        mask = (a > b)
        result = a * mask + b * (1 - mask)

        division = np.divide(producto, result, out = np.zeros_like(producto), where = (result != 0))
        phi_t[:, :, t] = division - np.diag(np.diag(division))
    return phi_t

def relatedness_density(M_cp, last = True):
    '''Toma la matriz RCA y calcula su Densidad de Similaridad calculando la Similaridad y matriz M_cp'''
    if last:
        M_cp = (M_cp[:, :, -1])[:, :, np.newaxis]

    c_len, p_len, t_len = M_cp.shape
    omega_t = np.zeros((c_len, p_len, t_len))
    phi_t = Relatedness(M_cp, last = last)
    for t in range(t_len):
        M = M_cp[:, :, t]
        phi = phi_t[:, :, t]
        phi_q = np.sum(phi, axis = 0)
        omega_t[:, :, t] = np.divide(np.matmul(M, phi), phi_q, out = np.zeros_like(M), where = (phi_q != 0))
    return omega_t

def Z_transf(K):
    '''Aplica la transformada Z sobre un vector K'''
    return (K - np.mean(K)) / np.std(K)

def Reflextion_method(M_cpt, n, last = False):
    '''DEPRECATED'''
    if last:
        M_cpt = (M_cpt[:, :, -1])[:, :, np.newaxis]

    c_len, p_len, t_len = M_cpt.shape

    eci_t = np.zeros((c_len, t_len))
    pci_t = np.zeros((p_len, t_len))
    for t in range(t_len):
        M = M_cpt[:, :, t]

        diversity = M.sum(axis=1)
        ubiquity = M.sum(axis=0)

        cntry_mask = np.argwhere(diversity == 0).squeeze()
        prod_mask = np.argwhere(ubiquity == 0).squeeze()
        k_c0 = diversity[diversity != 0][:, np.newaxis]
        k_p0 = ubiquity[ubiquity != 0][np.newaxis, :]
        M_cp = M[diversity != 0, :][:, ubiquity != 0]

        M_cp_1 = M_cp/k_c0
        M_cp_2 = (M_cp/k_p0).T

        M_pp = M_cp_2 @ M_cp_1
        M_cc = M_cp_1 @ M_cp_2

        c, p = M_cp.shape

        k_cN = k_c0
        k_pN = k_p0
        for j in range(n):
            k_cN = M_cc @ k_cN
            k_pN = k_pN @ M_pp

        s = np.sign(np.corrcoef(k_c0, k_cN)[0, 1])

        eci = Z_transf(s * k_cN)[:, 0]
        pci = Z_transf(s * k_pN)[0, :]

        for x in cntry_mask:
            eci = np.insert(eci, x, np.nan)
        for x in prod_mask:
            pci = np.insert(pci, x, np.nan)

        eci_t[:, t] = eci
        pci_t[:, t] = pci
    return (eci_t, pci_t)

def Eigen_method(M_cpt, last = False):
    '''Codigo extraido del modulo de ecomplexity'''
    if last:
        M_cpt = (M_cpt[:, :, -1])[:, :, np.newaxis]

    c_len, p_len, t_len = M_cpt.shape

    eci_t = np.zeros((c_len, t_len))
    pci_t = np.zeros((p_len, t_len))

    for t in range(t_len):
        M = M_cpt[:, :, t]

        diversity = M.sum(axis=1)
        ubiquity = M.sum(axis=0)

        cntry_mask = np.argwhere(diversity == 0).flatten()
        prod_mask = np.argwhere(ubiquity == 0).flatten()
        k_c0 = diversity[diversity != 0][:, np.newaxis]
        k_p0 = ubiquity[ubiquity != 0][np.newaxis, :]
        M_cp = M[diversity != 0, :][:, ubiquity != 0]

        M_cp1 = M_cp / k_c0
        M_cp2_t = (M_cp / k_p0).T.copy()

        Mcc = M_cp1 @ M_cp2_t
        Mpp = M_cp2_t @ M_cp1


        eigvals, eigvecs = np.linalg.eig(Mpp)
        eigvecs = np.real(eigvecs)
        # Get eigenvector corresponding to second largest eigenvalue
        eig_index = eigvals.argsort()[-2]
        kp = eigvecs[:, eig_index]
        kc = M_cp1 @ kp

        s1 = np.sign(np.corrcoef(k_c0[:, 0], kc)[0, 1])
        eci = Z_transf(s1 * kc)
        pci = Z_transf(s1 * kp)

        if len(cntry_mask) != 0:
            for x in cntry_mask:
                eci = np.insert(eci, x, np.nan)
        if len(prod_mask) != 0:
            for x in prod_mask:
                pci = np.insert(pci, x, np.nan)

        eci_t[:, t] = eci
        pci_t[:, t] = pci

    return (eci_t, pci_t)




