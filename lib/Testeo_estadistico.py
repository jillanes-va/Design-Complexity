import numpy as np
import lib.Calculo as calc

def Trans_matrix(RCA_t, threeshold = 0.5, n_t = None):
    C, P, T = RCA_t.shape
    Transicion = np.zeros((C, P))
    Intransicion = np.zeros((C, P))

    if n_t is None:
        n_t = T//2

    R_cp_0 = RCA_t[:, :, :n_t].sum(axis = 2) / (n_t + 1)
    R_cp_1 = RCA_t[:, :, n_t + 1:].sum(axis = 2) / (T - n_t)

    for c in range(C):
        for p in range(P):
            if R_cp_0[c,p] <= threeshold:
                if R_cp_1[c,p] >=1:
                    Transicion[c,p] = 1
                if R_cp_1[c,p] <=threeshold:
                    Intransicion[c,p] = 1
    return Transicion, Intransicion

def Principio
omega_cp_t = np.nan_to_num( np.matmul(M_cp_0, phi_pq_0) * Transicion / np.sum(phi_pq_0, axis = 0) )
omega_cp_i = np.nan_to_num( np.matmul(M_cp_0, phi_pq_0) * Intransicion / np.sum(phi_pq_0, axis = 0) )