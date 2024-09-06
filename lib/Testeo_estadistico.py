import numpy as np
import matplotlib.pyplot as plt
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

def Relatedness(RCA, threeshold = 0.5, n_t = None):
    M_cp = np.ones(RCA.shape) * (RCA >= 1)
    phi_0 = calc.Similaridad(M_cp)
    Transicion, _ = Trans_matrix(RCA, threeshold, n_t)
    omega_cp_t = np.nan_to_num( np.matmul(M_cp, phi_0) * Transicion / np.sum(phi_0, axis = 0) )

    cantidad_paises, cantidad_categorias = RCA.shape

    N_bins = 50

    Ocurrencias_t = np.zeros(N_bins)
    Total = np.zeros(N_bins)

    for c in range(cantidad_paises):
        for p in range(cantidad_categorias):

            phi_max = (M_cp[c, :] * phi_0[p, :]).max()
            rebanada = int(np.floor(phi_max * N_bins))
            if rebanada == N_bins:
                rebanada += -1
            Total[rebanada] += 1

            if Transicion[c, p] == 1:
                Ocurrencias_t[rebanada] += 1

    Prob_t = np.nan_to_num(Ocurrencias_t / Total)

    dom_phi = np.linspace(0, 1, N_bins)

    plt.bar(dom_phi, Prob_t, width=1 / N_bins, align='edge')
    plt.xlabel(r'$\phi$')
    plt.ylabel('%')