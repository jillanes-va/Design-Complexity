import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import linregress
from matplotlib.colors import rgb2hex

import distinctipy
import textalloc as ta
import lib.Tratamiento as trat

plt.style.use(['default'])


def graf(X, xlabel = '', ylabel = '', save = False, name = '', title = ''):
    plt.figure(figsize = (7,7))
    plt.imshow(X, interpolation = 'nearest')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save and len(name) != 0:
        plt.savefig('./figs/' + name + '.pdf')
        plt.close()
    else:
        plt.show()

def Clustering(phi, metodo  = 'complete', save = False, name = ''):
    plt.figure(figsize = (7,7))
    phi_star = np.zeros(phi.shape)
    linkage_matrix = linkage(squareform(1 - phi - np.identity(phi.shape[0])), method = metodo)
    indexes = leaves_list(linkage_matrix)

    for i,i_c in enumerate(indexes[::]):
        for j,j_c in enumerate(indexes[::]):
            phi_star[i,j] = phi[i_c, j_c]

    plt.imshow(phi_star)
    plt.title(r'$\phi_{pq}$')
    plt.xlabel('Categoria')
    plt.ylabel('Categoria')

    if save and len(name) != 0:
        plt.savefig('./figs/' + name + '.pdf')
        plt.close()
    else:
        plt.show()

def Density_plot(domain, prob, param = ['', '', ''], label = '', xlim_sup = 0, save = False, name = ''):
    plt.bar(domain, prob, width=1 / len(domain), align='edge', label = label)
    if xlim_sup != 0:
        plt.xlim([-0.05, xlim_sup])

    xlabel, ylabel, title = param
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    if save and len(name) != 0:
        plt.savefig('./figs/' + name + '.pdf')
        plt.close()
    else:
        plt.show()

def k_density(phi, bins = 30, save = False, name = ''):
    new_phi = phi.reshape(-1)
    plt.hist(new_phi, bins = bins, density = True)
    plt.xlabel('Relatedness')
    plt.ylabel('Densidad de Probabilidad')
    if save:
        plt.savefig(r'./figs/' + name + '.pdf')
        plt.close()
    else:
        plt.show()

def red(phi, by_com = True, diccionario =None, PCI = None, umbral_enlace = 0.5, save = False, name = ''):
    Red_original = nx.from_numpy_array(phi)
    Red_nueva = nx.maximum_spanning_tree(Red_original)


    pesos_red = []
    for enlace in list(Red_original.edges.data()):
        peso = enlace[2]['weight']
        if peso > umbral_enlace:
            Red_nueva.add_edge(enlace[0], enlace[1], weight = peso)
        pesos_red.append(peso)
    k_degree = np.sum([2 for enlaces in Red_nueva.edges])/Red_nueva.number_of_nodes()
    print(k_degree) #Printea el grado promedio de la red sin peso.
    print( (phi[phi > umbral_enlace].sum() / phi.sum())*100, '%' ) #Contea la reconexion del top %


    pesos = np.array([enlace[2]['weight'] for enlace in Red_nueva.edges.data()])
    posicion_red = nx.kamada_kawai_layout(Red_nueva, weight = 'weight')
    pesos = 2 * smooth(pesos)
    fig, ax = plt.subplots()
    if by_com:
        comunidades = nx.community.greedy_modularity_communities(Red_nueva)
        n = len(comunidades)
        n_comm = np.arange(n)
        coloracion = get_cmap(n_comm, N=n)[0]

        for comunidad, c, c_o in zip(comunidades, coloracion, coloracion):
            color = rgb2hex(c)
            color_oscuro = rgb2hex(c_o * np.array([0.7, 0.7, 0.7, 1]))
            nx.draw_networkx_nodes(Red_original, pos=posicion_red, nodelist=comunidad, node_color= color_oscuro, node_size=55)
            nx.draw_networkx_nodes(Red_original, pos=posicion_red, nodelist=comunidad, node_color = color, node_size = 25)
    else:
        coloracion, barra = get_cmap(PCI)
        nx.draw_networkx_nodes(Red_nueva, pos=posicion_red, node_size = 55, node_color=coloracion)
        plt.colorbar(barra, ax=plt.gca(), label = 'Product Complexity Index')

    nx.draw_networkx_edges(Red_nueva, pos = posicion_red, width = pesos, alpha = 0.3)
    #nx.draw_networkx_labels(Red_original, pos = posicion_red, font_size=8)


    if save and len(name) != 0:
        plt.axis('off')
        plt.savefig('./figs/' + name + '.pdf', transparent = True, bbox_inches = 'tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def get_cmap(PCI, name='inferno', N = None, pastel = 0.0):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    normalizacion = plt.Normalize(vmin = PCI.min(), vmax = PCI.max()+1)
    if N is None:
        cmap = plt.get_cmap(name)
    else:
        cmap = distinctipy.get_colormap(distinctipy.get_colors(N, pastel_factor= pastel))
    lista = np.array([cmap(normalizacion(valores)) for valores in PCI])
    colorbar = plt.cm.ScalarMappable(cmap = cmap, norm = normalizacion)
    colorbar.set_array([])
    return lista, colorbar

def get_cmap_evenly(N, name='inferno'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    normalizacion = plt.Normalize(vmin = 0, vmax = N - 1)
    cmap = plt.get_cmap(name)
    lista = np.array([cmap(normalizacion(valores)) for valores in range(N)])
    return lista

def smooth(x,n=0.5):
    y = ( x - np.min(x)) / (np.max(x) - np.min(x))
    return (y + 0.35)/1.15

def equal_vectors(comunidades):
    np.random.seed(None)
    print(np.random.get_state()[1][0])
    r = 0.3
    N = len(comunidades)
    diff = 2 * np.pi/(N)
    centro_comm = np.zeros((N, 2))
    nodes_pos = {}
    for i in range(0, N - 1):
        centro_comm[i +1,:] = np.random.random(), np.random.random()
    for n, comunidad in enumerate(comunidades):
        for nodo in comunidad:
            posicion = r*centro_comm[n, :] + 0.03* np.random.normal(size = 2)
            nodes_pos.update({nodo : posicion})

    print('')

    return nodes_pos

def scatter_lm(Z, listado = [], log = False, param = ['', '', ''], save = False, name = ''):
    fig, ax = plt.subplots(figsize=(6, 4))
    X = Z[:, 0]
    Y = Z[:, 1]
    if log:
        Y = np.log(Y)

    mask =  ~np.isnan(X) & ~np.isnan(Y)
    X = X[mask]
    Y = Y[mask]
    listado_arreglado = [listado[i] for i in range(len(mask)) if (mask[i] == True)]

    b_1, b_0, r, p, se = linregress(X, Y)

    ax.scatter(X, Y, color='tab:blue', s=4 ** 2)
    if len(listado) != 0:
        ta.allocate(
            ax, X, Y,
            listado_arreglado, x_scatter = X, y_scatter = Y, textsize=6,
            draw_lines=False
        )

    reg_x = np.array([np.min(X), np.max(X)])
    reg_y = b_1 * reg_x + b_0

    ax.plot(reg_x, reg_y, color='tab:red', alpha=0.6,
            label=f'm={b_1:.3f}\np={p:.3f}\n' + r'$r^2$' + f'={r ** 2:.3f}', linestyle='--')

    xlabel, ylabel, title = param
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.legend(loc='lower right')
    if save:
        plt.savefig(r'./figs/' + name + '.pdf')
        plt.close()
    else:
        plt.show()