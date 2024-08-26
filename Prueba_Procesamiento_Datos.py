import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#%% Cargamos los datos y limpiamos los NaNs

str_archivo = r'G:/Mi unidad/Estudio/Ciencias Físicas/Tesis/wrd_04_all-data.csv'

data = pd.read_csv(str_archivo)
data_sin_nan = data.loc[data['award_category'].notna() & data['designer_country'].notna() & data['award_period'].notna()]

print(data_sin_nan.columns) #Printea todas las columnas
#%% Definimos los datos de interes

datos_nombre_diseñador = data_sin_nan['designer_name'].values #En formato np.array
datos_pais_diseñador = data_sin_nan['designer_country'].values
datos_categoria_premio = data_sin_nan['award_category'].values
datos_periodo_premio = data_sin_nan['award_period'].values
datos_puntaje_premio = data_sin_nan['award_score'].values

N = len(datos_nombre_diseñador) #Cantidad de datos

#%% Funciones utiles

def Ordenamiento(RCA, M, lista_categorias, lista_paises):
    '''Funcion que toma una matriz de especialización y la reordena por ubicuidad... y entrega la matriz reordenada, con el diccionario correspondiente'''
    M_p = np.sum(M, axis = 0)
    M_c = np.sum(M, axis = 1)
    
    N_c, N_p = M.shape
    
    lista_p = [(M_p[i], i) for i in range(N_p)]
    lista_c = [(M_c[i], i) for i in range(N_c)]
    
    Shuffle_p = sorted(lista_p, key = lambda A:A[0], reverse = 1)
    Shuffle_c = sorted(lista_c, key = lambda A:A[0], reverse = 1)
    
    
    M_ordenada = np.zeros(M.shape)
    RCA_ordenada = np.zeros(RCA.shape)
    
    for i in range(N_c):
        for j in range(N_p):
            M_ordenada[i,j] = M[ Shuffle_c[i][1] , Shuffle_p[j][1] ]
            RCA_ordenada[i,j] = RCA[ Shuffle_c[i][1] , Shuffle_p[j][1] ]
    
    Nuevo_dict_num_p = dict([ (i, lista_categorias[Shuffle_p[i][1]]) for i in range(N_p) ])
    Nuevo_dict_p_num = dict([ (lista_categorias[Shuffle_p[i][1]], i) for i in range(N_p) ])
    Nuevo_dict_num_c = dict([ (i, lista_paises[Shuffle_c[i][1]]) for i in range(N_c) ])
    Nuevo_dict_c_num = dict([ (lista_paises[Shuffle_c[i][1]], i) for i in range(N_c) ])
    
    return RCA_ordenada, M_ordenada, Nuevo_dict_c_num, Nuevo_dict_p_num, Nuevo_dict_num_c, Nuevo_dict_num_p

def Matrices_interesantes(X):
    '''Función que toma una matriz de volumen de país-producción y devuelve la RCA y la matriz de especialización binaria'''
    c_len, p_len = X.shape
    RCA = np.zeros(X.shape)
    M = np.zeros(X.shape)
    
    alpha = np.zeros(X.shape)
    beta = np.zeros(X.shape)
    
    X_c = np.sum(X, axis = 1)
    X_p = np.sum(X, axis = 0)
    X_net = np.sum(X)
    
    for i in range(c_len):
        for j in range(p_len):
            if X_c[i] != 0:
                alpha[i,j] = X[i,j] / X_c[i]
            beta[i,j] = X_p[j] / X_net
            if X_p[j] != 0:
                RCA[i,j] = alpha[i,j] / beta[i,j]

    for i in range(c_len):
        for j in range(p_len):
            if RCA[i, j] >= 1:
                M[i,j] = 1
    return RCA, M, alpha, beta

def Similaridad(M):
    '''De una matriz de especialización binaria, obtiene la metrica de similaridad definida en Hidalgo et al 2009 entre actividades. Mantiene la diagonal igual a cero.'''
    c_len, p_len = M.shape
    phi = np.zeros((p_len, p_len))
    ubicuidad = np.sum(M, axis = 0) #p

    for p in range(p_len):
        for q in range(p_len):
            Maximo = np.max([ubicuidad[p], ubicuidad[q]])
            if p != q and Maximo != 0:
                S = 0
                for c in range(c_len):
                    S += M[c, p] * M[c, q]
                phi[p,q] = S / Maximo
    return phi

def Complexity_measures(M, n):
    '''Toma la matriz de especialización binaria y aplica el metodo de las reflexiones n veces devolviendo el vector de las iteración de las localidades y los productos'''
    M_c = np.sum(M, axis = 1)
    M_p = np.sum(M, axis = 0)
    
    K_c = M_c
    K_p = M_p
    for i in range(n):
        K_c = np.matmul(M, K_p) / M_c
        K_p = np.matmul(M.T, K_c) / M_p
    return K_c, K_p

def Z_transf(K):
    '''Aplica la transformada Z sobre un vector K'''
    return (K - np.mean(K))/np.std(K)
#%% Obtención de categorias, paises y nombres, y su cantidad, y un diccionario asociado

lista_nombres = []
lista_categorias =[]
lista_paises = []
lista_periodos = []
lista_puntajes = []

for nombre in datos_nombre_diseñador:
    if not nombre in lista_nombres:
        lista_nombres.append(nombre)

for categoria in datos_categoria_premio:
    if not categoria in lista_categorias:
        lista_categorias.append(categoria)

for pais in datos_pais_diseñador:
    if not pais in lista_paises:
        lista_paises.append(pais)

for periodo in datos_periodo_premio:
    if not periodo in lista_periodos:
        lista_periodos.append(periodo)

for puntaje in datos_puntaje_premio:
    if not puntaje in lista_puntajes:
        lista_puntajes.append(puntaje)

lista_periodos = sorted(lista_periodos) #ordenamos el listado por año
lista_puntajes = sorted(lista_puntajes) #ordenamos el listado por puntaje

cantidad_nombres = len(lista_nombres)
cantidad_categorias = len(lista_categorias)
cantidad_paises = len(lista_paises)
cantidad_periodos = len(lista_periodos)
cantidad_puntajes = len(lista_puntajes)

dict_nombres_num = dict([(lista_nombres[i], i) for i in range(cantidad_nombres)])
dict_categorias_num = dict([(lista_categorias[i], i) for i in range(cantidad_categorias)])
dict_paises_num = dict([(lista_paises[i], i) for i in range(cantidad_paises)])
dict_periodos_num = dict([(lista_periodos[i], i) for i in range(cantidad_periodos)])
dict_puntajes_num = dict([(lista_puntajes[i], i) for i in range(cantidad_puntajes)])

dict_num_nombres = dict([(i, lista_nombres[i]) for i in range(cantidad_nombres)])
dict_num_categorias = dict([(i, lista_categorias[i]) for i in range(cantidad_categorias)])
dict_num_paises = dict([(i, lista_paises[i]) for i in range(cantidad_paises)])
dict_num_periodos = dict([(i, lista_periodos[i]) for i in range(cantidad_periodos)])
dict_num_puntajes = dict([(i, lista_puntajes[i]) for i in range(cantidad_puntajes)])

#%% Creamos las matrices de... eso mismo

X_npct = np.empty((cantidad_nombres, cantidad_paises, cantidad_categorias, cantidad_periodos))


for j in range(N):
    n = dict_nombres_num[ datos_nombre_diseñador[j] ]
    p = dict_paises_num[ datos_pais_diseñador[j] ]
    c = dict_categorias_num[ datos_categoria_premio[j] ]
    t = dict_periodos_num[ datos_periodo_premio[j] ]
    X_npct[n, p, c, t] += datos_puntaje_premio[j]
            


#%% veamos la cantidad de datos por año

año_inicial = np.array([int(i[:4]) for i in lista_periodos] + ['2023'])
cantidad_datos_año = np.array([len(data_sin_nan.loc[ data['award_period'] == i ]) for i in lista_periodos]) / N

fig, ax = plt.subplots() 
  
n, bins, patches = ax.hist(sorted(datos_periodo_premio), 12, 
                            density = 0,  
                            color ='green',  
                            alpha = 0.7,
                            align = 'mid',
                            edgecolor = 'black') 

ax.set_xticks([i for i in range(14)], año_inicial);
ax.tick_params(axis = 'x', labelrotation = 90);
ax.set_xlabel('Años');
ax.set_ylabel('Frecuencia');
ax.set_title('Cantidad de datos por año');

#%% Promediando los datos. X es la matriz de volumen de paises - categorias de premios

X_npc = np.sum(X_npct, axis = 3) / cantidad_periodos #Promediar en el tiempo
X_cp = np.sum(X_npc, axis = 0)


#%% R es la matriz de ventaja comparativa y M es la matriz de especialiación binaria

R_cp, M_cp, _, _ = Matrices_interesantes(X_cp)
R_cp, M_cp, dict_paises_num, dict_categorias_num, dict_num_paises, dict_num_categorias = Ordenamiento(R_cp, M_cp, lista_categorias, lista_paises)
#%% Graficamos M_cp

plt.imshow(np.log10(R_cp + 1), interpolation = 'nearest', cmap = 'afmhot')
plt.xlabel('Productos') #debería decir categorias
plt.ylabel('Países')
plt.title(r'Especialización Binaria');
plt.tight_layout()

#%% Metricas de similaridad

phi_pq = Similaridad(M_cp)

diversidad = np.sum(M_cp, axis = 1) #c
ubicuidad = np.sum(M_cp, axis = 0) #p


#%% Grafico Matriz_pp sin diagonal

plt.imshow(phi_pq, interpolation = 'nearest')
plt.xlabel('Categoria')
plt.ylabel('Categoria')
plt.title(r'$\phi_{pq}$');

#%% Espacio de diseño

Design_space = nx.from_numpy_array(phi_pq)
Design_space_tree = nx.maximum_spanning_tree(Design_space)

Enlaces_arbol = {(u, v, d["weight"]) for (u, v, d) in Design_space_tree.edges(data=True)}
Enlaces_grandes = {(u, v, d["weight"]) for (u, v, d) in Design_space.edges(data=True) if d["weight"] >= 0.42} #Enlaces mayores a 0.4

Enlaces_pesados = list(Enlaces_arbol | Enlaces_grandes)

Anchos = np.array([d for (u, v, d) in Enlaces_pesados])
Enlaces = np.array([(u,v) for (u, v, d) in Enlaces_pesados])

Design_degree = np.array([Design_space_tree.degree(n) for n in Design_space.nodes()])

posicion = nx.kamada_kawai_layout(Design_space_tree) #Tengo que ver el layout
cmap = plt.cm.viridis #Y la elección de colores

nx.draw_networkx_nodes(Design_space_tree, pos = posicion, node_size = 10 * Design_degree)
nx.draw_networkx_edges(Design_space, pos = posicion, edge_color = Anchos, edge_cmap = cmap, edgelist = Enlaces, width = 0.5);

#%% Calculo del ECI y PCI

K_c, K_p = Complexity_measures(M_cp, 18)
ECI, PCI = Z_transf(K_c), Z_transf(K_p)

#%% Correlacion negativa entre la diversidad y k_c,1 
plt.scatter(diversidad, Complexity_measures(M_cp, 1)[0]);
plt.xlabel(r'$k_0$')
plt.ylabel(r'$k_1$')

#%% Veamos el principio de similaridad. Calculamos las dos matrices de volumen
X_cpt = np.sum(X_npct, axis = 0)

delta = 0

X_cp_0 = np.sum(X_cpt[:,:,5:9], axis = 2) / (4)
X_cp_1 = np.sum(X_cpt[:,:,9:12], axis = 2) / (3)

#%% Las dos RCA y M

R_cp_0, M_cp_0, alpha_0, beta_0 = Matrices_interesantes(X_cp_0)
R_cp_1, M_cp_1, _, _ = Matrices_interesantes(X_cp_1)

phi_pq_0 = Similaridad(M_cp_0)
phi_pq_1 = Similaridad(M_cp_1)

#%%

Transicion = np.zeros(R_cp_0.shape)
Intransicion = np.zeros(R_cp_0.shape)

for c in range(cantidad_paises):
    for p in range(cantidad_categorias):
        if R_cp_0[c,p] <= 0.5:
            if R_cp_1[c,p] >=1:
                Transicion[c,p] = 1
            if R_cp_1[c,p] <=0.5:
                Intransicion[c,p] = 1
#%% TODO ver como aplicar el principio de similaridad segun los papers

plt.subplots(1,3)

plt.subplot(1, 3, 1)
plt.imshow(phi_pq_0)

plt.subplot(1, 3, 2)
plt.imshow(phi_pq_1)

plt.subplot(1, 3, 3)
plt.imshow(phi_pq)

plt.tight_layout()

#%%

plt.subplots(1,2)

plt.subplot(1,2, 1)
plt.imshow(Transicion)

plt.subplot(1,2, 2)
plt.imshow(Intransicion)

plt.tight_layout()



#%%

omega_cp_t = np.nan_to_num( np.matmul(M_cp_0, phi_pq_0) * Transicion / np.sum(phi_pq_0, axis = 0) )
omega_cp_i = np.nan_to_num( np.matmul(M_cp_0, phi_pq_0) * Intransicion / np.sum(phi_pq_0, axis = 0) )

#%%
plt.subplots(2,1)

plt.subplot(2, 1, 1)
plt.hist(omega_cp_t[omega_cp_t >0], bins = 20)
plt.yscale('linear')
plt.title(r'Transicion')

plt.subplot(2, 1, 2)
plt.hist(omega_cp_i[omega_cp_i > 0], bins = 20)
plt.title(r'Intransicion');
plt.yscale('linear')
plt.tight_layout()

#%% Histograma similaridad

plt.subplots(3,1)

plt.subplot(3,1, 1)
histo_phi_0 = phi_pq_0[phi_pq_0 > 0]
plt.hist(histo_phi_0, bins = 20, density = 1)

plt.subplot(3,1, 2)
histo_phi = phi_pq_1[phi_pq_1 > 0]
plt.hist(histo_phi, bins = 20, density = 1)

plt.subplot(3,1, 3)
histo_phi = phi_pq[phi_pq > 0]
plt.hist(histo_phi, bins = 20, density = 1)

plt.tight_layout()

#%% Principio de similaridad?

N_bins = 50

Ocurrencias_t = np.zeros(N_bins)
Ocurrencias_i = np.zeros(N_bins)
Total = np.zeros(N_bins)


for c in range(cantidad_paises):
    for p in range(cantidad_categorias):
        
        phi_max = (M_cp_0[c,:] * phi_pq_0 [p,:]).max()
        rebanada = int(np.floor(phi_max * N_bins))
        if rebanada == N_bins:
            rebanada += -1
        Total[ rebanada ] += 1
        
        if Transicion[c, p] == 1:
            Ocurrencias_t[ rebanada ] += 1
        
        if Intransicion[c, p] == 1:
            Ocurrencias_i[ rebanada ] += 1

Prob_t = np.nan_to_num(Ocurrencias_t/ Total)
Prob_i = np.nan_to_num(Ocurrencias_i /Total)

dom_phi = np.linspace(0,1,N_bins)

plt.bar(dom_phi, Prob_t, width = 1/N_bins, align = 'edge')
plt.xlabel(r'$\phi$')
plt.ylabel('')
#%%

plt.hist(np.reshape(R_cp_0, 115*94),density = 0, bins = 1000, cumulative = 1)
plt.xlim([-2, 40])

#%%

